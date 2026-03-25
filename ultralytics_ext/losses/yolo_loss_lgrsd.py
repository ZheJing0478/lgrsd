from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from ultralytics.utils.loss import v8DetectionLoss

from ultralytics_ext.lgrsd.roi_pool import RoiAlignCfg, assign_fpn_levels, roi_align_by_levels


@dataclass(frozen=True)
class LGRSDCfg:
    enable: bool = False
    lambda_rsd: float = 0.5
    topk_per_image: int = 16
    crop_size: int = 224
    roi_size: int = 7
    feature_level: str = "auto"  # auto | P3-only
    embed_dim: int = 256
    # A4: small-box protection & crop context
    min_side_px: int = 16
    context_ratio: float = 1.3
    min_orig_side_px: int = 4
    min_area_ratio: float = 0.3
    # A5: sampling strategy
    sampling_strategy: str = "stratified_default"  # stratified_default | area_topk
    # Teacher update (optional): EMA teacher to reduce teacher-student drift.
    # 1.0 means "frozen teacher" (no update). Typical EMA values: 0.99~0.999.
    teacher_momentum: float = 1.0


class LGRSDHead(nn.Module):
    """Projection heads for ROI (student) and crop (teacher) embeddings."""

    def __init__(self, roi_in_dims: list[int], crop_in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.roi_proj = nn.ModuleList([nn.Linear(int(c), self.embed_dim) for c in roi_in_dims])
        self.crop_proj = nn.Linear(int(crop_in_dim), self.embed_dim)

    def proj_roi(self, lvl: int, x: torch.Tensor) -> torch.Tensor:
        z = self.roi_proj[lvl](x)
        return F.normalize(z, dim=-1, eps=1e-6)

    def proj_crop(self, x: torch.Tensor) -> torch.Tensor:
        z = self.crop_proj(x)
        return F.normalize(z, dim=-1, eps=1e-6)


def _get_lgrsd_cfg(args: Any) -> LGRSDCfg:
    return LGRSDCfg(
        enable=bool(getattr(args, "enable_lgrsd", False)),
        lambda_rsd=float(getattr(args, "lgrsd_lambda", 0.5)),
        topk_per_image=int(getattr(args, "lgrsd_topk_per_image", 16)),
        crop_size=int(getattr(args, "lgrsd_crop_size", 224)),
        roi_size=int(getattr(args, "lgrsd_roi_size", 7)),
        feature_level=str(getattr(args, "lgrsd_feature_level", "auto")),
        embed_dim=int(getattr(args, "lgrsd_embed_dim", 256)),
        min_side_px=int(getattr(args, "lgrsd_min_side_px", 16)),
        context_ratio=float(getattr(args, "lgrsd_context_ratio", 1.3)),
        min_orig_side_px=int(getattr(args, "lgrsd_min_orig_side_px", 4)),
        min_area_ratio=float(getattr(args, "lgrsd_min_area_ratio", 0.3)),
        sampling_strategy=str(getattr(args, "lgrsd_sampling_strategy", "stratified_default")),
        teacher_momentum=float(getattr(args, "lgrsd_teacher_momentum", 1.0)),
    )


@torch.no_grad()
def _ema_update_teacher(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    """Update teacher weights to track student with EMA (no gradients).

    We update only keys that exist in teacher.state_dict(). This automatically skips train-only heads
    like `lgrsd_head.*` that are not part of the detection backbone/neck.
    """
    m = float(momentum)
    if m >= 1.0:
        return
    if m < 0.0:
        raise ValueError(f"teacher_momentum must be >= 0, got {momentum}")

    tsd = teacher.state_dict()
    ssd = student.state_dict()
    for k, tv in tsd.items():
        sv = ssd.get(k)
        if sv is None:
            continue
        if tv.dtype.is_floating_point:
            tv.mul_(m).add_(sv, alpha=1.0 - m)
        else:
            tv.copy_(sv)


def _xywhn_to_xyxy_abs(xywhn: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Normalized xywh -> absolute xyxy (pixel coords)."""
    xywh = xywhn.clone()
    xywh[:, 0] *= float(img_w)
    xywh[:, 1] *= float(img_h)
    xywh[:, 2] *= float(img_w)
    xywh[:, 3] *= float(img_h)
    x, y, w, h = xywh.unbind(dim=1)
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return torch.stack([x1, y1, x2, y2], dim=1)


def _clip_xyxy(xyxy: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    xyxy = xyxy.clone()
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0.0, float(img_w))
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0.0, float(img_h))
    return xyxy


def _expand_xyxy_with_context(
    xyxy: torch.Tensor, *, img_w: int, img_h: int, context_ratio: float, min_side_px: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand each xyxy box by context_ratio and clamp min side, then clip to image.

    Returns:
      xyxy_ctx: [N,4] clipped expanded boxes
      area_ratio: [N] (clipped_area / expanded_area) used for skip rule
    """
    x1, y1, x2, y2 = xyxy.unbind(dim=1)
    w0 = (x2 - x1).clamp(min=1.0)
    h0 = (y2 - y1).clamp(min=1.0)
    xc = (x1 + x2) * 0.5
    yc = (y1 + y2) * 0.5

    w = (w0 * float(context_ratio)).clamp(min=float(min_side_px))
    h = (h0 * float(context_ratio)).clamp(min=float(min_side_px))
    x1e = xc - w * 0.5
    y1e = yc - h * 0.5
    x2e = xc + w * 0.5
    y2e = yc + h * 0.5
    expanded = torch.stack([x1e, y1e, x2e, y2e], dim=1)

    clipped = _clip_xyxy(expanded, img_w=img_w, img_h=img_h)
    expanded_area = (w * h).clamp(min=1.0)
    clipped_area = ((clipped[:, 2] - clipped[:, 0]).clamp(min=0.0) * (clipped[:, 3] - clipped[:, 1]).clamp(min=0.0)).clamp(min=0.0)
    area_ratio = clipped_area / (expanded_area + 1e-9)
    return clipped, area_ratio


def sample_boxes_for_lgrsd(
    *,
    batch_idx: torch.Tensor,
    area: torch.Tensor,
    batch_size: int,
    topk: int,
    strategy: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """A5: sample boxes per image using a small-object-friendly strategy.

    Args:
      batch_idx: [N] int64 image id per GT box
      area: [N] area in pixel^2 (based on original box, not expanded)
      batch_size: B
      topk: max boxes per image
      strategy: 'area_topk' or 'stratified_default'

    Returns:
      selected_idx: 1D long indices into the input arrays
      stats: dict with selected counts (averaged per image across this batch)
    """
    if batch_idx.numel() == 0:
        return batch_idx.new_zeros((0,), dtype=torch.long), {"skip_frac": 0.0, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}

    # COCO area thresholds in pixels^2
    thr_s = 32.0 * 32.0
    thr_m = 96.0 * 96.0

    def _bucket_counts(k: int) -> tuple[int, int, int]:
        if k <= 0:
            return 0, 0, 0
        if k == 16:
            return 8, 6, 2
        # scale 8/6/2 ~= 0.5/0.375/0.125
        s = max(1, int(round(k * 0.5)))
        m = max(1, int(round(k * 0.375)))
        l = max(0, k - s - m)
        if s + m + l > k:
            l = max(0, k - s - m)
        return s, m, l

    chosen_all: list[torch.Tensor] = []
    sel_s = 0
    sel_m = 0
    sel_l = 0
    for b in range(batch_size):
        idx = torch.nonzero(batch_idx == b, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue

        if strategy.lower() == "area_topk":
            # Deterministic: choose by area desc
            order = torch.argsort(area[idx], descending=True)
            pick = idx[order[: min(topk, idx.numel())]]
        else:
            # Stratified default: small/medium/large buckets, then fill remainder
            k_s, k_m, k_l = _bucket_counts(topk)
            a = area[idx]
            small = idx[a < thr_s]
            medium = idx[(a >= thr_s) & (a < thr_m)]
            large = idx[a >= thr_m]

            def _topk_by_area(i: torch.Tensor, k: int) -> torch.Tensor:
                if i.numel() == 0 or k <= 0:
                    return i.new_zeros((0,), dtype=torch.long)
                o = torch.argsort(area[i], descending=True)
                return i[o[: min(k, i.numel())]]

            pick_s = _topk_by_area(small, k_s)
            pick_m = _topk_by_area(medium, k_m)
            pick_l = _topk_by_area(large, k_l)
            picked = torch.cat([pick_s, pick_m, pick_l], dim=0)

            # Fill any leftover slots from remaining boxes (by area desc)
            if picked.numel() < min(topk, idx.numel()):
                remaining = idx[~torch.isin(idx, picked)]
                if remaining.numel():
                    o = torch.argsort(area[remaining], descending=True)
                    need = min(topk, idx.numel()) - picked.numel()
                    picked = torch.cat([picked, remaining[o[:need]]], dim=0)
            pick = picked

        chosen_all.append(pick)

        # per-bucket counts for logging (based on original area)
        a_pick = area[pick]
        sel_s += int((a_pick < thr_s).sum().item())
        sel_m += int(((a_pick >= thr_s) & (a_pick < thr_m)).sum().item())
        sel_l += int((a_pick >= thr_m).sum().item())

    selected_idx = torch.cat(chosen_all, dim=0) if chosen_all else batch_idx.new_zeros((0,), dtype=torch.long)

    denom = max(1, batch_size)
    stats = {
        "samp_s": float(sel_s) / float(denom),
        "samp_m": float(sel_m) / float(denom),
        "samp_l": float(sel_l) / float(denom),
    }
    return selected_idx, stats


def _run_embed(model: nn.Module, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Run model forward and return pooled embedding from given layer index.

    Uses Ultralytics BaseModel `embed` fast-path (stops at the requested layer).
    """
    out = model.predict(x, embed=[int(layer_idx)])
    if isinstance(out, (tuple, list)):
        return torch.stack(list(out), dim=0)
    return out


def ensure_lgrsd_modules(model: nn.Module) -> None:
    """Attach LG-RSD projection head to the model BEFORE optimizer creation."""
    if hasattr(model, "lgrsd_head"):
        return

    detect = model.model[-1]  # Detect()
    # Infer FPN channel dims from Detect head's first conv of each branch.
    # detect.cv2 is ModuleList[Sequential(Conv(x,c2), ...)], so cv2[i][0].conv.in_channels == x.
    try:
        roi_in_dims = [int(b[0].conv.in_channels) for b in detect.cv2]  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Cannot infer FPN channel dims from Detect head: {e}") from e

    args = getattr(model, "args", None)
    cfg = _get_lgrsd_cfg(args)
    # Teacher embedding is taken from the highest-stride FPN output (same dim as roi_in_dims[-1]).
    crop_in_dim = int(roi_in_dims[-1])
    head = LGRSDHead(roi_in_dims=roi_in_dims, crop_in_dim=crop_in_dim, embed_dim=cfg.embed_dim)
    device = next(model.parameters()).device
    head.to(device)
    setattr(model, "lgrsd_head", head)

    # Create a frozen teacher copy (same architecture, pretrained weights copied from student).
    # This is required for strict stop-grad verification: teacher params must have grad=None.
    if not hasattr(model, "lgrsd_teacher"):
        try:
            from ultralytics.nn.tasks import DetectionModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Cannot import DetectionModel for LG-RSD teacher construction: {e}") from e

        yaml_cfg = deepcopy(getattr(model, "yaml", {}))
        ch = int(getattr(model, "yaml", {}).get("channels", 3))
        nc = int(getattr(model, "yaml", {}).get("nc", 1))
        teacher = DetectionModel(yaml_cfg, ch=ch, nc=nc, verbose=False).to(device)
        teacher.load_state_dict(model.state_dict(), strict=False)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        # IMPORTANT: do NOT register teacher as a submodule.
        # - prevents counting teacher params in model.parameters() / cost tables
        # - prevents bloating checkpoints (teacher is train-time-only)
        model.__dict__["lgrsd_teacher"] = teacher

    # Store teacher layer index for crop embedding: use highest-stride feature feeding Detect.
    f = getattr(detect, "f", None)
    if isinstance(f, (list, tuple)) and f:
        setattr(model, "lgrsd_teacher_layer", int(f[-1]))
    else:
        # Fallback: backbone last layer for yolov8 is typically 9 (SPPF output).
        setattr(model, "lgrsd_teacher_layer", 9)


class v8DetectionLossWithRSD:
    """YOLOv8 detection loss + Local-Global Region Self-Distillation (LG-RSD)."""

    def __init__(self, model: nn.Module, tal_topk: int = 10) -> None:
        self.model = model
        self.base = v8DetectionLoss(model, tal_topk=tal_topk)

        # Ensure projection heads exist (must be attached before optimizer creation).
        ensure_lgrsd_modules(model)

        self.device = next(model.parameters()).device

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        base_loss, base_items = self.base(preds, batch)
        args = getattr(self.model, "args", None)
        cfg = _get_lgrsd_cfg(args)
        if not cfg.enable or cfg.lambda_rsd <= 0:
            return base_loss, base_items

        rsd, extra = self._compute_rsd(batch, cfg)
        bs = batch["img"].shape[0]
        # IMPORTANT: only losses are included in `loss` (used for backward). Extra metrics are appended to `loss_items`.
        loss = torch.cat([base_loss, (rsd * bs).unsqueeze(0)])
        extra_items = batch["img"].new_tensor(
            [
                float(extra.get("skip_frac", 0.0)),
                float(extra.get("samp_s", 0.0)),
                float(extra.get("samp_m", 0.0)),
                float(extra.get("samp_l", 0.0)),
            ]
        )
        items = torch.cat([base_items, rsd.detach().unsqueeze(0), extra_items.detach()])
        return loss, items

    def _compute_rsd(self, batch: dict[str, torch.Tensor], cfg: LGRSDCfg) -> tuple[torch.Tensor, dict[str, float]]:
        # Training-time only regularizer (skip in val to keep evaluation fast and clean).
        if not self.model.training:
            return batch["img"].new_zeros(()), {"skip_frac": 0.0, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}

        img = batch["img"]
        bs, _, img_h, img_w = img.shape

        batch_idx = batch["batch_idx"].view(-1).to(dtype=torch.long)
        bboxes = batch["bboxes"].view(-1, 4)

        if bboxes.numel() == 0:
            return img.new_zeros(()), {"skip_frac": 0.0, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}

        xyxy = _clip_xyxy(_xywhn_to_xyxy_abs(bboxes, img_w=img_w, img_h=img_h), img_w=img_w, img_h=img_h)
        w0 = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0.0)
        h0 = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0.0)
        valid = (w0 > 1.0) & (h0 > 1.0)
        if valid.sum() == 0:
            return img.new_zeros(()), {"skip_frac": 0.0, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}
        batch_idx = batch_idx[valid]
        xyxy = xyxy[valid]
        w0 = w0[valid]
        h0 = h0[valid]
        area0 = (w0 * h0).clamp(min=0.0)

        # A4: skip extremely small originals; apply context expansion + clip; skip if mostly outside.
        min_side0 = torch.minimum(w0, h0)
        keep_mask = min_side0 >= float(cfg.min_orig_side_px)
        xyxy = xyxy[keep_mask]
        batch_idx = batch_idx[keep_mask]
        area0 = area0[keep_mask]
        if xyxy.numel() == 0:
            return img.new_zeros(()), {"skip_frac": 1.0, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}

        xyxy_ctx, area_ratio = _expand_xyxy_with_context(
            xyxy, img_w=img_w, img_h=img_h, context_ratio=cfg.context_ratio, min_side_px=cfg.min_side_px
        )
        keep_mask2 = area_ratio >= float(cfg.min_area_ratio)
        candidate_total = int(valid.sum().item())
        kept_total = int(keep_mask.sum().item())  # after min_orig_side filter
        kept_total2 = int(keep_mask2.sum().item())  # after area_ratio filter
        skip_frac = 0.0
        if candidate_total > 0:
            skip_frac = float(candidate_total - kept_total2) / float(candidate_total)

        batch_idx = batch_idx[keep_mask2]
        xyxy_ctx = xyxy_ctx[keep_mask2]
        area0 = area0[keep_mask2]
        if xyxy_ctx.numel() == 0:
            return img.new_zeros(()), {"skip_frac": skip_frac, "samp_s": 0.0, "samp_m": 0.0, "samp_l": 0.0}

        # A5: top-K sampling (stratified default)
        sel_idx, samp_stats = sample_boxes_for_lgrsd(
            batch_idx=batch_idx,
            area=area0,
            batch_size=bs,
            topk=cfg.topk_per_image,
            strategy=cfg.sampling_strategy,
        )
        if sel_idx.numel() == 0:
            extra = {"skip_frac": skip_frac, **samp_stats}
            return img.new_zeros(()), extra

        batch_idx = batch_idx[sel_idx]
        xyxy_ctx = xyxy_ctx[sel_idx]

        rois = torch.cat([batch_idx.to(xyxy_ctx.dtype).unsqueeze(1), xyxy_ctx], dim=1)  # [N,5]

        # Student multi-level FPN features captured from the student forward pass.
        saved: dict[int, torch.Tensor] = getattr(self.model, "_lgrsd_saved_feats", {})
        detect = self.model.model[-1]
        f = getattr(detect, "f", None)
        if not isinstance(f, (list, tuple)) or len(f) != 3:
            extra = {"skip_frac": skip_frac, **samp_stats}
            return img.new_zeros(()), extra
        feat_indices = [int(x) for x in f]
        # During validation (or when preds are precomputed), the cache may not be populated.
        # LG-RSD is a train-time regularizer; safely skip if missing.
        if not saved or any(i not in saved for i in feat_indices):
            extra = {"skip_frac": skip_frac, **samp_stats}
            return img.new_zeros(()), extra
        feats = [saved[i] for i in feat_indices]
        strides = [float(s) for s in detect.stride]  # type: ignore[attr-defined]

        levels = assign_fpn_levels(xyxy_ctx, feature_level=cfg.feature_level, num_levels=len(feats))
        pooled = roi_align_by_levels(feats, rois, levels, strides, cfg=RoiAlignCfg(output_size=cfg.roi_size))

        head: LGRSDHead = getattr(self.model, "lgrsd_head")
        z_roi = img.new_zeros((rois.shape[0], head.embed_dim))
        for lvl, (roi_idx, feat_roi) in pooled.items():
            v = feat_roi.mean(dim=(2, 3))  # GAP -> [n, C]
            z_roi[roi_idx] = head.proj_roi(lvl, v)

        # Crop-teacher (stop-grad): crop patches from the image, then embed using a teacher layer.
        crop_patches = roi_align(
            img,
            rois,
            output_size=cfg.crop_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )

        teacher_layer = int(getattr(self.model, "lgrsd_teacher_layer", feat_indices[-1]))
        teacher: nn.Module | None = getattr(self.model, "lgrsd_teacher", None)
        if teacher is None:
            # Fallback (should not happen once ensure_lgrsd_modules ran)
            teacher = self.model
        else:
            # Optional EMA teacher to reduce teacher-student drift (esp. important when the student learns attention).
            _ema_update_teacher(teacher, self.model, cfg.teacher_momentum)
        with torch.no_grad():
            z_crop_raw = _run_embed(teacher, crop_patches, layer_idx=teacher_layer).to(self.device)

        # Stop-grad on teacher encoder output, but keep crop projection trainable.
        z_crop = head.proj_crop(z_crop_raw.detach())

        # Cosine distillation (both normalized)
        cos = (z_roi * z_crop).sum(dim=1).clamp(-1.0, 1.0)
        rsd = (1.0 - cos).mean()
        extra = {"skip_frac": skip_frac, **samp_stats}
        return rsd * float(cfg.lambda_rsd), extra


def compute_region_cosine_for_batch(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute cos(z_loc, z_glob) per GT region for a single batch (for export/plotting).

    Requires model to have been run with _lgrsd_cache=True so _lgrsd_saved_feats is populated.
    Returns 1D tensor of cosine values (one per sampled GT region); empty if no regions or cache missing.
    """
    if device is None:
        device = next(model.parameters()).device
    args = getattr(model, "args", None)
    cfg = _get_lgrsd_cfg(args)
    img = batch["img"].to(device)
    bs, _, img_h, img_w = img.shape

    batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
    bboxes = batch["bboxes"].view(-1, 4).to(device)

    if bboxes.numel() == 0:
        return img.new_zeros(0)

    xyxy = _clip_xyxy(_xywhn_to_xyxy_abs(bboxes, img_w=img_w, img_h=img_h), img_w=img_w, img_h=img_h)
    w0 = (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0.0)
    h0 = (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0.0)
    valid = (w0 > 1.0) & (h0 > 1.0)
    if valid.sum() == 0:
        return img.new_zeros(0)
    batch_idx = batch_idx[valid]
    xyxy = xyxy[valid]
    w0 = w0[valid]
    h0 = h0[valid]
    area0 = (w0 * h0).clamp(min=0.0)

    min_side0 = torch.minimum(w0, h0)
    keep_mask = min_side0 >= float(cfg.min_orig_side_px)
    xyxy = xyxy[keep_mask]
    batch_idx = batch_idx[keep_mask]
    area0 = area0[keep_mask]
    if xyxy.numel() == 0:
        return img.new_zeros(0)

    xyxy_ctx, area_ratio = _expand_xyxy_with_context(
        xyxy, img_w=img_w, img_h=img_h, context_ratio=cfg.context_ratio, min_side_px=cfg.min_side_px
    )
    keep_mask2 = area_ratio >= float(cfg.min_area_ratio)
    batch_idx = batch_idx[keep_mask2]
    xyxy_ctx = xyxy_ctx[keep_mask2]
    area0 = area0[keep_mask2]
    if xyxy_ctx.numel() == 0:
        return img.new_zeros(0)

    sel_idx, _ = sample_boxes_for_lgrsd(
        batch_idx=batch_idx,
        area=area0,
        batch_size=bs,
        topk=cfg.topk_per_image,
        strategy=cfg.sampling_strategy,
    )
    if sel_idx.numel() == 0:
        return img.new_zeros(0)

    batch_idx = batch_idx[sel_idx]
    xyxy_ctx = xyxy_ctx[sel_idx]
    rois = torch.cat([batch_idx.to(xyxy_ctx.dtype).unsqueeze(1), xyxy_ctx], dim=1)

    saved: dict[int, torch.Tensor] = getattr(model, "_lgrsd_saved_feats", {})
    detect = model.model[-1]
    f = getattr(detect, "f", None)
    if not isinstance(f, (list, tuple)) or len(f) != 3:
        return img.new_zeros(0)
    feat_indices = [int(x) for x in f]
    if not saved or any(i not in saved for i in feat_indices):
        return img.new_zeros(0)

    feats = [saved[i] for i in feat_indices]
    strides = [float(s) for s in detect.stride]
    levels = assign_fpn_levels(xyxy_ctx, feature_level=cfg.feature_level, num_levels=len(feats))
    pooled = roi_align_by_levels(feats, rois, levels, strides, cfg=RoiAlignCfg(output_size=cfg.roi_size))

    head: LGRSDHead = getattr(model, "lgrsd_head")
    z_roi = img.new_zeros((rois.shape[0], head.embed_dim))
    for lvl, (roi_idx, feat_roi) in pooled.items():
        v = feat_roi.mean(dim=(2, 3))
        z_roi[roi_idx] = head.proj_roi(lvl, v)

    crop_patches = roi_align(
        img, rois, output_size=cfg.crop_size, spatial_scale=1.0, sampling_ratio=2, aligned=True
    )
    teacher_layer = int(getattr(model, "lgrsd_teacher_layer", feat_indices[-1]))
    teacher: nn.Module | None = getattr(model, "lgrsd_teacher", None)
    if teacher is None:
        teacher = model
    with torch.no_grad():
        z_crop_raw = _run_embed(teacher, crop_patches, layer_idx=teacher_layer).to(device)
    z_crop = head.proj_crop(z_crop_raw.detach())
    cos = (z_roi * z_crop).sum(dim=1).clamp(-1.0, 1.0)
    return cos


