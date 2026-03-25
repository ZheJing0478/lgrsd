from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import roi_align


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure local packages (ultralytics_ext/) are importable when running via `conda run`.
sys.path.insert(0, str(_repo_root()))


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """img_chw: float tensor [C,H,W] in 0..1 or 0..255."""
    x = img_chw.detach().float().cpu()
    if x.max() <= 1.5:
        x = (x * 255.0).clamp(0, 255)
    x = x.to(torch.uint8)
    if x.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(x.shape)}")
    c, h, w = x.shape
    if c == 1:
        x = x.repeat(3, 1, 1)
    x = x.permute(1, 2, 0).numpy()  # HWC
    return x


def _draw_boxes(img: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2) in xyxy.astype(np.int32):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return out


def _draw_one_box(img: np.ndarray, xyxy: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    x1, y1, x2, y2 = [int(v) for v in xyxy.astype(np.int32)]
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        out,
        text,
        (max(0, x1), max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _xywhn_to_xyxy_abs(xywhn: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
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


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    if x.numel() == 0:
        return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "nonzero_frac": float("nan")}
    xf = x.detach().float()
    nz = (xf.abs() > 1e-12).float().mean().item()
    return {"mean": xf.mean().item(), "std": xf.std(unbiased=False).item(), "max": xf.abs().max().item(), "nonzero_frac": nz}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Verify LG-RSD crop alignment (augmented image) and ROIAlign coordinate correctness."
    )
    p.add_argument("--data", required=True, help="Dataset yaml, e.g. datasets/HRSID/hrsid.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base model checkpoint (for feature extraction)")
    p.add_argument("--device", default="0", help="Device string, e.g. 0 or cpu")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_imgs", type=int, default=2)
    p.add_argument("--boxes_per_img", type=int, default=3)
    p.add_argument("--feature_level", default="auto", choices=["auto", "P3-only"])
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--roi_size", type=int, default=7)
    p.add_argument("--outdir", default="runs/debug", help="Output directory root, default runs/debug")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_all(args.seed)

    repo_root = _repo_root()
    os.chdir(repo_root)

    out_root = Path(args.outdir)
    align_dir = out_root / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # Import after sys.path injection
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics_ext.lgrsd.roi_pool import RoiAlignCfg, assign_fpn_levels, roi_align_by_levels

    overrides: dict[str, Any] = dict(
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        project=str(out_root),
        name="verify_lgrsd_alignment",
        exist_ok=True,
        # Enable LG-RSD knobs for using the same coordinate conventions (but we won't train).
        enable_lgrsd=True,
        lgrsd_feature_level=args.feature_level,
        lgrsd_crop_size=args.crop_size,
        lgrsd_roi_size=args.roi_size,
        lgrsd_topk_per_image=64,  # keep many; we will sub-sample for visualization
        lgrsd_lambda=0.5,
    )

    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    model = trainer.model
    model.eval()  # deterministic features

    # rank=-1 to avoid DistributedSampler (no init_process_group in this standalone script).
    train_loader = trainer.get_dataloader(trainer.data["train"], batch_size=args.batch, rank=-1, mode="train")
    batch = next(iter(train_loader))
    batch = trainer.preprocess_batch(batch)

    imgs = batch["img"]  # [B,C,H,W], float in 0..1 (preprocess_batch divides by 255)
    bs, _, img_h, img_w = imgs.shape

    # Trigger a student forward pass and capture P3/P4/P5 features (via our cache flag).
    setattr(model, "_lgrsd_cache", True)
    with torch.no_grad():
        _ = model.predict(imgs)
    setattr(model, "_lgrsd_cache", False)
    saved: dict[int, torch.Tensor] = getattr(model, "_lgrsd_saved_feats", {})

    detect = model.model[-1]
    feat_indices = list(getattr(detect, "f", []))
    strides = [float(s) for s in getattr(detect, "stride", [])]

    # Prepare GT boxes in pixel xyxy, and select a small subset for visualization.
    batch_idx = batch["batch_idx"].view(-1).to(dtype=torch.long)
    bboxes = batch["bboxes"].view(-1, 4)
    xyxy = _clip_xyxy(_xywhn_to_xyxy_abs(bboxes, img_w=img_w, img_h=img_h), img_w=img_w, img_h=img_h)
    valid = (xyxy[:, 2] - xyxy[:, 0] > 1.0) & (xyxy[:, 3] - xyxy[:, 1] > 1.0)
    batch_idx = batch_idx[valid]
    xyxy = xyxy[valid]

    # Choose up to num_imgs images from this batch
    img_ids = list(range(min(args.num_imgs, bs)))
    chosen: list[int] = []
    for b in img_ids:
        idx = torch.nonzero(batch_idx == b, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        idx_list = idx.tolist()
        random.shuffle(idx_list)
        chosen.extend(idx_list[: min(args.boxes_per_img, len(idx_list))])
    chosen_idx = torch.tensor(chosen, device=xyxy.device, dtype=torch.long) if chosen else xyxy.new_zeros((0,), dtype=torch.long)

    rois = torch.cat([batch_idx[chosen_idx].to(xyxy.dtype).unsqueeze(1), xyxy[chosen_idx]], dim=1) if chosen_idx.numel() else xyxy.new_zeros((0, 5))

    # A1 evidence: save augmented images + boxes and the corresponding crop patches from imgs tensor.
    roi_stats: dict[str, Any] = {
        "img_shape": [int(x) for x in imgs.shape],
        "xyxy_min": [float(x) for x in xyxy.min(dim=0).values.cpu()] if xyxy.numel() else None,
        "xyxy_max": [float(x) for x in xyxy.max(dim=0).values.cpu()] if xyxy.numel() else None,
        "feature_level": args.feature_level,
        "feat_indices": [int(i) for i in feat_indices],
        "strides": [float(s) for s in strides],
    }

    # Save img visualizations per selected image id
    for b in img_ids:
        idx = torch.nonzero(batch_idx == b, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        img_np = _to_uint8_hwc(imgs[b])
        xyxy_b = xyxy[idx].detach().cpu().numpy()
        vis = _draw_boxes(img_np, xyxy_b)
        cv2.imwrite(str(align_dir / f"img_b{b}.png"), vis)

    # Save crop patches for chosen rois (from augmented imgs tensor)
    if rois.numel():
        # Store mapping crop_k -> (image_id, xyxy_px)
        roi_list: list[dict[str, Any]] = []
        for k in range(int(rois.shape[0])):
            roi_list.append(
                {
                    "roi_id": k,
                    "image_id": int(rois[k, 0].item()),
                    "xyxy_px": [float(v) for v in rois[k, 1:].detach().cpu().tolist()],
                    "crop_file": f"crop_{k}.png",
                }
            )
        roi_stats["chosen_rois"] = roi_list

        crops = roi_align(
            imgs,
            rois,
            output_size=args.crop_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        roi_stats["crop_patches"] = {
            "shape": [int(x) for x in crops.shape],
            "stats": _tensor_stats(crops),
        }
        for k in range(min(crops.shape[0], args.num_imgs * args.boxes_per_img)):
            crop_np = _to_uint8_hwc(crops[k])
            cv2.imwrite(str(align_dir / f"crop_{k}.png"), crop_np)

            # Also save a per-ROI annotated full image to make alignment unambiguous.
            b = int(rois[k, 0].item())
            img_np = _to_uint8_hwc(imgs[b])
            xyxy_k = rois[k, 1:].detach().cpu().numpy()
            vis_one = _draw_one_box(img_np, xyxy_k, text=f"roi{k}")
            cv2.imwrite(str(align_dir / f"img_roi{k}.png"), vis_one)
    else:
        roi_stats["crop_patches"] = {"shape": None, "stats": None}
        roi_stats["chosen_rois"] = []

    # A2 evidence: ROIAlign over P3/P4/P5 with correct spatial_scale, plus stats per level.
    roi_debug_lines: list[str] = []
    roi_debug_lines.append(f"imgs: {tuple(imgs.shape)} (H={img_h}, W={img_w})")
    roi_debug_lines.append(f"bboxes(norm xywh) range: min={bboxes.min().item():.6f} max={bboxes.max().item():.6f}")
    roi_debug_lines.append(f"xyxy(pixel) range: min={xyxy.min(dim=0).values.tolist()} max={xyxy.max(dim=0).values.tolist()}")
    roi_debug_lines.append(f"detect.f (feat_indices): {feat_indices}")
    roi_debug_lines.append(f"detect.stride: {strides}")
    roi_debug_lines.append(f"chosen_rois: {int(rois.shape[0])}")

    if feat_indices and saved and all(int(i) in saved for i in feat_indices):
        feats = [saved[int(i)] for i in feat_indices]
        for li, f in enumerate(feats):
            roi_debug_lines.append(f"feat[{li}] idx={feat_indices[li]} shape={tuple(f.shape)}")

        if rois.numel():
            levels = assign_fpn_levels(xyxy[chosen_idx], feature_level=args.feature_level, num_levels=len(feats))
            per_lvl_counts = {int(l): int((levels == l).sum().item()) for l in range(len(feats))}
            roi_stats["assigned_counts"] = per_lvl_counts
            roi_debug_lines.append(f"assigned_counts: {per_lvl_counts}")
            # Attach per-ROI assigned level in the mapping (if present)
            if "chosen_rois" in roi_stats:
                for k, lvl in enumerate(levels.detach().cpu().tolist()):
                    if k < len(roi_stats["chosen_rois"]):
                        roi_stats["chosen_rois"][k]["assigned_level"] = int(lvl)

            pooled = roi_align_by_levels(
                feats,
                rois,
                levels,
                strides,
                cfg=RoiAlignCfg(output_size=args.roi_size, sampling_ratio=2, aligned=True),
            )
            pooled_stats: dict[str, Any] = {}
            for lvl, (roi_i, p) in pooled.items():
                pooled_stats[str(lvl)] = {
                    "roi_indices": roi_i.detach().cpu().tolist(),
                    "pooled_shape": [int(x) for x in p.shape],
                    "stats": _tensor_stats(p),
                }
                roi_debug_lines.append(f"pooled[lvl={lvl}] shape={tuple(p.shape)} stats={pooled_stats[str(lvl)]['stats']}")
            roi_stats["pooled_by_level"] = pooled_stats
        else:
            roi_stats["assigned_counts"] = None
            roi_stats["pooled_by_level"] = None
    else:
        roi_debug_lines.append("WARNING: missing saved feats for detect.f indices; cannot compute ROIAlign stats.")
        roi_stats["assigned_counts"] = None
        roi_stats["pooled_by_level"] = None

    (align_dir / "roi_stats.json").write_text(json.dumps(roi_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_root / "roi_align_debug.txt").write_text("\n".join(roi_debug_lines) + "\n", encoding="utf-8")

    print(f"[OK] alignment images -> {align_dir}")
    print(f"[OK] roi_stats.json -> {align_dir/'roi_stats.json'}")
    print(f"[OK] roi_align_debug.txt -> {out_root/'roi_align_debug.txt'}")


if __name__ == "__main__":
    main()


