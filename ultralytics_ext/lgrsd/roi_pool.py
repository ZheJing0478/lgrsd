from __future__ import annotations

from dataclasses import dataclass

import torch
from torchvision.ops import roi_align


@dataclass(frozen=True)
class RoiAlignCfg:
    output_size: int = 7
    sampling_ratio: int = 2
    aligned: bool = True


def assign_fpn_levels(
    xyxy: torch.Tensor,
    *,
    feature_level: str,
    num_levels: int = 3,
) -> torch.Tensor:
    """Assign each box to an FPN level index (0..num_levels-1).

    Args:
      xyxy: Tensor [N, 4] in absolute pixel coords.
      feature_level: 'auto' or 'P3-only'.
      num_levels: 3 for P3/P4/P5.
    """
    if xyxy.numel() == 0:
        return torch.empty((0,), device=xyxy.device, dtype=torch.long)
    if feature_level.lower() in {"p3", "p3-only", "p3_only"}:
        return torch.zeros((xyxy.shape[0],), device=xyxy.device, dtype=torch.long)

    # FPN assignment (Detectron-style):
    # level = floor(k0 + log2(s / 224)), where s = sqrt(area), k0=4 for P4 at 224.
    x1, y1, x2, y2 = xyxy.unbind(dim=1)
    w = (x2 - x1).clamp(min=1.0)
    h = (y2 - y1).clamp(min=1.0)
    s = torch.sqrt(w * h)
    level = torch.floor(4.0 + torch.log2(s / 224.0 + 1e-9))
    level = level.clamp(min=3.0, max=3.0 + (num_levels - 1))  # -> {3,4,5}
    return (level.to(torch.long) - 3).clamp(min=0, max=num_levels - 1)


def roi_align_by_levels(
    feats: list[torch.Tensor],
    rois: torch.Tensor,
    levels: torch.Tensor,
    strides: list[float],
    cfg: RoiAlignCfg,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Multi-level ROIAlign.

    Args:
      feats: list of feature maps [B, C_l, H_l, W_l] for each level.
      rois: [N, 5] (batch_idx, x1, y1, x2, y2) in input-image pixel coords.
      levels: [N] in {0..L-1} level assignment for each ROI.
      strides: list length L, e.g. [8, 16, 32]
      cfg: roi_align hyperparams.

    Returns:
      dict[level] = (roi_indices, pooled_feats)
      - roi_indices: [n_l] indices into the original rois array
      - pooled_feats: [n_l, C_l, output_size, output_size]
    """
    if rois.numel() == 0:
        return {}
    out: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for lvl in range(len(feats)):
        idx = torch.nonzero(levels == lvl, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        pooled = roi_align(
            feats[lvl],
            rois[idx],
            output_size=cfg.output_size,
            spatial_scale=1.0 / float(strides[lvl]),
            sampling_ratio=cfg.sampling_ratio,
            aligned=cfg.aligned,
        )
        out[lvl] = (idx, pooled)
    return out


