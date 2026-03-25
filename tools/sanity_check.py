from __future__ import annotations

import argparse
import sys
from typing import Any

import torch
from torchvision.ops import roi_align


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LG-RSD sanity check: forward + loss + backward on synthetic data.")
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--boxes_per_image", type=int, default=3)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Ensure repo-root packages are importable when running as `python tools/...`.
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ultralytics_ext.losses import ensure_lgrsd_modules
    from ultralytics_ext.lgrsd.roi_pool import RoiAlignCfg, assign_fpn_levels, roi_align_by_levels

    # Build a baseline YOLOv8n detect model from YAML for a lightweight unit-style test.
    model = DetectionModel("method/yolov8/ultralytics/cfg/models/v8/yolov8.yaml", ch=3, nc=1, verbose=False).to(device)
    model.train(True)

    # Attach args (Ultralytics expects attribute-style access).
    cfg: dict[str, Any] = dict(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "task": "detect",
            "imgsz": args.imgsz,
            "enable_lgrsd": True,
            "lgrsd_lambda": 0.5,
            "lgrsd_topk_per_image": max(1, int(args.boxes_per_image)),
            "lgrsd_crop_size": min(224, int(args.imgsz)),
            "lgrsd_roi_size": 7,
            "lgrsd_feature_level": "auto",
            "lgrsd_embed_dim": 256,
            "enable_region_contrastive": False,
        }
    )
    model.args = IterableSimpleNamespace(**cfg)
    ensure_lgrsd_modules(model)
    model.criterion = model.init_criterion()

    # Synthetic batch: normalized xywh boxes.
    B = int(args.batch)
    H = W = int(args.imgsz)
    img = torch.rand(B, 3, H, W, device=device)

    all_boxes = []
    all_cls = []
    all_batch_idx = []
    for b in range(B):
        for _ in range(int(args.boxes_per_image)):
            xc = torch.empty(1).uniform_(0.2, 0.8)
            yc = torch.empty(1).uniform_(0.2, 0.8)
            bw = torch.empty(1).uniform_(0.05, 0.3)
            bh = torch.empty(1).uniform_(0.05, 0.3)
            all_boxes.append(torch.cat([xc, yc, bw, bh], dim=0))
            all_cls.append(torch.tensor(0.0))
            all_batch_idx.append(torch.tensor(float(b)))

    bboxes = torch.stack(all_boxes, dim=0).to(device)
    cls = torch.stack(all_cls, dim=0).to(device)
    batch_idx = torch.stack(all_batch_idx, dim=0).to(device)

    batch = {"img": img, "bboxes": bboxes, "cls": cls, "batch_idx": batch_idx}

    # ---------- ROIAlign coordinate sanity: roi_align vs naive pooling (same order-of-magnitude) ----------
    # Capture FPN feats once from a forward pass
    setattr(model, "_lgrsd_cache", True)
    with torch.no_grad():
        _ = model.predict(img)
    setattr(model, "_lgrsd_cache", False)
    saved: dict[int, torch.Tensor] = getattr(model, "_lgrsd_saved_feats", {})
    detect = model.model[-1]
    feat_indices = list(getattr(detect, "f", []))
    strides = [float(s) for s in getattr(detect, "stride", [])]
    if feat_indices and saved and all(int(i) in saved for i in feat_indices):
        feats = [saved[int(i)] for i in feat_indices]

        # Use the first GT box of the first image
        bs, _, H, W = img.shape
        xywhn0 = bboxes[0:1]
        # normalized xywh -> absolute xyxy
        xywh = xywhn0.clone()
        xywh[:, 0] *= float(W)
        xywh[:, 1] *= float(H)
        xywh[:, 2] *= float(W)
        xywh[:, 3] *= float(H)
        xc, yc, bw, bh = xywh.unbind(dim=1)
        xyxy0 = torch.stack([xc - bw / 2.0, yc - bh / 2.0, xc + bw / 2.0, yc + bh / 2.0], dim=1)
        xyxy0[:, [0, 2]] = xyxy0[:, [0, 2]].clamp(0.0, float(W))
        xyxy0[:, [1, 3]] = xyxy0[:, [1, 3]].clamp(0.0, float(H))
        rois0 = torch.cat([torch.zeros((1, 1), device=device), xyxy0], dim=1)  # [1,5], batch_idx=0

        levels = assign_fpn_levels(xyxy0, feature_level="auto", num_levels=len(feats))
        pooled = roi_align_by_levels(feats, rois0, levels, strides, cfg=RoiAlignCfg(output_size=7, sampling_ratio=2, aligned=True))
        lvl = int(levels.item())
        _, feat_roi = pooled[lvl]
        roi_vec = feat_roi.mean(dim=(2, 3)).squeeze(0)  # [C]

        # Naive pooling: mean over the corresponding region on the same feature level
        stride = float(strides[lvl])
        f = feats[lvl][0]  # [C, h, w]
        x1, y1, x2, y2 = xyxy0[0]
        fx1 = int(torch.floor(x1 / stride).item())
        fy1 = int(torch.floor(y1 / stride).item())
        fx2 = int(torch.ceil(x2 / stride).item())
        fy2 = int(torch.ceil(y2 / stride).item())
        fx1 = max(0, min(fx1, f.shape[-1] - 1))
        fy1 = max(0, min(fy1, f.shape[-2] - 1))
        fx2 = max(fx1 + 1, min(fx2, f.shape[-1]))
        fy2 = max(fy1 + 1, min(fy2, f.shape[-2]))
        naive_vec = f[:, fy1:fy2, fx1:fx2].mean(dim=(1, 2))

        roi_mag = float(roi_vec.abs().mean().detach().cpu())
        naive_mag = float(naive_vec.abs().mean().detach().cpu())
        ratio = roi_mag / (naive_mag + 1e-12)
        print(f"roi_align_vs_naive: lvl={lvl} stride={stride} roi_mag={roi_mag:.6f} naive_mag={naive_mag:.6f} ratio={ratio:.3f}")
        if not (0.1 <= ratio <= 10.0):
            raise RuntimeError(f"ROIAlign magnitude mismatch (ratio={ratio:.3f}) suggests coordinate/scale bug.")
    else:
        print("[WARN] Could not capture FPN features for ROIAlign sanity check; skipping.")

    # Forward + loss
    loss_vec, loss_items = model(batch)
    loss = loss_vec.sum()
    print("loss_vec:", loss_vec.detach().cpu().tolist())
    print("loss_items:", loss_items.detach().cpu().tolist())

    # Backward
    loss.backward()

    # Basic gradient checks (student path should have gradients; teacher path is no_grad).
    head = getattr(model, "lgrsd_head", None)
    if head is None:
        raise RuntimeError("lgrsd_head missing")
    g = head.crop_proj.weight.grad
    print("grad(lgrsd_head.crop_proj.weight):", None if g is None else float(g.abs().mean().detach().cpu()))

    teacher = getattr(model, "lgrsd_teacher", None)
    if teacher is None:
        raise RuntimeError("lgrsd_teacher missing (A3 requires frozen teacher for grad checks)")
    any_teacher_grad = False
    for p in teacher.parameters():
        if p.grad is not None and float(p.grad.detach().abs().sum().cpu()) > 0:
            any_teacher_grad = True
            break
    print("teacher_grad_all_none_or_zero:", (not any_teacher_grad))
    if any_teacher_grad:
        raise RuntimeError("Teacher received gradients (stop-grad broken).")
    print("[OK] sanity_check passed")


if __name__ == "__main__":
    main()


