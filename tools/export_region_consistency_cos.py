#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export cos(z_loc, z_glob) per GT region for baseline and LG-RSD/Ours, then save as .npy for plotting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_ULTRALYTICS_ROOT = _REPO_ROOT / "method" / "yolov8"
if _ULTRALYTICS_ROOT.exists() and str(_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ULTRALYTICS_ROOT))

from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from ultralytics_ext.losses import ensure_lgrsd_modules
from ultralytics_ext.losses.yolo_loss_lgrsd import compute_region_cosine_for_batch


def _default_lgrsd_args() -> SimpleNamespace:
    ns = SimpleNamespace()
    ns.enable_lgrsd = True
    ns.lgrsd_lambda = 0.5
    ns.lgrsd_topk_per_image = 16
    ns.lgrsd_crop_size = 224
    ns.lgrsd_roi_size = 7
    ns.lgrsd_feature_level = "auto"
    ns.lgrsd_embed_dim = 256
    ns.lgrsd_min_side_px = 16
    ns.lgrsd_context_ratio = 1.3
    ns.lgrsd_min_orig_side_px = 4
    ns.lgrsd_min_area_ratio = 0.3
    ns.lgrsd_sampling_strategy = "stratified_default"
    ns.lgrsd_teacher_momentum = 1.0
    return ns


def parse_args():
    p = argparse.ArgumentParser(description="Export cos(z_loc, z_glob) per GT region to .npy for histogram plot.")
    p.add_argument("--model", required=True, help="Path to model weights (e.g. runs/hrsid/baseline_yolov8n/weights/best.pt).")
    p.add_argument("--data", required=True, help="Path to data yaml (e.g. datasets/HRSID/hrsid.yaml).")
    p.add_argument("--out", required=True, help="Output .npy path (e.g. runs/hrsid/baseline_cos.npy).")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="0")
    p.add_argument("--max_batches", type=int, default=None, help="Cap number of batches (default: all).")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Load model (inner DetectionModel)
    y = YOLO(args.model)
    model = y.model
    model.to(device)
    model.eval()

    # Ensure LG-RSD head exists (for baseline we add it with random init)
    defs = _default_lgrsd_args()
    defs_dict = vars(defs)
    raw_args = getattr(model, "args", None)
    if isinstance(raw_args, dict):
        train_args = SimpleNamespace(**{**raw_args, **defs_dict})
    else:
        train_args = raw_args or SimpleNamespace()
        for k, v in defs_dict.items():
            if not hasattr(train_args, k):
                setattr(train_args, k, v)
    train_args.enable_lgrsd = True
    model.args = train_args
    ensure_lgrsd_modules(model)
    # Teacher is not a submodule; move to device for export
    teacher = getattr(model, "lgrsd_teacher", None)
    if teacher is not None:
        teacher.to(device)

    # Data: use train (or val) so we have GT labels; check_det_dataset resolves paths
    data = check_det_dataset(args.data)
    val_path = data.get("train") or data.get("val") or data.get("test")
    if not val_path:
        raise FileNotFoundError(f"No train/val/test in data: {args.data}")
    val_path = val_path[0] if isinstance(val_path, (list, tuple)) else val_path
    val_path = str(Path(val_path).resolve())

    stride = 32
    if hasattr(model, "stride") and model.stride is not None:
        s = model.stride
        stride = int(s.max().item()) if hasattr(s, "max") else int(s) if isinstance(s, (int, float)) else 32
    # classes=None avoids filtering by include_class in update_labels (which can clear bboxes when dict is passed)
    cfg = SimpleNamespace(
        imgsz=args.imgsz,
        rect=True,
        cache=False,
        single_cls=data.get("single_cls", False) or (data.get("nc") == 1),
        task="detect",
        classes=None,
        bgr=0.0,
        mask_ratio=4,
        overlap_mask=True,
        fraction=1.0,
    )
    dataset = build_yolo_dataset(cfg, val_path, args.batch, data, mode="val", rect=False, stride=stride)
    loader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1, drop_last=False)

    cos_list = []
    n_batches = 0
    for batch in loader:
        if args.max_batches is not None and n_batches >= args.max_batches:
            break
        # Move to device and normalize img as in training
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float() / 255.0

        # Populate _lgrsd_saved_feats (model.predict -> _predict_once saves when _lgrsd_cache and m.i in self.save)
        setattr(model, "_lgrsd_cache", True)
        with torch.no_grad():
            _ = model(batch["img"])
        setattr(model, "_lgrsd_cache", False)
        cos = compute_region_cosine_for_batch(model, batch, device=device)
        if cos.numel() > 0:
            cos_list.append(cos.detach().cpu().numpy())
        n_batches += 1

    arr = np.concatenate(cos_list, axis=0).astype(np.float32) if cos_list else np.array([], dtype=np.float32)
    np.save(out_path, arr)
    print(f"[OK] Saved {arr.size} cosine values to {out_path}")


if __name__ == "__main__":
    main()
