from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A3: verify LG-RSD teacher receives no gradients (1 iter, real dataloader).")
    p.add_argument("--data", required=True, help="Dataset yaml, e.g. datasets/HRSID/hrsid.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/debug/teacher_grad_check.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _seed_all(args.seed)

    repo_root = _repo_root()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from ultralytics.models.yolo.detect.train import DetectionTrainer

    overrides: dict[str, Any] = dict(
        model=args.model,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        seed=args.seed,
        deterministic=True,
        project="runs/debug",
        name="teacher_grad_check",
        exist_ok=True,
        enable_lgrsd=True,
        lgrsd_lambda=0.5,
        lgrsd_topk_per_image=16,
        lgrsd_crop_size=224,
        lgrsd_roi_size=7,
        lgrsd_feature_level="auto",
        lgrsd_embed_dim=256,
        enable_region_contrastive=False,
    )

    trainer = DetectionTrainer(overrides=overrides)
    trainer.setup_model()
    trainer.model = trainer.model.to(trainer.device)
    trainer.set_model_attributes()
    model = trainer.model
    model.train(True)

    loader = trainer.get_dataloader(trainer.data["train"], batch_size=args.batch, rank=-1, mode="train")
    batch = next(iter(loader))
    batch = trainer.preprocess_batch(batch)

    # One forward + backward
    loss_vec, loss_items = model(batch)
    loss = loss_vec.sum()
    loss.backward()

    teacher = getattr(model, "lgrsd_teacher", None)
    if teacher is None:
        raise RuntimeError("lgrsd_teacher missing")

    total_params = 0
    params_with_grad = 0
    max_abs_grad = 0.0
    for p in teacher.parameters():
        total_params += 1
        if p.grad is not None:
            g = float(p.grad.detach().abs().max().cpu())
            if g > 0:
                params_with_grad += 1
                max_abs_grad = max(max_abs_grad, g)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"data: {args.data}",
        f"model: {args.model}",
        f"imgsz: {args.imgsz}",
        f"batch: {args.batch}",
        f"seed: {args.seed}",
        f"loss_vec: {loss_vec.detach().cpu().tolist()}",
        f"loss_items: {loss_items.detach().cpu().tolist()}",
        f"teacher_total_params: {total_params}",
        f"teacher_params_with_nonzero_grad: {params_with_grad}",
        f"teacher_max_abs_grad: {max_abs_grad}",
        "PASS" if params_with_grad == 0 else "FAIL",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {out_path} (PASS={params_with_grad == 0})")


if __name__ == "__main__":
    main()


