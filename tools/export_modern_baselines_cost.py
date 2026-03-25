"""Export computational cost (Params, GFLOPs @ 800×800) for modern baselines (e.g. RT-DETR-R18)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sys
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Use local ultralytics (method/yolov8) for get_flops/get_num_params
_ULTRALYTICS_ROOT = _REPO_ROOT / "method" / "yolov8"
if _ULTRALYTICS_ROOT.exists() and str(_ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ULTRALYTICS_ROOT))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

IMGSZ = 800


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Params and GFLOPs for modern baseline models.")
    p.add_argument("--imgsz", type=int, default=IMGSZ, help="Input size for GFLOPs (square).")
    p.add_argument("--outdir", default="runs", help="Output directory for cost JSON.")
    p.add_argument("--device", default="0")
    return p.parse_args()


def _fmt_float(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{digits}f}"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0") if (str(args.device) != "cpu" and torch.cuda.is_available()) else torch.device("cpu")

    # Model display name -> path to best.pt (use HRSID runs; architecture is same across datasets)
    r = _REPO_ROOT / "runs/hrsid"
    models: list[tuple[str, str]] = [
        ("Baseline (YOLOv8n)", str(r / "baseline_yolov8n/weights/best.pt")),
        ("AttnFPN (YOLOv8n+Attn)", str(r / "attn_yolov8n/weights/best.pt")),
        ("LG-RSD (YOLOv8n+LG-RSD)", str(r / "lgrsd_yolov8n_lam0p5/weights/best.pt")),
        ("Ours (LG-RSD-AttnFPN)", str(r / "final_yolov8n_rgb3_b5_1024_teachermom0p999_lam0p8_ctx1p0_crop160/weights/best.pt")),
        ("YOLOv8n (modern)", str(r / "modern_yolov8n_gray1/weights/best.pt")),
        ("RT-DETR-R18", str(r / "modern_rtdetr-r18_gray1/weights/best.pt")),
    ]

    rows: list[dict[str, Any]] = []
    for name, path in models:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            rows.append({"model": name, "params_M": None, "gflops": None, "model_path": str(path), "error": "not_found"})
            continue
        try:
            y = YOLO(str(path))
            y.model.to(dev)
            params = int(get_num_params(y.model))
            params_train_only = int(sum(p.numel() for n, p in y.model.named_parameters() if n.startswith("lgrsd_head.")))
            params_infer = params - params_train_only
            gflops = float(get_flops(y.model, args.imgsz))
            rows.append({
                "model": name,
                "params_M": round(params_infer / 1e6, 3),
                "params_total_M": round(params / 1e6, 3),
                "params_train_only_M": round(params_train_only / 1e6, 3) if params_train_only else 0,
                "gflops": round(gflops, 3),
                "imgsz": args.imgsz,
                "model_path": str(path),
            })
        except Exception as e:
            rows.append({"model": name, "params_M": None, "gflops": None, "model_path": str(path), "error": str(e)})

    cost = {"imgsz": args.imgsz, "device": str(args.device), "rows": rows}
    out_json = outdir / "cost_modern_baselines.json"
    out_json.write_text(json.dumps(cost, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # LaTeX snippet
    out_tex = outdir / "table_cost_modern_baselines.tex"
    header = [
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r"Model & Params (M) & GFLOPs@800 \\",
        r"\midrule",
    ]
    body = []
    for r in rows:
        pm = _fmt_float(r.get("params_M")) if r.get("params_M") is not None else "-"
        gf = _fmt_float(r.get("gflops")) if r.get("gflops") is not None else "-"
        body.append(f"{r['model']} & {pm} & {gf} \\\\")
    footer = [r"\bottomrule", r"\end{tabular}", ""]
    out_tex.write_text("\n".join(header + body + footer), encoding="utf-8")

    print(f"[OK] {out_json}")
    print(f"[OK] {out_tex}")
    for r in rows:
        print(f"  {r['model']}: Params={r.get('params_M')} M, GFLOPs@800={r.get('gflops')}")


if __name__ == "__main__":
    main()
