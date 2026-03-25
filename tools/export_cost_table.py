from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import sys
import torch
import yaml

# Ensure project-local packages (e.g., ultralytics_ext/) are importable when running via `conda run`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops, get_num_params  # noqa: E402


METHODS = ("baseline", "attn", "lgrsd", "final")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export cost table (params/GFLOPs/train time/infer time) for Phase 2.")
    p.add_argument("--summary", default="runs/summary.json", help="Path to runs/summary.json")
    p.add_argument("--outdir", default="runs", help="Runs directory root.")
    p.add_argument("--model_scale", default="n", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--datasets", default="hrsid,ssdd", help="Comma-separated datasets to average timings over.")
    p.add_argument("--sar_input", default="rgb3", choices=["rgb3", "gray1"])
    p.add_argument(
        "--filter_tag",
        default="",
        help="Only use runs whose exp_key contains this substring (prevents mixing protocols). "
        "Example: --filter_tag p2final_1024",
    )
    p.add_argument("--imgsz", type=int, default=1024, help="Image size used for GFLOPs/inference timing.")
    p.add_argument("--device", default="0", help="Device for profiling (e.g., 0 or cpu).")
    p.add_argument("--half", action="store_true", help="Use FP16 during inference profiling (CUDA only).")
    p.add_argument("--max_images", type=int, default=200, help="Max images per dataset for inference profiling.")
    p.add_argument("--warmup", type=int, default=10, help="Warmup images before timing.")
    p.add_argument("--no_profile", action="store_true", help="Skip inference-time profiling (fills with NaN).")
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict json: {path}")
    return data


def _load_data_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {path}")
    return data


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def _pick_key(summary: dict[str, Any], *, ds: str, method: str, scale: str, sar: str, filter_tag: str) -> str | None:
    # Primary: exp_key encodes sar (A7): "<ds>/<method>_yolov8{scale}_{sar}[_{tag}]"
    prefix = f"{ds}/{method}_yolov8{scale}_{sar}"
    candidates = [k for k in summary.keys() if isinstance(k, str) and k.startswith(prefix)]
    # Legacy fallback: "<ds>/<method>_yolov8{scale}..."
    if not candidates and sar == "rgb3":
        legacy_prefix = f"{ds}/{method}_yolov8{scale}"
        candidates = [k for k in summary.keys() if isinstance(k, str) and k.startswith(legacy_prefix)]
    if not candidates:
        return None

    # Filter by sar_input field when present (legacy missing => assume rgb3).
    filtered: list[str] = []
    for k in candidates:
        m = summary.get(k, {})
        sar_k = m.get("sar_input") if isinstance(m, dict) else None
        if isinstance(sar_k, str):
            if sar_k == sar:
                filtered.append(k)
        else:
            if sar == "rgb3":
                filtered.append(k)
    candidates = filtered
    if not candidates:
        return None

    if filter_tag:
        candidates = [k for k in candidates if filter_tag in k]
        if not candidates:
            return None

    # Prefer highest AP (sanity); for cost table any key is fine as long as protocol matches.
    def _ap(k: str) -> float:
        try:
            return float(summary[k].get("AP", float("-inf")))
        except Exception:
            return float("-inf")

    return max(candidates, key=_ap)


def _run_dir(outdir: Path, exp_key: str) -> Path:
    ds, run_name = exp_key.split("/", 1)
    return outdir / ds / run_name


def _train_time_sec_per_epoch(results_csv: Path) -> float:
    if not results_csv.exists():
        return float("nan")
    lines = results_csv.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 2:
        return float("nan")
    last = lines[-1].split(",")
    if len(last) < 2:
        return float("nan")
    try:
        epoch = float(last[0])
        t = float(last[1])
        if epoch <= 0:
            return float("nan")
        return t / epoch
    except Exception:
        return float("nan")


def _profile_inference_ms_per_image(
    yolo: YOLO,
    images_dir: Path,
    *,
    imgsz: int,
    device: str,
    half: bool,
    max_images: int,
    warmup: int,
) -> float:
    files = _list_images(images_dir)
    if not files:
        return float("nan")
    files = files[: max_images + warmup]

    # Warmup
    n_seen = 0
    for _ in yolo.predict(
        source=[str(p) for p in files[:warmup]],
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        max_det=500,
        batch=1,
        half=bool(half),
        device=device,
        stream=True,
        save=False,
        verbose=False,
    ):
        pass
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    n = 0
    for _ in yolo.predict(
        source=[str(p) for p in files[warmup:]],
        imgsz=imgsz,
        conf=0.001,
        iou=0.7,
        max_det=500,
        batch=1,
        half=bool(half),
        device=device,
        stream=True,
        save=False,
        verbose=False,
    ):
        n_seen += 1
        n += 1
        if n >= max_images:
            break
    if torch.cuda.is_available() and str(device) != "cpu":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    if n <= 0:
        return float("nan")
    return (t1 - t0) * 1000.0 / float(n)


def _fmt_float(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    return f"{v:.{digits}f}"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    summary = _load_json(Path(args.summary).expanduser().resolve())

    scale = str(args.model_scale)
    sar = str(args.sar_input)
    filter_tag = str(args.filter_tag).strip()
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]

    # Locate one representative model path per method for params/FLOPs (architecture).
    # We'll use the first dataset where the method exists under the chosen protocol.
    method_model_path: dict[str, str] = {}
    method_runs: dict[str, dict[str, Any]] = {}

    for method in METHODS:
        picked_key = None
        for ds in datasets:
            k = _pick_key(summary, ds=ds, method=method, scale=scale, sar=sar, filter_tag=filter_tag)
            if k:
                picked_key = k
                break
        if not picked_key:
            continue
        run_dir = _run_dir(outdir, picked_key)
        strict_dir = run_dir.parent / f"{run_dir.name}_strict"
        eval_args_path = strict_dir / "eval_args.json"
        if not eval_args_path.exists():
            continue
        eval_args = _load_json(eval_args_path)
        model_path = str(eval_args.get("model", ""))
        if model_path:
            method_model_path[method] = model_path
            method_runs[method] = {"exp_key": picked_key, "run_dir": str(run_dir), "strict_dir": str(strict_dir)}

    # Compute params/FLOPs
    arch_info: dict[str, dict[str, Any]] = {}
    for method, model_path in method_model_path.items():
        y = YOLO(model_path)
        # Move to device to compute FLOPs consistently (thop requires a device)
        dev = torch.device("cuda:0") if (str(args.device) != "cpu" and torch.cuda.is_available()) else torch.device("cpu")
        y.model.to(dev)
        params_total = int(get_num_params(y.model))
        # LG-RSD adds a training-only projection head that is not used in inference forward.
        # We count it separately so the cost table can reflect "inference-effective params".
        params_train_only = int(sum(p.numel() for n, p in y.model.named_parameters() if n.startswith("lgrsd_head.")))
        params_infer = int(params_total - params_train_only)
        flops = float(get_flops(y.model, args.imgsz))
        arch_info[method] = {
            "params_total": params_total,
            "params_train_only": params_train_only,
            "params_infer": params_infer,
            "gflops": flops,
            "model_path": model_path,
        }

    # Compute timings per dataset (train sec/epoch from results.csv, infer ms/img by profiling)
    timings: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        ds_tim: dict[str, Any] = {}
        for method in METHODS:
            k = _pick_key(summary, ds=ds, method=method, scale=scale, sar=sar, filter_tag=filter_tag)
            if not k:
                continue
            run_dir = _run_dir(outdir, k)
            results_csv = run_dir / "results.csv"
            train_sec_ep = _train_time_sec_per_epoch(results_csv)

            infer_ms = float("nan")
            if not args.no_profile:
                # Use dataset yaml from strict eval args if available, else skip.
                strict_dir = run_dir.parent / f"{run_dir.name}_strict"
                eval_args_path = strict_dir / "eval_args.json"
                if eval_args_path.exists():
                    eval_args = _load_json(eval_args_path)
                    data_yaml = Path(str(eval_args.get("data", ""))).expanduser().resolve()
                    if data_yaml.exists() and method in method_model_path:
                        data = _load_data_yaml(data_yaml)
                        ds_root = Path(str(data.get("path", ""))).expanduser().resolve()
                        val_rel = data.get("val") or data.get("test") or data.get("train")
                        if val_rel:
                            images_dir = (ds_root / str(val_rel)).resolve()
                            y = YOLO(method_model_path[method])
                            infer_ms = _profile_inference_ms_per_image(
                                y,
                                images_dir,
                                imgsz=args.imgsz,
                                device=str(args.device),
                                half=bool(args.half),
                                max_images=int(args.max_images),
                                warmup=int(args.warmup),
                            )

            ds_tim[method] = {"exp_key": k, "train_sec_per_epoch": train_sec_ep, "infer_ms_per_image": infer_ms}
        timings[ds] = ds_tim

    # Aggregate (mean over datasets that have numbers)
    def _mean(vals: list[float]) -> float:
        vals = [v for v in vals if isinstance(v, float) and not math.isnan(v) and not math.isinf(v)]
        return sum(vals) / len(vals) if vals else float("nan")

    rows: list[dict[str, Any]] = []
    for method in METHODS:
        ai = arch_info.get(method, {})
        train_vals = [timings.get(ds, {}).get(method, {}).get("train_sec_per_epoch", float("nan")) for ds in datasets]
        infer_vals = [timings.get(ds, {}).get(method, {}).get("infer_ms_per_image", float("nan")) for ds in datasets]
        params_infer = float(ai.get("params_infer", float("nan")))
        params_train_only = float(ai.get("params_train_only", float("nan")))
        params_total = float(ai.get("params_total", float("nan")))
        rows.append(
            {
                "method": method,
                # Report inference-effective params for fair comparison (LG-RSD is training-only).
                "params_M": (params_infer / 1e6) if not math.isnan(params_infer) else float("nan"),
                "params_total_M": (params_total / 1e6) if not math.isnan(params_total) else float("nan"),
                "params_train_only_M": (params_train_only / 1e6) if not math.isnan(params_train_only) else float("nan"),
                "gflops_1024": float(ai.get("gflops", float("nan"))),
                "train_sec_per_epoch_avg": _mean([float(v) for v in train_vals]),
                "infer_ms_per_image_avg": _mean([float(v) for v in infer_vals]),
                "note": (
                    "LG-RSD has 0 inference compute; params_M excludes train-only lgrsd_head."
                    if method in ("lgrsd", "final")
                    else ""
                ),
            }
        )

    outdir.mkdir(parents=True, exist_ok=True)
    cost_json = {
        "meta": {
            "imgsz": args.imgsz,
            "device": args.device,
            "half": bool(args.half),
            "sar_input": sar,
            "model_scale": scale,
            "filter_tag": filter_tag,
            "datasets": datasets,
        },
        "arch": arch_info,
        "timings": timings,
        "rows": rows,
        "method_runs": method_runs,
    }
    (outdir / "cost.json").write_text(json.dumps(cost_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # LaTeX table
    header = [
        r"\begin{tabular}{l r r r r}",
        r"\toprule",
        r"Method & Params (M) & GFLOPs@1024 & Train s/epoch & Infer ms/img \\",
        r"\midrule",
    ]
    body: list[str] = []
    for r in rows:
        body.append(
            " & ".join(
                [
                    r["method"],
                    _fmt_float(float(r["params_M"]), 2),
                    _fmt_float(float(r["gflops_1024"]), 2),
                    _fmt_float(float(r["train_sec_per_epoch_avg"]), 2),
                    _fmt_float(float(r["infer_ms_per_image_avg"]), 2),
                ]
            )
            + r" \\"
        )
    footer = [r"\bottomrule", r"\end{tabular}", ""]
    (outdir / "table_cost.tex").write_text("\n".join(header + body + footer), encoding="utf-8")

    print(f"[OK] {outdir/'cost.json'}")
    print(f"[OK] {outdir/'table_cost.tex'}")


if __name__ == "__main__":
    main()


