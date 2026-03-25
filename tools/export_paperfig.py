from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure local packages (ultralytics_ext/) are importable when running via `conda run`.
_REPO_ROOT = _repo_root()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export paper-ready figures into paperfig/ (curves/tables/debug/qual).")
    p.add_argument("--outdir", default="paperfig", help="Output directory (relative to repo root).")
    p.add_argument("--sar_input", default="rgb3", choices=["rgb3", "gray1"])
    p.add_argument("--datasets", default="hrsid,ssdd", help="Comma-separated: hrsid,ssdd")
    p.add_argument("--tag", default="p2final_1024_lam0p8_ctx1p0_crop160", help="Run tag used to locate weights.")
    p.add_argument(
        "--methods",
        default="baseline,final",
        help="Comma-separated methods to visualize qualitatively (weights resolved from runs/<ds>/).",
    )
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", default="0")
    p.add_argument("--half", action="store_true", help="Use FP16 inference if supported.")
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy_runs_figures", action="store_true", help="Copy runs/figures/* into paperfig/figures/")
    p.add_argument("--copy_runs_tables", action="store_true", help="Copy runs/table_*.tex into paperfig/tables/")
    p.add_argument("--copy_debug_alignment", action="store_true", help="Copy runs/debug/alignment/* into paperfig/debug/")
    p.add_argument("--qualitative", action="store_true", help="Generate qualitative overlays (GT + preds) into paperfig/qual/")
    return p.parse_args()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> int:
    _safe_mkdir(dst_dir)
    n = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            n += 1
    return n


def _load_data_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset yaml: {path}")
    return data


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def _yolo_label_path(ds_root: Path, split: str, img_path: Path) -> Path:
    # Our prepared datasets always place labels in labels/<split>/<stem>.txt
    return ds_root / "labels" / split / f"{img_path.stem}.txt"


def _read_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Return list of GT boxes as pixel xyxy."""
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = parts
        xc = float(xc) * img_w
        yc = float(yc) * img_h
        w = float(w) * img_w
        h = float(h) * img_h
        x1 = max(0.0, xc - w / 2.0)
        y1 = max(0.0, yc - h / 2.0)
        x2 = min(float(img_w), xc + w / 2.0)
        y2 = min(float(img_h), yc + h / 2.0)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _draw_boxes(
    img_bgr: Any,
    boxes: list[tuple[float, float, float, float]],
    *,
    color: tuple[int, int, int],
    thickness: int = 2,
    label: str | None = None,
) -> Any:
    out = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(
                out,
                label,
                (max(0, x1), max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _draw_preds(img_bgr: Any, preds_xyxy: Any, confs: Any) -> Any:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), s in zip(preds_xyxy, confs):
        x1, y1, x2, y2 = [int(round(float(v))) for v in (x1, y1, x2, y2)]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"{float(s):.2f}",
            (max(0, x1), max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _resolve_data_yaml(ds: str, sar_input: str) -> Path:
    if ds == "hrsid":
        p = _REPO_ROOT / "datasets" / "HRSID" / ("hrsid_gray1.yaml" if sar_input == "gray1" else "hrsid.yaml")
    elif ds == "ssdd":
        p = _REPO_ROOT / "datasets" / "SSDD" / ("ssdd_gray1.yaml" if sar_input == "gray1" else "ssdd.yaml")
    else:
        raise ValueError(f"Unknown dataset: {ds}")
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _resolve_weights(ds: str, method: str, sar_input: str, tag: str) -> Path:
    run = _REPO_ROOT / "runs" / ds / f"{method}_yolov8n_{sar_input}_{tag}" / "weights" / "best.pt"
    if run.exists():
        return run
    # fallback to last.pt
    run_last = run.parent / "last.pt"
    if run_last.exists():
        return run_last
    raise FileNotFoundError(f"Missing weights for ds={ds} method={method}: {run} (or {run_last})")


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))

    out_root = (_REPO_ROOT / args.outdir).resolve()
    _safe_mkdir(out_root)

    # 1) Copy existing paper-ready plots/tables/debug artifacts
    if args.copy_runs_figures:
        n = _copy_glob(_REPO_ROOT / "runs" / "figures", "*.png", out_root / "figures")
        print(f"[OK] copied runs/figures/*.png -> {out_root/'figures'} (n={n})")

    if args.copy_runs_tables:
        tables_dir = out_root / "tables"
        _safe_mkdir(tables_dir)
        n = 0
        for pat in ("table_*.tex", "table_*.json", "table_*.csv"):
            n += _copy_glob(_REPO_ROOT / "runs", pat, tables_dir)
        # Also include cost table artifacts
        for f in ("table_cost.tex", "cost.json", "exp_meta.txt"):
            src = _REPO_ROOT / "runs" / f
            if src.exists():
                shutil.copy2(src, tables_dir / src.name)
                n += 1
        print(f"[OK] copied runs/tables+meta -> {tables_dir} (n~={n})")

    if args.copy_debug_alignment:
        src = _REPO_ROOT / "runs" / "debug" / "alignment"
        if src.exists():
            dst = out_root / "debug" / "alignment"
            _safe_mkdir(dst)
            n = 0
            for pat in ("*.png", "*.json"):
                n += _copy_glob(src, pat, dst)
            # also copy roi_align_debug.txt if exists
            extra = _REPO_ROOT / "runs" / "debug" / "roi_align_debug.txt"
            if extra.exists():
                shutil.copy2(extra, out_root / "debug" / extra.name)
                n += 1
            print(f"[OK] copied debug alignment -> {dst} (n~={n})")

    # 2) Qualitative: overlay GT + predictions
    if args.qualitative:
        from ultralytics import YOLO  # local ultralytics (method/yolov8) is on sys.path

        ds_list = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
        sar_input = str(args.sar_input)
        tag = str(args.tag)

        qual_root = out_root / "qual"
        _safe_mkdir(qual_root)

        manifest: dict[str, Any] = {
            "meta": {
                "sar_input": sar_input,
                "datasets": ds_list,
                "methods": methods,
                "tag": tag,
                "imgsz": int(args.imgsz),
                "conf": float(args.conf),
                "iou": float(args.iou),
                "device": str(args.device),
                "half": bool(args.half),
                "seed": int(args.seed),
                "num_images": int(args.num_images),
            },
            "items": [],
        }

        for ds in ds_list:
            data_yaml = _resolve_data_yaml(ds, sar_input)
            data = _load_data_yaml(data_yaml)
            ds_root = Path(str(data["path"])).expanduser().resolve()
            images_dir = (ds_root / str(data["val"])).resolve()
            files = _list_images(images_dir)
            if not files:
                print(f"[WARN] no images found: {images_dir}")
                continue

            pick = files if len(files) <= int(args.num_images) else random.sample(files, int(args.num_images))
            pick = sorted(pick)

            # Load models once
            yolo_models: dict[str, Any] = {}
            for method in methods:
                w = _resolve_weights(ds, method, sar_input, tag)
                yolo_models[method] = YOLO(str(w))

            ds_out = qual_root / ds
            _safe_mkdir(ds_out)

            # per-method dirs
            for method in methods:
                _safe_mkdir(ds_out / method)
            # optional compare (baseline vs final)
            compare_dir = ds_out / "compare_baseline_final"
            if "baseline" in methods and "final" in methods:
                _safe_mkdir(compare_dir)

            for img_path in pick:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                h, w = img.shape[:2]
                gt_boxes = _read_yolo_labels(_yolo_label_path(ds_root, "val", img_path), img_w=w, img_h=h)

                rendered: dict[str, Path] = {}
                for method in methods:
                    m = yolo_models[method]
                    r = m.predict(
                        source=str(img_path),
                        imgsz=int(args.imgsz),
                        conf=float(args.conf),
                        iou=float(args.iou),
                        max_det=500,
                        device=str(args.device),
                        half=bool(args.half),
                        save=False,
                        verbose=False,
                    )[0]
                    out_img = img.copy()
                    # GT in green
                    out_img = _draw_boxes(out_img, gt_boxes, color=(0, 255, 0), thickness=2, label="GT")
                    # preds in red
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None and boxes.xyxy is not None and boxes.conf is not None:
                        out_img = _draw_preds(out_img, boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())

                    out_path = (ds_out / method / f"{img_path.stem}.png").resolve()
                    cv2.imwrite(str(out_path), out_img)
                    rendered[method] = out_path

                if "baseline" in rendered and "final" in rendered:
                    a = cv2.imread(str(rendered["baseline"]))
                    b = cv2.imread(str(rendered["final"]))
                    if a is not None and b is not None and a.shape == b.shape:
                        combo = cv2.hconcat([a, b])
                        cv2.putText(
                            combo,
                            "baseline",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            combo,
                            "final",
                            (a.shape[1] + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        out_path = (compare_dir / f"{img_path.stem}.png").resolve()
                        cv2.imwrite(str(out_path), combo)

                manifest["items"].append(
                    {
                        "dataset": ds,
                        "image": str(img_path),
                        "outputs": {k: str(v) for k, v in rendered.items()},
                    }
                )

        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] qualitative -> {qual_root}")
        print(f"[OK] manifest -> {out_root/'manifest.json'}")


if __name__ == "__main__":
    os.chdir(_REPO_ROOT)
    main()


