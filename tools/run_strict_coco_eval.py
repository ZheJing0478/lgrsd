from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Ensure project-local packages (e.g., ultralytics_ext/) are importable when running as `python tools/...`.
    sys.path.insert(0, str(_REPO_ROOT))

import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


@dataclass
class EvalArgs:
    model: str
    data: str
    split: str
    imgsz: int
    device: str
    conf: float
    iou: float
    max_det: int
    batch: int
    half: bool
    outdir: str
    name: str
    gt_json_override: str | None
    exp_key: str | None
    summary_json: str


def _load_data_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data yaml: {path}")
    return data


def _list_images(images_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    return sorted(files)


def _coco_file_name_key(file_name: str) -> str:
    # Robust to 'subdir/xxx.jpg' in COCO json.
    return Path(file_name).name


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _update_summary(summary_path: Path, exp_key: str, metrics: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[exp_key] = metrics
    _save_json(summary_path, data)


def parse_args() -> EvalArgs:
    p = argparse.ArgumentParser(description="Strict COCOeval for Ultralytics YOLOv8 predictions (bbox).")
    p.add_argument("--model", required=True, help="Path to .pt or model spec")
    p.add_argument("--data", required=True, help="Ultralytics dataset yaml (with coco_val key)")
    p.add_argument("--split", default="val", choices=["val", "test", "train"])
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max_det", type=int, default=500)
    p.add_argument("--batch", type=int, default=1, help="Inference batch size for prediction. Keep small to avoid OOM.")
    p.add_argument("--half", action="store_true", help="Use FP16 inference (recommended on CUDA).")
    p.add_argument("--outdir", default="runs")
    p.add_argument("--name", default="strict_eval")
    p.add_argument("--gt_json_override", default=None)
    p.add_argument("--exp_key", default=None, help="If set, write metrics into summary_json under this key.")
    p.add_argument("--summary_json", default="runs/summary.json")
    a = p.parse_args()
    return EvalArgs(**vars(a))


def main() -> None:
    args = parse_args()

    data_yaml = Path(args.data).resolve()
    data = _load_data_yaml(data_yaml)
    ds_root = Path(str(data.get("path", ""))).expanduser().resolve()
    split_key = {"train": "train", "val": "val", "test": "test"}.get(args.split, "val")
    split_rel = data.get(split_key)
    if not split_rel:
        raise KeyError(f"Missing '{split_key}' in dataset yaml: {data_yaml}")
    images_dir = (ds_root / str(split_rel)).resolve() if not str(split_rel).startswith(("/", "~")) else Path(str(split_rel)).expanduser().resolve()

    # GT json
    gt_json = args.gt_json_override or data.get("coco_val") or data.get("coco_gt") or None
    if not gt_json:
        raise KeyError("Dataset yaml must contain 'coco_val' (or pass --gt_json_override).")
    gt_json_path = Path(str(gt_json)).expanduser().resolve()

    out_root = Path(args.outdir).expanduser().resolve() / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    _save_json(out_root / "eval_args.json", asdict(args))

    coco_gt = COCO(str(gt_json_path))
    # Map file_name -> image_id
    file_to_imgid: dict[str, int] = {}
    for im in coco_gt.dataset.get("images", []):
        key = _coco_file_name_key(str(im.get("file_name", "")))
        if key:
            file_to_imgid[key] = int(im["id"])

    # A6: Map class index -> COCO category_id using explicit label_map.json (preferred).
    cat_ids = sorted([int(c["id"]) for c in coco_gt.dataset.get("categories", [])])
    if not cat_ids:
        raise RuntimeError(f"No categories in GT json: {gt_json_path}")

    label_map_path = data.get("label_map") or str((ds_root / "label_map.json").resolve())
    class_to_cat: dict[int, int] | None = None
    label_map_note = ""
    try:
        p = Path(str(label_map_path)).expanduser().resolve()
        if p.exists():
            lm = json.loads(p.read_text(encoding="utf-8"))
            m = lm.get("class_to_category_id", {})
            if isinstance(m, dict) and m:
                class_to_cat = {int(k): int(v) for k, v in m.items()}
                label_map_note = f"label_map={p}"
    except Exception as e:
        label_map_note = f"label_map_load_error={e}"

    if class_to_cat is None:
        # Fallback: by sorting COCO category ids
        if len(cat_ids) != int(data.get("nc", 1)):
            if int(data.get("nc", 1)) == 1:
                class_to_cat = {0: cat_ids[0]}
            else:
                raise RuntimeError(f"Category count mismatch: nc={data.get('nc')} vs GT={len(cat_ids)}")
        else:
            class_to_cat = {i: cid for i, cid in enumerate(cat_ids)}
        label_map_note = "label_map=missing_fallback_by_sorted_cat_ids"

    # Validate mapping against GT categories and emit a debug check file.
    bad = [cid for cid in class_to_cat.values() if int(cid) not in set(cat_ids)]
    ok = not bad
    debug_dir = Path("runs/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "label_map_check.txt").write_text(
        "\n".join(
            [
                f"data_yaml: {data_yaml}",
                f"gt_json: {gt_json_path}",
                f"gt_category_ids: {cat_ids}",
                f"class_to_category_id: {class_to_cat}",
                f"status: {'PASS' if ok else 'FAIL'}",
                f"note: {label_map_note}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if not ok:
        raise RuntimeError(f"label_map contains category_id not in GT: {bad}")

    img_paths = _list_images(images_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {images_dir}")

    model = YOLO(args.model)

    preds: list[dict[str, Any]] = []
    # stream=True to avoid holding all results in memory
    for r in model.predict(
        source=str(images_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        half=bool(args.half),
        device=args.device,
        stream=True,
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        verbose=False,
    ):
        file_key = Path(str(r.path)).name
        image_id = file_to_imgid.get(file_key)
        if image_id is None:
            # Skip if not in GT (should not happen for our prepared datasets)
            continue
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.data is None:
            continue
        # boxes.xyxy, boxes.conf, boxes.cls are torch tensors
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), score, cls in zip(xyxy, confs, clss):
            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 0 or h <= 0:
                continue
            cat_id = class_to_cat.get(int(cls), cat_ids[0])
            preds.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [float(x1), float(y1), w, h],
                    "score": float(score),
                }
            )

    pred_path = out_root / "predictions.json"
    _save_json(pred_path, preds)

    if not preds:
        # pycocotools cannot loadRes() on an empty list (it indexes anns[0]).
        metrics = {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APS": 0.0,
            "APM": 0.0,
            "APL": 0.0,
            "AR1": 0.0,
            "AR10": 0.0,
            "ARmax": 0.0,
            "ARS": 0.0,
            "ARM": 0.0,
            "ARL": 0.0,
            "sar_input": str(data.get("sar_input", "rgb3")),
            "channels": int(data.get("channels", 3)),
            "data_yaml": str(data_yaml),
            "gt_json": str(gt_json_path),
            "predictions_json": str(pred_path),
            "num_predictions": 0,
            "note": "empty_predictions",
        }
        _save_json(out_root / "metrics.json", metrics)
        if args.exp_key:
            _update_summary(Path(args.summary_json).expanduser().resolve(), args.exp_key, metrics)
        print(f"[OK] empty predictions -> {out_root/'metrics.json'}")
        return

    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    # IMPORTANT:
    # pycocotools COCOeval's standard AP (mAP@[.50:.95]) is defined with maxDets=100.
    # We keep COCOeval maxDets to [1, 10, 100] for canonical AP/AP50/AP75/APS/APM/APL,
    # while `args.max_det` controls inference-time max predictions per image.
    coco_eval.params.maxDets = [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCOeval.stats = [AP, AP50, AP75, APS, APM, APL, AR1, AR10, AR100, ARS, ARM, ARL]
    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "APS": float(coco_eval.stats[3]),
        "APM": float(coco_eval.stats[4]),
        "APL": float(coco_eval.stats[5]),
        "AR1": float(coco_eval.stats[6]),
        "AR10": float(coco_eval.stats[7]),
        "ARmax": float(coco_eval.stats[8]),
        "ARS": float(coco_eval.stats[9]),
        "ARM": float(coco_eval.stats[10]),
        "ARL": float(coco_eval.stats[11]),
        # A7: record SAR input protocol to prevent “口径漂移”
        "sar_input": str(data.get("sar_input", "rgb3")),
        "channels": int(data.get("channels", 3)),
        "data_yaml": str(data_yaml),
        "gt_json": str(gt_json_path),
        "predictions_json": str(pred_path),
        "num_predictions": len(preds),
    }
    _save_json(out_root / "metrics.json", metrics)

    # Optional global summary
    if args.exp_key:
        _update_summary(Path(args.summary_json).expanduser().resolve(), args.exp_key, metrics)

    print(f"[OK] metrics.json -> {out_root/'metrics.json'}")


if __name__ == "__main__":
    main()


