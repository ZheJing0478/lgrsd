from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CocoSpec:
    name: str
    train_json: Path
    val_json: Path
    train_images_dir: Path
    val_images_dir: Path
    class_name: str = "ship"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    ensure_dir(dst.parent)

    if dst.exists() or dst.is_symlink():
        # Idempotent: if existing symlink already points to src, do nothing.
        try:
            if dst.is_symlink() and dst.resolve() == src.resolve():
                return
        except FileNotFoundError:
            pass
        dst.unlink()

    if use_symlinks:
        rel = os.path.relpath(src, start=dst.parent)
        os.symlink(rel, dst)
    else:
        shutil.copy2(src, dst)


def clip_bbox_xywh(bbox: Any, img_w: int, img_h: int) -> tuple[float, float, float, float] | None:
    """Clip COCO bbox [x, y, w, h] to image boundary; return None if invalid."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None

    x, y, w, h = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_w), x + w)
    y2 = min(float(img_h), y + h)
    w2 = x2 - x1
    h2 = y2 - y1
    if w2 <= 1e-6 or h2 <= 1e-6:
        return None
    return x1, y1, w2, h2


def xywh_to_yolo(
    x: float, y: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    xc = (x + w / 2.0) / float(img_w)
    yc = (y + h / 2.0) / float(img_h)
    wn = w / float(img_w)
    hn = h / float(img_h)

    # Clamp to [0, 1] for numerical safety.
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn))
    hn = max(0.0, min(1.0, hn))
    return xc, yc, wn, hn


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Minimal YAML writer for our simple dict/list structure (no external deps)."""
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {vv}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_split(
    coco_json: Path,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    *,
    use_symlinks: bool,
    overwrite: bool,
) -> dict[str, Any]:
    coco = read_json(coco_json)
    images = coco.get("images") or []
    annotations = coco.get("annotations") or []
    categories = coco.get("categories") or []

    # COCO category_id -> 0..nc-1 (deterministic by sorted COCO id)
    cat_id_to_name: dict[int, str] = {int(c["id"]): str(c.get("name", "")) for c in categories}
    if not cat_id_to_name:
        raise RuntimeError(f"No categories found in {coco_json}")
    sorted_cat_ids = sorted(cat_id_to_name)
    cat_id_to_cls = {cid: i for i, cid in enumerate(sorted_cat_ids)}

    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in annotations:
        if int(a.get("iscrowd", 0)) == 1:
            continue
        if int(a.get("ignore", 0)) == 1:
            continue
        anns_by_img[int(a["image_id"])].append(a)

    reset_dir(out_images_dir, overwrite=overwrite)
    reset_dir(out_labels_dir, overwrite=overwrite)

    n_images = 0
    n_written = 0
    n_skipped = 0

    for im in sorted(images, key=lambda d: int(d.get("id", 0))):
        image_id = int(im["id"])
        file_name = str(im["file_name"])
        img_w = int(im["width"])
        img_h = int(im["height"])

        src = images_dir / file_name
        if not src.exists():
            raise FileNotFoundError(f"Missing image file: {src} (from {coco_json})")
        dst = out_images_dir / file_name
        symlink_or_copy(src, dst, use_symlinks=use_symlinks)

        label_path = out_labels_dir / f"{Path(file_name).stem}.txt"
        lines: list[str] = []
        for a in anns_by_img.get(image_id, []):
            cid = int(a["category_id"])
            if cid not in cat_id_to_cls:
                n_skipped += 1
                continue
            clipped = clip_bbox_xywh(a.get("bbox"), img_w=img_w, img_h=img_h)
            if clipped is None:
                n_skipped += 1
                continue
            x, y, w, h = clipped
            xc, yc, wn, hn = xywh_to_yolo(x, y, w, h, img_w=img_w, img_h=img_h)
            cls = cat_id_to_cls[cid]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            n_written += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_images += 1

    return {
        "coco_json": str(coco_json),
        "images_dir": str(images_dir),
        "out_images_dir": str(out_images_dir),
        "out_labels_dir": str(out_labels_dir),
        "num_images": n_images,
        "num_annotations_total": len(annotations),
        "num_annotations_written": n_written,
        "num_annotations_skipped": n_skipped,
        "categories": [{"id": cid, "name": cat_id_to_name[cid], "cls": cat_id_to_cls[cid]} for cid in sorted_cat_ids],
    }


def prepare_dataset(spec: CocoSpec, out_root: Path, *, use_symlinks: bool, overwrite: bool) -> None:
    ds_root = out_root / spec.name.upper()
    ensure_dir(ds_root / "annotations")

    # Copy GT json for later strict COCOeval (keep original file_name/id/category_id).
    gt_train = ds_root / "annotations" / spec.train_json.name
    gt_val = ds_root / "annotations" / spec.val_json.name
    if overwrite or not gt_train.exists():
        shutil.copy2(spec.train_json, gt_train)
    if overwrite or not gt_val.exists():
        shutil.copy2(spec.val_json, gt_val)

    train_stats = convert_split(
        spec.train_json,
        spec.train_images_dir,
        ds_root / "images" / "train",
        ds_root / "labels" / "train",
        use_symlinks=use_symlinks,
        overwrite=overwrite,
    )
    val_stats = convert_split(
        spec.val_json,
        spec.val_images_dir,
        ds_root / "images" / "val",
        ds_root / "labels" / "val",
        use_symlinks=use_symlinks,
        overwrite=overwrite,
    )

    # A6: write explicit label map (YOLO cls -> COCO category_id) for strict COCOeval consistency.
    cats_train = train_stats.get("categories", [])
    cats_val = val_stats.get("categories", [])
    if cats_train and cats_val and [c["id"] for c in cats_train] != [c["id"] for c in cats_val]:
        raise RuntimeError(f"Train/val category mismatch for {spec.name}: {cats_train} vs {cats_val}")
    label_map_path = ds_root / "label_map.json"
    class_to_cat = {int(c["cls"]): int(c["id"]) for c in cats_train} if cats_train else {0: 1}
    cat_to_name = {int(c["id"]): str(c.get("name", "")) for c in cats_train} if cats_train else {1: spec.class_name}
    label_map = {
        "dataset": spec.name,
        "class_to_category_id": class_to_cat,
        "category_id_to_name": cat_to_name,
        "source_train_json": str(spec.train_json),
        "source_val_json": str(spec.val_json),
    }
    if overwrite or not label_map_path.exists():
        label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    yaml_path = ds_root / f"{spec.name}.yaml"
    yaml_data: dict[str, Any] = {
        "path": str(ds_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: spec.class_name},
        "label_map": str(label_map_path.resolve()),
        # Extra keys for our strict COCOeval script.
        "coco_train": str(gt_train.resolve()),
        "coco_val": str(gt_val.resolve()),
        # Default SAR input strategy (A): replicate to 3 channels (no backbone change).
        "sar_input": "rgb3",
    }
    write_yaml(yaml_path, yaml_data)

    # Optional SAR input strategy (B): grayscale 1-channel (requires 'channels: 1').
    # Ultralytics will read images with cv2.IMREAD_GRAYSCALE and build the model with ch=1.
    yaml_gray1_path = ds_root / f"{spec.name}_gray1.yaml"
    yaml_gray1_data = dict(yaml_data)
    yaml_gray1_data["channels"] = 1
    yaml_gray1_data["sar_input"] = "gray1"
    write_yaml(yaml_gray1_path, yaml_gray1_data)

    stats_path = ds_root / "stats.json"
    stats = {
        "dataset": spec.name,
        "train": train_stats,
        "val": val_stats,
        "yaml_rgb3": str(yaml_path.resolve()),
        "yaml_gray1": str(yaml_gray1_path.resolve()),
    }
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OK] {spec.name}: {yaml_path}")


def get_default_specs(repo_root: Path) -> dict[str, CocoSpec]:
    hrsid_root = repo_root / "dataset" / "sar_ship_datasets" / "HRSID"
    ssdd_root = (
        repo_root
        / "dataset"
        / "sar_ship_datasets"
        / "SSDD"
        / "Official-SSDD-OPEN"
        / "BBox_SSDD"
        / "coco_style"
    )
    return {
        "hrsid": CocoSpec(
            name="hrsid",
            train_json=hrsid_root / "annotations" / "train2017.json",
            val_json=hrsid_root / "annotations" / "test2017.json",
            train_images_dir=hrsid_root / "images",
            val_images_dir=hrsid_root / "images",
        ),
        "ssdd": CocoSpec(
            name="ssdd",
            train_json=ssdd_root / "annotations" / "train.json",
            val_json=ssdd_root / "annotations" / "test.json",
            train_images_dir=ssdd_root / "images" / "train",
            val_images_dir=ssdd_root / "images" / "test",
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HRSID/SSDD datasets for Ultralytics YOLOv8.")
    p.add_argument("--dataset", choices=["hrsid", "ssdd", "all"], default="all")
    p.add_argument("--out_root", default="datasets", help="Output root directory, default: datasets/")
    p.add_argument("--copy", action="store_true", help="Copy images instead of symlinking (uses more disk).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared dataset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    specs = get_default_specs(repo_root)
    targets = ["hrsid", "ssdd"] if args.dataset == "all" else [args.dataset]
    for name in targets:
        prepare_dataset(specs[name], out_root, use_symlinks=not args.copy, overwrite=args.overwrite)


if __name__ == "__main__":
    main()


