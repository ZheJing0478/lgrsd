#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".npy":
        arr = np.load(p)
        return np.asarray(arr, dtype=np.float32)

    if suf == ".npz":
        data = np.load(p)
        # prefer common keys
        for k in ["cos", "cosine", "similarity", "sims"]:
            if k in data:
                return np.asarray(data[k], dtype=np.float32)
        # otherwise take first key
        first_key = list(data.keys())[0]
        return np.asarray(data[first_key], dtype=np.float32)

    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            for k in ["cos", "cosine", "similarity", "sims"]:
                if k in obj:
                    return np.asarray(obj[k], dtype=np.float32)
            # take first value
            return np.asarray(next(iter(obj.values())), dtype=np.float32)
        return np.asarray(obj, dtype=np.float32)

    if suf == ".csv":
        # lightweight CSV reader (no pandas dependency)
        import csv
        vals = []
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            # pick a reasonable column
            cand = None
            for name in ["cos", "cosine", "similarity", "sim"]:
                if name in cols:
                    cand = name
                    break
            if cand is None:
                # fallback: if only 1 column, use it
                if len(cols) == 1:
                    cand = cols[0]
                else:
                    raise ValueError(
                        f"{p}: cannot infer cosine column name. "
                        f"Expected one of cos/cosine/similarity/sim, got {cols}"
                    )
            for row in reader:
                try:
                    vals.append(float(row[cand]))
                except Exception:
                    continue
        return np.asarray(vals, dtype=np.float32)

    raise ValueError(f"Unsupported file format: {p} (supported: .npy/.npz/.json/.csv)")


def _clean(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    # cosine should be in [-1, 1], clip for safety
    arr = np.clip(arr, -1.0, 1.0)
    return arr


def _summary(arr: np.ndarray) -> dict:
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "p25": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "p75": float(np.percentile(arr, 75)) if arr.size else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Plot histogram of cosine(z_loc, z_glob) for region consistency."
    )
    ap.add_argument("--baseline", required=True, help="Path to baseline cosine array (npy/npz/json/csv).")
    ap.add_argument("--method", required=True, help="Path to LG-RSD/Ours cosine array (npy/npz/json/csv).")
    ap.add_argument("--baseline_label", default="Baseline", help="Legend label for baseline.")
    ap.add_argument("--method_label", default="LG-RSD / Ours", help="Legend label for method.")
    ap.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    ap.add_argument("--range_min", type=float, default=-1.0, help="Histogram range min.")
    ap.add_argument("--range_max", type=float, default=1.0, help="Histogram range max.")
    ap.add_argument("--title", default="Region Semantic Consistency (cosine similarity)", help="Plot title.")
    ap.add_argument("--out", required=True, help="Output image path (e.g., runs/figs/region_consistency_hist.png).")
    ap.add_argument("--out_stats", default=None, help="Optional output JSON for summary stats.")
    ap.add_argument("--density", action="store_true", help="Plot probability density instead of counts.")
    args = ap.parse_args()

    base = _clean(_load_array(args.baseline))
    meth = _clean(_load_array(args.method))

    base_stats = _summary(base)
    meth_stats = _summary(meth)

    print("=== Baseline stats ===")
    print(json.dumps(base_stats, indent=2))
    print("=== Method stats ===")
    print(json.dumps(meth_stats, indent=2))

    if args.out_stats:
        out_stats_path = Path(args.out_stats)
        out_stats_path.parent.mkdir(parents=True, exist_ok=True)
        out_obj = {
            "baseline": {"path": args.baseline, "label": args.baseline_label, **base_stats},
            "method": {"path": args.method, "label": args.method_label, **meth_stats},
        }
        out_stats_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure()
    hist_range = (args.range_min, args.range_max)

    # Use default colors; only set alpha for overlay readability.
    plt.hist(base, bins=args.bins, range=hist_range, density=args.density, alpha=0.6, label=args.baseline_label)
    plt.hist(meth, bins=args.bins, range=hist_range, density=args.density, alpha=0.6, label=args.method_label)

    # Means (default line styles/colors)
    plt.axvline(base_stats["mean"], linestyle="--")
    plt.axvline(meth_stats["mean"], linestyle="--")

    plt.xlabel(r"$\cos(z_{\mathrm{loc}}, z_{\mathrm{glob}})$")
    plt.ylabel("Density" if args.density else "Count")
    plt.title(args.title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
