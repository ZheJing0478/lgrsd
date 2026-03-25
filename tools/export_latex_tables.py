from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRIC_COLS = ["AP", "AP50", "AP75", "APS", "APM", "APL"]


METHOD_SPECS = {
    "baseline": ("Ultralytics", "YOLOv8{scale}({sar})"),
    "attn": ("Ours", "YOLOv8{scale}+AttnFPN({sar})"),
    "lgrsd": ("Ours", "YOLOv8{scale}+LG-RSD({sar})"),
    "final": ("Ours", "YOLOv8{scale}+LG-RSD+AttnFPN({sar})"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export strict-COCO metrics in runs/summary.json to LaTeX tables and plots.")
    p.add_argument("--summary", default="runs/summary.json", help="Path to runs/summary.json")
    p.add_argument("--outdir", default="runs", help="Output directory (tables + figures)")
    p.add_argument("--model_scale", default="n", choices=["n", "s", "m", "l", "x"], help="Model scale used in run names.")
    p.add_argument("--sar_inputs", default="rgb3,gray1", help="Comma-separated SAR input protocols to export.")
    p.add_argument(
        "--filter_tag",
        default="",
        help="If set, only consider summary exp_keys that contain this substring (prevents mixing protocols). "
        "Example: --filter_tag b1_1024 or --filter_tag p2final_1024",
    )
    p.add_argument("--percent", action="store_true", help="Format numbers as percentage (x100). Default True.")
    p.add_argument("--no_plots", action="store_true", help="Disable curve plotting.")
    return p.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary.json must be a dict")
    return data


def fmt(v: float, as_percent: bool) -> str:
    if math.isnan(v):
        return "-"
    if as_percent:
        return f"{v * 100.0:.1f}"
    return f"{v:.4f}"


def to_latex_table(rows: list[dict[str, Any]], *, as_percent: bool) -> str:
    # Best per column
    best: dict[str, float] = {}
    for c in METRIC_COLS:
        vals = [float(r.get(c, float("nan"))) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        best[c] = max(vals) if vals else float("nan")

    header = [
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Method & Year/Reference & AP & AP50 & AP75 & APS & APM & APL \\",
        r"\midrule",
    ]
    body: list[str] = []
    for r in rows:
        cells = [r["Method"], r["Year/Reference"]]
        for c in METRIC_COLS:
            v = float(r.get(c, float("nan")))
            s = fmt(v, as_percent=as_percent)
            if not math.isnan(v) and not math.isnan(best[c]) and abs(v - best[c]) <= 1e-12:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        body.append(" & ".join(cells) + r" \\")
    footer = [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(header + body + footer) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["Method"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_results_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _exp_key_to_results_csv(outdir: Path, exp_key: str) -> Path:
    # exp_key: "<ds>/<run_name>"
    if "/" not in exp_key:
        return outdir / exp_key / "results.csv"
    ds, run_name = exp_key.split("/", 1)
    return outdir / ds / run_name / "results.csv"


def _suffix_for_sar(sar: str) -> str:
    return "" if sar == "rgb3" else f"_{sar}"


def plot_curves(
    outdir: Path, summary: dict[str, Any], *, scale: str, sar_inputs: list[str], filter_tag: str = ""
) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    for ds in ("hrsid", "ssdd"):
        for sar in sar_inputs:
            suffix = _suffix_for_sar(sar)

            # Plot mAP50-95(B) curves for the best run per method (prevents stale hard-coded run names).
            plt.figure(figsize=(7, 4))
            any_plotted = False
            for method in ("baseline", "attn", "lgrsd", "final"):
                k = _pick_best_key(summary, ds=ds, method=method, scale=scale, sar_input=sar, filter_tag=filter_tag)
                if not k:
                    continue
                csv_path = _exp_key_to_results_csv(outdir, k)
                rows = read_results_csv(csv_path)
                if not rows:
                    continue
                label = f"{method}({sar})"
                x = [int(float(r.get("epoch", "0"))) for r in rows]
                y = [float(r.get("metrics/mAP50-95(B)", "nan")) for r in rows]
                plt.plot(x, y, marker="o", linewidth=1.5, label=label)
                any_plotted = True
            if any_plotted:
                plt.title(f"{ds.upper()} mAP50-95 vs epoch (Ultralytics val)")
                plt.xlabel("epoch")
                plt.ylabel("mAP50-95")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(figdir / f"{ds}_ap_curve{suffix}.png", dpi=200)
            plt.close()

            # Plot loss curves (box + lgrsd if present) for the best run per method.
            plt.figure(figsize=(8, 4))
            any_plotted = False
            for method in ("baseline", "attn", "lgrsd", "final"):
                k = _pick_best_key(summary, ds=ds, method=method, scale=scale, sar_input=sar, filter_tag=filter_tag)
                if not k:
                    continue
                csv_path = _exp_key_to_results_csv(outdir, k)
                rows = read_results_csv(csv_path)
                if not rows:
                    continue
                label = f"{method}({sar})"
                x = [int(float(r.get("epoch", "0"))) for r in rows]
                y = [float(r.get("train/box_loss", "nan")) for r in rows]
                plt.plot(x, y, marker="o", linewidth=1.5, label=f"{label}:box")
                any_plotted = True
                if "train/lgrsd_loss" in rows[0]:
                    y2 = [float(r.get("train/lgrsd_loss", "nan")) for r in rows]
                    plt.plot(x, y2, marker="x", linewidth=1.0, linestyle="--", label=f"{label}:lgrsd")
            if any_plotted:
                plt.title(f"{ds.upper()} train loss vs epoch")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                plt.savefig(figdir / f"{ds}_loss_curve{suffix}.png", dpi=200)
            plt.close()


def _parse_lambda_from_key(exp_key: str) -> float | None:
    # Accept "..._lam0p1", "..._lam1p0" etc.
    m = re.search(r"_lam(\d+)p(\d+)", exp_key)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def plot_lambda_sweep(
    summary: dict[str, Any], outdir: Path, *, scale: str, sar_inputs: list[str], filter_tag: str = ""
) -> None:
    """Plot AP/APS vs lambda for any runs that follow the *_lamX pattern (quick-sweep runs)."""
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Build per-dataset data. We support both lgrsd and final sweeps.
    sweep_methods = ("lgrsd", "final")
    ds_list = ("hrsid", "ssdd")

    # Collect
    ds_data: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    # key: (ds, sar, method) -> list of (lambda, AP, APS)
    for ds in ds_list:
        for sar in sar_inputs:
            for method in sweep_methods:
                prefix = f"{ds}/{method}_yolov8{scale}"
                candidates = [k for k in summary.keys() if isinstance(k, str) and k.startswith(prefix) and "_lam" in k]
                if filter_tag:
                    candidates = [k for k in candidates if filter_tag in k]
                # Filter by sar_input when possible (legacy keys may not encode sar; then use summary field).
                filtered: list[str] = []
                for k in candidates:
                    m = summary.get(k, {})
                    sar_k = m.get("sar_input") if isinstance(m, dict) else None
                    if isinstance(sar_k, str):
                        if sar_k == sar:
                            filtered.append(k)
                    else:
                        # Legacy: no sar_input recorded => assume rgb3 only.
                        if sar == "rgb3":
                            filtered.append(k)
                candidates = filtered
                pts: list[tuple[float, float, float]] = []
                for k in candidates:
                    lam = _parse_lambda_from_key(k)
                    if lam is None:
                        continue
                    m = summary.get(k, {})
                    if not isinstance(m, dict):
                        continue
                    ap = float(m.get("AP", float("nan")))
                    aps = float(m.get("APS", float("nan")))
                    if math.isnan(ap) or math.isnan(aps):
                        continue
                    pts.append((lam, ap, aps))
                if pts:
                    pts.sort(key=lambda t: t[0])
                    ds_data[(ds, sar, method)] = pts

    if not ds_data:
        return

    # Plot (combined figure): 2 rows (AP/APS) x 2 cols (HRSID/SSDD) for rgb3 only by default.
    # If only one dataset exists, still emit a single figure with available panels.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), squeeze=False)
    for ci, ds in enumerate(ds_list):
        for ri, metric_name in enumerate(("AP", "APS")):
            ax = axes[ri][ci]
            any_line = False
            for method in sweep_methods:
                # Prefer rgb3 for the sweep plot unless user explicitly wants gray1 sweep too.
                sar = "rgb3" if "rgb3" in sar_inputs else sar_inputs[0]
                pts = ds_data.get((ds, sar, method))
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] if metric_name == "AP" else p[2] for p in pts]
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{method}({sar})")
                any_line = True
            ax.set_title(f"{ds.upper()} {metric_name} vs lgrsd_lambda (strict COCO)")
            ax.set_xlabel("lgrsd_lambda")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            if any_line:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figdir / "lambda_sweep_ap_curve.png", dpi=200)
    plt.close()


def _pick_best_key(
    summary: dict[str, Any], *, ds: str, method: str, scale: str, sar_input: str, filter_tag: str = ""
) -> str | None:
    # Preferred (A7): run_name encodes sar_input, e.g. baseline_yolov8n_rgb3
    prefix = f"{ds}/{method}_yolov8{scale}_{sar_input}"
    candidates = [k for k in summary.keys() if isinstance(k, str) and k.startswith(prefix)]
    # Backward-compatible fallback: older runs may not encode sar_input in key (assume rgb3 only).
    if not candidates and sar_input == "rgb3":
        legacy_prefix = f"{ds}/{method}_yolov8{scale}"
        candidates = [k for k in summary.keys() if isinstance(k, str) and k.startswith(legacy_prefix)]
    if not candidates:
        return None
    # Filter by sar_input if present in summary entries; assume missing==rgb3 (legacy).
    filtered: list[str] = []
    for k in candidates:
        m = summary.get(k, {})
        sar = None
        if isinstance(m, dict):
            sar = m.get("sar_input")
        sar = sar if isinstance(sar, str) else ("rgb3" if sar_input == "rgb3" else None)
        if sar == sar_input:
            filtered.append(k)
    candidates = filtered
    if not candidates:
        return None
    if filter_tag:
        candidates = [k for k in candidates if filter_tag in k]
        if not candidates:
            return None
    # pick highest AP
    def _ap(k: str) -> float:
        try:
            return float(summary[k].get("AP", float("-inf")))
        except Exception:
            return float("-inf")
    return max(candidates, key=_ap)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    as_percent = True if args.percent is False else True  # default True

    sar_inputs = [s.strip() for s in str(args.sar_inputs).split(",") if s.strip()]
    scale = str(args.model_scale)
    filter_tag = str(getattr(args, "filter_tag", "")).strip()

    export_rows_all: dict[str, list[dict[str, Any]]] = {}
    for ds in ("hrsid", "ssdd"):
        for sar in sar_inputs:
            rows: list[dict[str, Any]] = []
            for method, (ref, templ) in METHOD_SPECS.items():
                k = _pick_best_key(summary, ds=ds, method=method, scale=scale, sar_input=sar, filter_tag=filter_tag)
                if not k:
                    continue
                m = summary.get(k)
                if not isinstance(m, dict):
                    continue
                row = {"Method": templ.format(scale=scale, sar=sar), "Year/Reference": ref, "ExpKey": k, "sar_input": sar}
                for c in METRIC_COLS:
                    row[c] = float(m.get(c, float("nan")))
                rows.append(row)

            suffix = "" if sar == "rgb3" else f"_{sar}"
            tex_path = outdir / f"table_{ds}_quant{suffix}.tex"
            csv_path = outdir / f"table_{ds}_quant{suffix}.csv"
            json_path = outdir / f"table_{ds}_quant{suffix}.json"

            if not rows:
                # Remove stale outputs if they exist (prevents accidental protocol mixing).
                for p in (tex_path, csv_path, json_path):
                    if p.exists():
                        p.unlink()
                continue

            export_rows_all[f"{ds}:{sar}"] = rows

            tex = to_latex_table(rows, as_percent=as_percent)
            tex_path.write_text(tex, encoding="utf-8")

            write_csv(csv_path, rows)
            json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

            print(f"[OK] {tex_path}")

    # Global export (raw summary snapshot)
    (outdir / "summary_export.json").write_text(json.dumps(export_rows_all, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.no_plots:
        plot_curves(outdir, summary, scale=scale, sar_inputs=sar_inputs, filter_tag=filter_tag)
        plot_lambda_sweep(summary, outdir, scale=scale, sar_inputs=sar_inputs, filter_tag=filter_tag)
        print(f"[OK] curves -> {outdir/'figures'}")


if __name__ == "__main__":
    main()


