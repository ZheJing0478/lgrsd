from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


LAM_RE = re.compile(r"_lam(\d+)p(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize lambda sweep runs from runs/summary.json.")
    p.add_argument("--summary", default="runs/summary.json")
    p.add_argument("--outdir", default="runs/debug")
    p.add_argument("--datasets", default="hrsid,ssdd")
    p.add_argument("--method", default="final", choices=["lgrsd", "final"])
    p.add_argument("--model_scale", default="n")
    p.add_argument("--sar_input", default="rgb3", choices=["rgb3", "gray1"])
    p.add_argument("--filter_tag", default="b2_1024", help="Only consider exp_keys containing this substring.")
    p.add_argument("--primary_metric", default="APS", choices=["APS", "AP"], help="Primary metric for ranking.")
    p.add_argument(
        "--score",
        default="mean_ap_plus_aps_minus_std",
        choices=["mean_aps", "mean_ap", "mean_ap_plus_aps", "mean_ap_plus_aps_minus_std"],
        help="How to pick best lambda when multiple datasets are present.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict json: {path}")
    return data


def _parse_lambda(exp_key: str) -> float | None:
    m = LAM_RE.search(exp_key)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, float) and not math.isnan(x) and not math.isinf(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    xs = [x for x in xs if isinstance(x, float) and not math.isnan(x) and not math.isinf(x)]
    if len(xs) <= 1:
        return 0.0 if xs else float("nan")
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def main() -> None:
    args = parse_args()
    summary = _load_json(Path(args.summary).expanduser().resolve())
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    method = str(args.method)
    scale = str(args.model_scale)
    sar = str(args.sar_input)
    tag = str(args.filter_tag).strip()

    # Collect: lambda -> dataset -> metrics
    data: dict[float, dict[str, dict[str, float]]] = {}
    for ds in datasets:
        prefix = f"{ds}/{method}_yolov8{scale}_{sar}"
        keys = [k for k in summary.keys() if isinstance(k, str) and k.startswith(prefix) and "_lam" in k]
        if tag:
            keys = [k for k in keys if tag in k]
        for k in keys:
            lam = _parse_lambda(k)
            if lam is None:
                continue
            m = summary.get(k, {})
            if not isinstance(m, dict):
                continue
            ap = float(m.get("AP", float("nan")))
            aps = float(m.get("APS", float("nan")))
            ap50 = float(m.get("AP50", float("nan")))
            ap75 = float(m.get("AP75", float("nan")))
            apm = float(m.get("APM", float("nan")))
            apl = float(m.get("APL", float("nan")))
            data.setdefault(lam, {})[ds] = {
                "AP": ap,
                "APS": aps,
                "AP50": ap50,
                "AP75": ap75,
                "APM": apm,
                "APL": apl,
            }

    # Flatten rows
    rows: list[dict[str, Any]] = []
    for lam in sorted(data.keys()):
        for ds in datasets:
            m = data[lam].get(ds)
            if not m:
                continue
            rows.append({"dataset": ds, "lambda": lam, **m})

    # Aggregate & pick best
    lam_scores: dict[float, dict[str, float]] = {}
    for lam in sorted(data.keys()):
        aps_vals = [data[lam].get(ds, {}).get("APS", float("nan")) for ds in datasets]
        ap_vals = [data[lam].get(ds, {}).get("AP", float("nan")) for ds in datasets]
        m_aps = _mean([float(x) for x in aps_vals])
        m_ap = _mean([float(x) for x in ap_vals])
        s_aps = _std([float(x) for x in aps_vals])
        s_ap = _std([float(x) for x in ap_vals])

        if args.score == "mean_aps":
            score = m_aps
        elif args.score == "mean_ap":
            score = m_ap
        elif args.score == "mean_ap_plus_aps":
            score = m_ap + m_aps
        else:
            # prefer high mean AP+APS and low variance across datasets
            score = (m_ap + m_aps) - 0.5 * (s_ap + s_aps)

        lam_scores[lam] = {
            "mean_AP": m_ap,
            "mean_APS": m_aps,
            "std_AP": s_ap,
            "std_APS": s_aps,
            "score": score,
        }

    best_lam = max(lam_scores.keys(), key=lambda x: lam_scores[x]["score"]) if lam_scores else None

    out_json = {
        "meta": {
            "datasets": datasets,
            "method": method,
            "model_scale": scale,
            "sar_input": sar,
            "filter_tag": tag,
            "score": args.score,
        },
        "per_lambda": lam_scores,
        "best_lambda": best_lam,
        "rows": rows,
    }
    json_path = outdir / f"lambda_sweep_{method}_{sar}_{tag}.json"
    json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = outdir / f"lambda_sweep_{method}_{sar}_{tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Print short summary
    print(f"[OK] {json_path}")
    if rows:
        print(f"[OK] {csv_path}")
    if best_lam is not None:
        best = lam_scores[best_lam]
        print(f"best_lambda={best_lam} score={best['score']:.6f} mean_AP={best['mean_AP']:.6f} mean_APS={best['mean_APS']:.6f}")
    else:
        print("best_lambda=None (no matching runs yet)")


if __name__ == "__main__":
    main()


