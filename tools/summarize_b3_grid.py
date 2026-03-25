from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


CTX_CROP_RE = re.compile(r"_ctx(?P<ctx_i>\d+)p(?P<ctx_f>\d+)_crop(?P<crop>\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize B3 (context_ratio x crop_size) grid runs from runs/summary.json.")
    p.add_argument("--summary", default="runs/summary.json")
    p.add_argument("--outdir", default="runs/debug")
    p.add_argument("--datasets", default="hrsid", help="Comma-separated datasets, e.g. 'hrsid' or 'hrsid,ssdd'")
    p.add_argument("--method", default="final", choices=["final", "lgrsd"])
    p.add_argument("--model_scale", default="n")
    p.add_argument("--sar_input", default="rgb3", choices=["rgb3", "gray1"])
    p.add_argument("--filter_tag", default="b3_1024", help="Only consider exp_keys containing this substring.")
    p.add_argument(
        "--score",
        default="mean_aps_minus_std",
        choices=["mean_aps", "mean_ap", "mean_aps_minus_std", "mean_ap_plus_aps_minus_std"],
        help="How to pick best (ctx, crop) when multiple datasets are present.",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict json: {path}")
    return data


def _parse_ctx_crop(exp_key: str) -> tuple[float, int] | None:
    m = CTX_CROP_RE.search(exp_key)
    if not m:
        return None
    ctx = float(f"{m.group('ctx_i')}.{m.group('ctx_f')}")
    crop = int(m.group("crop"))
    return ctx, crop


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

    # Collect: (ctx, crop) -> dataset -> metrics
    data: dict[tuple[float, int], dict[str, dict[str, float]]] = {}
    for ds in datasets:
        prefix = f"{ds}/{method}_yolov8{scale}_{sar}"
        keys = [k for k in summary.keys() if isinstance(k, str) and k.startswith(prefix) and "_ctx" in k and "_crop" in k]
        if tag:
            keys = [k for k in keys if tag in k]
        for k in keys:
            cc = _parse_ctx_crop(k)
            if cc is None:
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
            data.setdefault(cc, {})[ds] = {"AP": ap, "APS": aps, "AP50": ap50, "AP75": ap75, "APM": apm, "APL": apl}

    # Flatten rows
    rows: list[dict[str, Any]] = []
    for (ctx, crop) in sorted(data.keys(), key=lambda x: (x[0], x[1])):
        for ds in datasets:
            m = data[(ctx, crop)].get(ds)
            if not m:
                continue
            rows.append({"dataset": ds, "context_ratio": ctx, "crop_size": crop, **m})

    # Aggregate & pick best
    cfg_scores: dict[str, dict[str, float]] = {}
    cfg_key_to_cc: dict[str, tuple[float, int]] = {}
    for (ctx, crop) in data.keys():
        aps_vals = [data[(ctx, crop)].get(ds, {}).get("APS", float("nan")) for ds in datasets]
        ap_vals = [data[(ctx, crop)].get(ds, {}).get("AP", float("nan")) for ds in datasets]
        m_aps = _mean([float(x) for x in aps_vals])
        m_ap = _mean([float(x) for x in ap_vals])
        s_aps = _std([float(x) for x in aps_vals])
        s_ap = _std([float(x) for x in ap_vals])

        if args.score == "mean_aps":
            score = m_aps
        elif args.score == "mean_ap":
            score = m_ap
        elif args.score == "mean_ap_plus_aps_minus_std":
            score = (m_ap + m_aps) - 0.5 * (s_ap + s_aps)
        else:
            # APS-first with a light stability penalty across datasets
            score = m_aps - 0.5 * s_aps + 0.25 * m_ap - 0.25 * s_ap

        key = f"ctx{ctx:.1f}_crop{crop}"
        cfg_key_to_cc[key] = (ctx, crop)
        cfg_scores[key] = {"mean_AP": m_ap, "mean_APS": m_aps, "std_AP": s_ap, "std_APS": s_aps, "score": score}

    best_key = max(cfg_scores.keys(), key=lambda k: cfg_scores[k]["score"]) if cfg_scores else None
    best = cfg_key_to_cc.get(best_key) if best_key else None

    out_json = {
        "meta": {
            "datasets": datasets,
            "method": method,
            "model_scale": scale,
            "sar_input": sar,
            "filter_tag": tag,
            "score": args.score,
        },
        "per_config": cfg_scores,
        "best_config_key": best_key,
        "best_context_ratio": (best[0] if best else None),
        "best_crop_size": (best[1] if best else None),
        "rows": rows,
    }
    json_path = outdir / f"b3_grid_{method}_{sar}_{tag}.json"
    json_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = outdir / f"b3_grid_{method}_{sar}_{tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"[OK] {json_path}")
    if rows:
        print(f"[OK] {csv_path}")
    if best_key:
        b = cfg_scores[best_key]
        print(
            f"best={best_key} score={b['score']:.6f} mean_AP={b['mean_AP']:.6f} mean_APS={b['mean_APS']:.6f} "
            f"(ctx={out_json['best_context_ratio']}, crop={out_json['best_crop_size']})"
        )
    else:
        print("best=None (no matching runs yet)")


if __name__ == "__main__":
    main()


