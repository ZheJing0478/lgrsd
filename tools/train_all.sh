#!/usr/bin/env bash
set -euo pipefail

# Ensure repo-root packages (e.g., ultralytics_ext/) are importable even when using the installed `yolo` entrypoint.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

run_in_env() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "${CONDA_ENV}" "$@"
  else
    "$@"
  fi
}

DATASET="all"        # hrsid|ssdd|all
SAR_INPUT="rgb3"     # rgb3 | gray1
IMGSZ="800"
EPOCHS="100"
MODEL_SCALE="n"      # n|s|m (we officially support n in this repo)
SEED="42"
DEVICE="0"
BATCH="-1"
WORKERS="8"
METHODS="baseline"   # comma-separated: baseline,attn,lgrsd,final
TAG=""               # optional suffix for run name (used for ablations), e.g. lam0p3

# AttnFPN variant (B5): p345 (default: P3/P4/P5), p3 (P3-only), or p45 (P4/P5-only)
ATTN_VARIANT="p345"

# LG-RSD knobs (used when method=lgrsd or final)
LGRSD_LAMBDA="0.5"
LGRSD_FEATURE_LEVEL="auto"  # auto | P3-only
LGRSD_TOPK="16"
LGRSD_CROP_SIZE="224"
LGRSD_ROI_SIZE="7"
LGRSD_EMBED_DIM="256"
LGRSD_CONTEXT_RATIO="1.3"
LGRSD_MIN_SIDE_PX="16"
LGRSD_MIN_ORIG_SIDE_PX="4"
LGRSD_MIN_AREA_RATIO="0.3"
LGRSD_SAMPLING_STRATEGY="stratified_default"  # stratified_default | area_topk
LGRSD_TEACHER_MOMENTUM="1.0"  # 1.0=frozen teacher, e.g. 0.99~0.999 to enable EMA teacher

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_all.sh [--dataset hrsid|ssdd|all] [--imgsz 800] [--epochs 100]
                          [--model_scale n] [--seed 42] [--device 0]
                          [--batch -1] [--workers 8]
                          [--methods baseline,attn,lgrsd,final]
                          [--sar_input rgb3|gray1]
                          [--attn_variant p345|p3|p45]
                          [--lgrsd_lambda 0.5] [--lgrsd_feature_level auto|P3-only]
                          [--lgrsd_crop_size 224] [--lgrsd_context_ratio 1.3]
                          [--lgrsd_sampling_strategy stratified_default|area_topk]
                          [--lgrsd_teacher_momentum 1.0]
                          [--tag your_suffix]

Notes:
  - Activate your Python environment before running this script, or set `CONDA_ENV=<env_name>`.
  - Prepared datasets are expected at:
      datasets/HRSID/hrsid.yaml
      datasets/HRSID/hrsid_gray1.yaml
      datasets/SSDD/ssdd.yaml
      datasets/SSDD/ssdd_gray1.yaml
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --sar_input) SAR_INPUT="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --model_scale) MODEL_SCALE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --attn_variant) ATTN_VARIANT="$2"; shift 2 ;;
    --lgrsd_lambda) LGRSD_LAMBDA="$2"; shift 2 ;;
    --lgrsd_feature_level) LGRSD_FEATURE_LEVEL="$2"; shift 2 ;;
    --lgrsd_topk) LGRSD_TOPK="$2"; shift 2 ;;
    --lgrsd_crop_size) LGRSD_CROP_SIZE="$2"; shift 2 ;;
    --lgrsd_roi_size) LGRSD_ROI_SIZE="$2"; shift 2 ;;
    --lgrsd_embed_dim) LGRSD_EMBED_DIM="$2"; shift 2 ;;
    --lgrsd_context_ratio) LGRSD_CONTEXT_RATIO="$2"; shift 2 ;;
    --lgrsd_min_side_px) LGRSD_MIN_SIDE_PX="$2"; shift 2 ;;
    --lgrsd_min_orig_side_px) LGRSD_MIN_ORIG_SIDE_PX="$2"; shift 2 ;;
    --lgrsd_min_area_ratio) LGRSD_MIN_AREA_RATIO="$2"; shift 2 ;;
    --lgrsd_sampling_strategy) LGRSD_SAMPLING_STRATEGY="$2"; shift 2 ;;
    --lgrsd_teacher_momentum) LGRSD_TEACHER_MOMENTUM="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

run_train_and_eval() {
  local ds="$1"       # hrsid|ssdd
  local method="$2"   # baseline|attn|lgrsd|final

  local data_yaml=""
  if [[ "$ds" == "hrsid" ]]; then
    data_yaml="datasets/HRSID/hrsid.yaml"
  elif [[ "$ds" == "ssdd" ]]; then
    data_yaml="datasets/SSDD/ssdd.yaml"
  else
    echo "Unknown dataset: $ds" >&2
    exit 1
  fi

  if [[ "${SAR_INPUT}" == "gray1" ]]; then
    if [[ "$ds" == "hrsid" ]]; then
      data_yaml="datasets/HRSID/hrsid_gray1.yaml"
    else
      data_yaml="datasets/SSDD/ssdd_gray1.yaml"
    fi
  fi

  local project="runs/${ds}"
  # A7: encode SAR input protocol into run name to prevent mixed-protocol comparisons.
  local run_name="${method}_yolov8${MODEL_SCALE}_${SAR_INPUT}"
  # B5: encode attention variant into run name (only affects attn/final).
  if [[ ("$method" == "attn" || "$method" == "final") && "${ATTN_VARIANT}" != "p345" ]]; then
    run_name="${run_name}_attn${ATTN_VARIANT}"
  fi
  if [[ -n "${TAG}" ]]; then
    run_name="${run_name}_${TAG}"
  fi

  local model_arg=""
  local pretrained_arg=()
  local extra_train_args=()
  if [[ "$method" == "baseline" ]]; then
    model_arg="yolov8${MODEL_SCALE}.pt"
  elif [[ "$method" == "attn" ]]; then
    if [[ "${ATTN_VARIANT}" == "p3" ]]; then
      model_arg="custom_models/yolov8${MODEL_SCALE}_attn_p3.yaml"
    elif [[ "${ATTN_VARIANT}" == "p45" ]]; then
      model_arg="custom_models/yolov8${MODEL_SCALE}_attn_p45.yaml"
    else
      model_arg="custom_models/yolov8${MODEL_SCALE}_attn.yaml"
    fi
    pretrained_arg=(pretrained="yolov8${MODEL_SCALE}.pt")
  elif [[ "$method" == "lgrsd" ]]; then
    model_arg="yolov8${MODEL_SCALE}.pt"
    extra_train_args+=(
      enable_lgrsd=True
      lgrsd_lambda="${LGRSD_LAMBDA}"
      lgrsd_topk_per_image="${LGRSD_TOPK}"
      lgrsd_crop_size="${LGRSD_CROP_SIZE}"
      lgrsd_roi_size="${LGRSD_ROI_SIZE}"
      lgrsd_feature_level="${LGRSD_FEATURE_LEVEL}"
      lgrsd_embed_dim="${LGRSD_EMBED_DIM}"
      lgrsd_context_ratio="${LGRSD_CONTEXT_RATIO}"
      lgrsd_min_side_px="${LGRSD_MIN_SIDE_PX}"
      lgrsd_min_orig_side_px="${LGRSD_MIN_ORIG_SIDE_PX}"
      lgrsd_min_area_ratio="${LGRSD_MIN_AREA_RATIO}"
      lgrsd_sampling_strategy="${LGRSD_SAMPLING_STRATEGY}"
      lgrsd_teacher_momentum="${LGRSD_TEACHER_MOMENTUM}"
      enable_region_contrastive=False
    )
  elif [[ "$method" == "final" ]]; then
    if [[ "${ATTN_VARIANT}" == "p3" ]]; then
      model_arg="custom_models/yolov8${MODEL_SCALE}_lgrsd_attn_p3.yaml"
    elif [[ "${ATTN_VARIANT}" == "p45" ]]; then
      model_arg="custom_models/yolov8${MODEL_SCALE}_lgrsd_attn_p45.yaml"
    else
      model_arg="custom_models/yolov8${MODEL_SCALE}_lgrsd_attn.yaml"
    fi
    pretrained_arg=(pretrained="yolov8${MODEL_SCALE}.pt")
    extra_train_args+=(
      enable_lgrsd=True
      lgrsd_lambda="${LGRSD_LAMBDA}"
      lgrsd_topk_per_image="${LGRSD_TOPK}"
      lgrsd_crop_size="${LGRSD_CROP_SIZE}"
      lgrsd_roi_size="${LGRSD_ROI_SIZE}"
      lgrsd_feature_level="${LGRSD_FEATURE_LEVEL}"
      lgrsd_embed_dim="${LGRSD_EMBED_DIM}"
      lgrsd_context_ratio="${LGRSD_CONTEXT_RATIO}"
      lgrsd_min_side_px="${LGRSD_MIN_SIDE_PX}"
      lgrsd_min_orig_side_px="${LGRSD_MIN_ORIG_SIDE_PX}"
      lgrsd_min_area_ratio="${LGRSD_MIN_AREA_RATIO}"
      lgrsd_sampling_strategy="${LGRSD_SAMPLING_STRATEGY}"
      lgrsd_teacher_momentum="${LGRSD_TEACHER_MOMENTUM}"
      enable_region_contrastive=False
    )
  else
    echo "Unknown method: $method" >&2
    exit 1
  fi

  echo "=== Train: dataset=${ds} method=${method} model=${model_arg} ==="
  run_in_env yolo detect train \
    cfg="tools/phase2_base_cfg.yaml" \
    model="${model_arg}" \
    "${pretrained_arg[@]}" \
    data="${data_yaml}" \
    imgsz="${IMGSZ}" \
    epochs="${EPOCHS}" \
    batch="${BATCH}" \
    device="${DEVICE}" \
    workers="${WORKERS}" \
    seed="${SEED}" \
    optimizer="SGD" \
    "${extra_train_args[@]}" \
    project="${project}" \
    name="${run_name}" \
    exist_ok=True

  # A8: persist the fixed base config into the run directory for full reproducibility.
  if [[ -d "${project}/${run_name}" ]]; then
    cp -f "tools/phase2_base_cfg.yaml" "${project}/${run_name}/phase2_base_cfg.yaml"
  fi

  local weights="${project}/${run_name}/weights/best.pt"
  if [[ ! -f "${weights}" ]]; then
    echo "WARN: best.pt not found, fallback to last.pt"
    weights="${project}/${run_name}/weights/last.pt"
  fi

  echo "=== Strict COCOeval: ${weights} ==="
  run_in_env python tools/run_strict_coco_eval.py \
    --model "${weights}" \
    --data "${data_yaml}" \
    --split val \
    --imgsz "${IMGSZ}" \
    --device "${DEVICE}" \
    --conf 0.001 \
    --iou 0.7 \
    --max_det 500 \
    --batch 1 \
    --half \
    --outdir "${project}" \
    --name "${run_name}_strict" \
    --exp_key "${ds}/${run_name}" \
    --summary_json "runs/summary.json"
}

IFS=',' read -r -a methods_arr <<< "${METHODS}"

if [[ "${DATASET}" == "all" ]]; then
  datasets=("hrsid" "ssdd")
else
  datasets=("${DATASET}")
fi

for ds in "${datasets[@]}"; do
  for method in "${methods_arr[@]}"; do
    run_train_and_eval "${ds}" "${method}"
  done
done

