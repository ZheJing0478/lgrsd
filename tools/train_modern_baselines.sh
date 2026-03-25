#!/usr/bin/env bash
set -euo pipefail

# Ensure repo-root packages (e.g., ultralytics_ext/ and method/yolov8) are importable.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/method/yolov8:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

run_in_env() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "${CONDA_ENV}" "$@"
  else
    "$@"
  fi
}

DATASET="all"           # hrsid|ssdd|all
SAR_INPUT="gray1"       # gray1 (Protocol II)
IMGSZ="800"
EPOCHS="12"
SEED="42"
DEVICE="0"
BATCH="8"
WORKERS="8"
MODELS="yolov8n,rtdetr-r18"  # comma-separated
TAG=""

CFG="tools/phase2_modern_baselines_cfg.yaml"

usage() {
  cat <<'EOF'
Usage:
  bash tools/train_modern_baselines.sh [--dataset hrsid|ssdd|all]
                                       [--sar_input gray1]
                                       [--imgsz 800] [--epochs 12]
                                       [--seed 42] [--device 0]
                                       [--batch 8] [--workers 8]
                                       [--models yolov8n,rtdetr-r18]
                                       [--tag your_suffix]
Notes:
  - Activate your Python environment before running this script, or set `CONDA_ENV=<env_name>`.
  - Uses the local ultralytics fork via PYTHONPATH.
  - Protocol II single-channel input is enforced via *gray1* dataset yaml.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --sar_input) SAR_INPUT="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

run_train_and_eval() {
  local ds="$1"      # hrsid|ssdd
  local model="$2"   # yolov8n|rtdetr-r18

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

  local model_arg=""
  local pretrained_arg=()
  if [[ "$model" == "yolov8n" ]]; then
    model_arg="yolov8n.pt"
  elif [[ "$model" == "rtdetr-r18" ]]; then
    model_arg="custom_models/rtdetr-r18.yaml"
    if [[ -f "rtdetr-r18.pt" ]]; then
      pretrained_arg=(pretrained="rtdetr-r18.pt")
    fi
  else
    echo "Unknown model: $model" >&2
    exit 1
  fi

  local project="runs/${ds}"
  local run_name="modern_${model}_${SAR_INPUT}"
  if [[ -n "${TAG}" ]]; then
    run_name="${run_name}_${TAG}"
  fi

  echo "=== Train: dataset=${ds} model=${model_arg} ==="
  run_in_env yolo detect train \
    cfg="${CFG}" \
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
    project="${project}" \
    name="${run_name}" \
    exist_ok=True

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

IFS=',' read -r -a models_arr <<< "${MODELS}"

if [[ "${DATASET}" == "all" ]]; then
  datasets=("hrsid" "ssdd")
else
  datasets=("${DATASET}")
fi

for ds in "${datasets[@]}"; do
  for model in "${models_arr[@]}"; do
    run_train_and_eval "${ds}" "${model}"
  done
done
