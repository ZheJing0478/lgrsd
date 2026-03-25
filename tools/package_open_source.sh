#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_NAME="${PKG_NAME:-lgrsd-attnfpn-open-source}"
DIST_DIR="${REPO_ROOT}/dist"
OUT_DIR="${DIST_DIR}/${PKG_NAME}"
TMP_DIR="${OUT_DIR}.tmp"
ARCHIVE_PATH="${DIST_DIR}/${PKG_NAME}.tar.gz"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

copy_file() {
  local src_rel="$1"
  local dst_rel="$2"
  mkdir -p "$(dirname "${TMP_DIR}/${dst_rel}")"
  rsync -a "${REPO_ROOT}/${src_rel}" "${TMP_DIR}/${dst_rel}"
}

copy_dir() {
  local src_rel="$1"
  local dst_rel="$2"
  shift 2
  mkdir -p "${TMP_DIR}/${dst_rel}"
  rsync -a "$@" "${REPO_ROOT}/${src_rel}/" "${TMP_DIR}/${dst_rel}/"
}

sanitize_tree() {
  find "${TMP_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${TMP_DIR}" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name ".DS_Store" \) -delete
}

escape_for_rg() {
  printf '%s' "$1" | sed -e 's/[][(){}.^$*+?|\\-]/\\&/g'
}

privacy_scan() {
  local hits=""
  local slash="/"
  local local_user=""
  local repo_name=""
  local patterns=()
  local joined=""
  local escaped=""
  local first=1

  patterns+=("${slash}home${slash}")
  patterns+=("${slash}media${slash}")

  local_user="$(id -un 2>/dev/null || true)"
  repo_name="$(basename "${REPO_ROOT}")"
  if [[ -n "${local_user}" ]]; then
    patterns+=("${local_user}")
  fi
  if [[ -n "${repo_name}" ]]; then
    patterns+=("${repo_name}")
  fi

  for p in "${patterns[@]}"; do
    escaped="$(escape_for_rg "${p}")"
    if (( first )); then
      joined="${escaped}"
      first=0
    else
      joined="${joined}|${escaped}"
    fi
  done

  hits="$(rg -n --hidden --no-messages "${joined}" "${TMP_DIR}" || true)"
  if [[ -n "${hits}" ]]; then
    echo "Privacy scan failed. Local path or user information was found in the package:" >&2
    echo "${hits}" >&2
    exit 1
  fi
}

build_package() {
  rm -rf "${TMP_DIR}" "${OUT_DIR}" "${ARCHIVE_PATH}"
  mkdir -p "${DIST_DIR}" "${TMP_DIR}"

  copy_file ".gitignore" ".gitignore"
  copy_file "README.md" "README.md"
  copy_file "requirements.txt" "requirements.txt"
  copy_file "RUN_ALL.sh" "RUN_ALL.sh"
  copy_file "yolov8_custom_diff.patch" "yolov8_custom_diff.patch"

  copy_dir "docs" "docs"
  copy_dir "custom_models" "custom_models" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude "*.pt" \
    --exclude "*.pth" \
    --exclude "*.onnx" \
    --exclude "*.engine"
  copy_dir "tools" "tools" \
    --exclude "__pycache__" \
    --exclude "*.pyc"
  copy_dir "ultralytics_ext" "ultralytics_ext" \
    --exclude "__pycache__" \
    --exclude "*.pyc"

  copy_file "method/yolov8/LICENSE" "LICENSE"
  copy_file "method/yolov8/LICENSE" "method/yolov8/LICENSE"
  copy_file "method/yolov8/README.md" "method/yolov8/README.md"
  copy_file "method/yolov8/pyproject.toml" "method/yolov8/pyproject.toml"
  copy_dir "method/yolov8/ultralytics" "method/yolov8/ultralytics" \
    --exclude "__pycache__" \
    --exclude "*.pyc"

  sanitize_tree
  privacy_scan

  mv "${TMP_DIR}" "${OUT_DIR}"
  tar -czf "${ARCHIVE_PATH}" -C "${DIST_DIR}" "${PKG_NAME}"

  echo "Created package directory: ${OUT_DIR}"
  echo "Created archive: ${ARCHIVE_PATH}"
}

need_cmd rsync
need_cmd tar
need_cmd rg
build_package
