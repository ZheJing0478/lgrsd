#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper required by the reproduction checklist.
# Delegates to tools/train_all.sh with the same arguments.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

bash tools/train_all.sh "$@"


