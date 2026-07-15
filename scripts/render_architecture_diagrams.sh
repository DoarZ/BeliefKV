#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIGURE_DIR="${ROOT_DIR}/docs/figures"

if ! command -v dot >/dev/null 2>&1; then
  echo "Graphviz 'dot' is required to render architecture diagrams." >&2
  exit 1
fi

for name in beliefkv_architecture_status beliefkv_phase_status; do
  dot -Tsvg "${FIGURE_DIR}/${name}.dot" -o "${FIGURE_DIR}/${name}.svg"
  dot -Tpng -Gdpi=150 \
    "${FIGURE_DIR}/${name}.dot" -o "${FIGURE_DIR}/${name}.png"
done

echo "Rendered BeliefKV architecture diagrams in ${FIGURE_DIR}"
