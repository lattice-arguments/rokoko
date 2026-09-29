#!/usr/bin/env bash
# Usage: ./scripts/submit-rokoko-bench.sh [sbatch options...]
# Preview Slurm's estimated start time, override sbatch options, then confirm submission.
# The main slurm file is defined in rokoko-bench.slum.

set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  cat <<'EOF'
Usage: ./scripts/submit-rokoko-bench.sh [sbatch options...]
Preview Slurm's estimated start time, override sbatch options, then confirm submission.
Example: ./scripts/submit-rokoko-bench.sh --exclusive --mem=80G
EOF
  exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."
options=("$@")

echo "Estimated start time:"
sbatch --test-only "${options[@]}" scripts/rokoko-bench.slurm

read -r -p "Submit job? [y/N] " answer || exit 0
case "$answer" in
  y|Y|yes|YES) sbatch "${options[@]}" scripts/rokoko-bench.slurm ;;
  *) echo "Cancelled." ;;
esac
