#!/usr/bin/env bash
# Usage: ./scripts/submit-rokoko-bench.sh [--modes=LIST] [--sizes=LIST] [--reps=N] [--features=LIST] [sbatch options...]
# Preview the queue start estimate, then confirm. Lists are comma separated.
# Example: ./scripts/submit-rokoko-bench.sh --modes=parallel --sizes=p-30 --features=events --reps=5

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/submit-rokoko-bench.sh [--modes=LIST] [--sizes=LIST] [--reps=N] [--features=LIST] [sbatch options...]
Preview the queue start estimate, then confirm. Lists are comma separated.
Modes: serial,parallel  Sizes: p-26,p-28,p-30  Default reps: 3
Features: additional Cargo features; parallel is selected with --modes.
Example: ./scripts/submit-rokoko-bench.sh --modes=parallel --sizes=p-30 --features=events --reps=5
EOF
}

die() { echo "$*" >&2; exit 2; }

modes='' sizes='' reps='' features=''
options=()
while (($#)); do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --modes|--sizes|--reps|--features)
      key=$1
      shift
      (($#)) || die "$key needs a value"
      value=$1 ;;
    --modes=*|--sizes=*|--reps=*|--features=*)
      key=${1%%=*}
      value=${1#*=} ;;
    *) options+=("$1"); shift; continue ;;
  esac

  case $key in
    --modes)
      [[ $value =~ ^(serial|parallel)(,(serial|parallel))*$ ]] || die "Invalid modes: $value"
      modes=${value//,/ } ;;
    --sizes)
      [[ $value =~ ^p-?(26|28|30)(,p-?(26|28|30))*$ ]] || die "Invalid sizes: $value"
      sizes=${value//,/ } ;;
    --reps)
      [[ $value =~ ^[1-9][0-9]*$ ]] || die "Reps must be a positive integer"
      reps=$value ;;
    --features)
      [[ $value =~ ^[a-zA-Z0-9_-]+(,[a-zA-Z0-9_-]+)*$ ]] || die "Invalid features: $value"
      [[ ,$value, != *,parallel,* ]] || die "Select parallel with --modes, not --features"
      features=$value ;;
  esac
  shift
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
script_args=("$modes" "$sizes" "$reps" "$features")

echo "Estimated start time:"
sbatch --test-only "${options[@]}" scripts/rokoko-bench.slurm "${script_args[@]}"

read -r -p "Submit job? [y/N] " answer || exit 0
case "$answer" in
  y|Y|yes|YES) sbatch "${options[@]}" scripts/rokoko-bench.slurm "${script_args[@]}" ;;
  *) echo "Cancelled." ;;
esac
