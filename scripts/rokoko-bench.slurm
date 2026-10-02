#!/bin/bash -l
#SBATCH --job-name=rokoko-bench
#SBATCH --partition=gpu-h200-71g-ia
#SBATCH --constraint=emerald
#SBATCH --gres=gpu:1
#SBATCH --mem-bind=local
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --output=rokoko-bench-%j.out
#SBATCH --error=rokoko-bench-%j.out

set -u

MODES=${1:-"serial parallel"}
SIZES=${2:-"p-26 p-28 p-30"}
REPS=${3:-3}
FEATURES=${4:-}
TOOLCHAIN="+nightly"

cd "$SLURM_SUBMIT_DIR" || exit 1

OUT="bench-${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
TIMES="$OUT/times.txt"
: > "$TIMES"

echo "host $(hostname)"
date
echo "logs in $OUT/"
lscpu | grep -E "Model name|^CPU\(s\):|L3 cache"
free -g | head -2
echo "cpus visible: $(nproc)"
export RAYON_NUM_THREADS=$(nproc)
echo "RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
git log --oneline -1

module load triton-dev/2025.1-gcc gcc/14.2.0 2>/dev/null

if ! command -v gcc > /dev/null
then
  echo "no gcc on this node; try: module avail gcc"
  exit 1
fi
export CC=gcc

BIN="$PWD/$OUT/bin"
mkdir -p "$BIN"
ln -sf "$(command -v gcc)" "$BIN/cc"
export PATH="$BIN:$PATH"
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc
echo "gcc: $(command -v gcc)"
echo "cc:  $(command -v cc)"

if ! command -v cargo > /dev/null
then
  . "$HOME/.cargo/env"
fi
cargo $TOOLCHAIN --version || exit 1

for m in $MODES
do
  for s in $SIZES
  do
    case $s in
      p26|p-26) param=p-26 ;;
      p28|p-28) param=p-28 ;;
      p30|p-30) param=p-30 ;;
      *) echo "Unknown parameter set: $s" >&2; exit 2 ;;
    esac

    f=$FEATURES
    feature_args=()
    if [ "$m" = parallel ]
    then
      f="${f:+$f,}parallel"
    fi
    if [ -n "$f" ]
    then
      feature_args=(--features "$f")
    fi

    tag="$m-${param/p-/p}"
    blog="$OUT/build-$tag.log"
    echo
    echo "======== $tag   features=${f:-default} param=$param"
    cargo $TOOLCHAIN build --release "${feature_args[@]}" > "$blog" 2>&1
    if [ $? -ne 0 ]
    then
      echo "BUILD FAILED, see $blog"
      tail -n 5 "$blog"
      continue
    fi
    tail -n 1 "$blog"

    r=1
    while [ "$r" -le "$REPS" ]
    do
      log="$OUT/$tag-rep$r.log"
      echo "-- $tag rep $r/$REPS  $(date +%H:%M:%S)  -> $log"
      fmt="==> $tag rep $r: %e s wall, %U s user, %M KB peak"
      /usr/bin/time -a -o "$TIMES" -f "$fmt" \
        cargo $TOOLCHAIN run --release "${feature_args[@]}" -- "$param" \
        > "$log" 2>&1
      tail -n 1 "$TIMES"
      r=$((r + 1))
    done
  done
done

echo
date
echo "==== summary ===="
cat "$TIMES"
echo "per-run output in $OUT/"
