#!/bin/bash -l
#SBATCH --job-name=rokoko-bench
#SBATCH --partition=gpu-h100-80g-short
#SBATCH --constraint=saphr
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --output=rokoko-bench-%j.out

set -u

MODES="serial parallel"
SIZES="p26 p28 p30"
REPS=3
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
NPROC=$(nproc)
echo "cpus in this allocation: $NPROC"
export RAYON_NUM_THREADS=$NPROC
git log --oneline -1

module load triton-dev/2025.1-gcc gcc/14.2.0 2>/dev/null
export CC=${CC:-gcc}
if ! command -v cargo > /dev/null
then
    . "$HOME/.cargo/env"
fi
cargo $TOOLCHAIN --version || exit 1

for m in $MODES
do
  for s in $SIZES
  do
    f=""
    case $s in
      p26) f="${f:+$f,}p-26" ;;
      p30) f="${f:+$f,}p-30" ;;
    esac
    if [ "$m" = parallel ]
    then
      f="${f:+$f,}parallel"
    fi

    args=""
    if [ -n "$f" ]
    then
      args="--features=$f"
    fi

    tag="$m-$s"
    blog="$OUT/build-$tag.log"
    echo
    echo "======== $tag   $args"
    cargo $TOOLCHAIN build --release $args > "$blog" 2>&1
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
        cargo $TOOLCHAIN run --release $args \
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
