#!/usr/bin/env bash
# Launch the four training runs: three GPU-bound DeepHit tasks in parallel plus the
# CPU-bound XGBoost sweep. See README for the rationale (3 idle GPUs = free 3x).
#
#   bash launch.sh            # start everything
#   bash launch.sh status     # progress of each run
#
# NOTE: do not `pkill -f run_all.py` from an interactive shell whose own command line
# contains that string -- pkill matches it and kills the caller. Use `bash launch.sh stop`.

set -uo pipefail

PY="${PY:-/data/zilinjing/ethos_fm/envs/tte/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R=/data/zilinjing/ethos_fm/storage/ora/result/meds_tab
GPUS=(${GPUS:-0 1 4})
TASKS=(ami stroke masld)
# DeepHit: n_jobs only parallelises the closing 10k C-index bootstrap (training is on
# GPU), so 8 costs ~3 extra minutes per task and frees 24 cores for other work.
DH_NJOBS="${DH_NJOBS:-8}"
# full = 48 configs (layers [], [64], [128] x lr x batch x dropout x wd)
DH_GRID="${DH_GRID:-full}"
# XGBoost: runs on GPU (~9x faster than 32 CPU threads, equivalent AUROC), so n_jobs
# matters little here; it only covers the data marshalling.
XGB_NJOBS="${XGB_NJOBS:-8}"
XGB_GPU="${XGB_GPU:-5}"

mkdir -p "$R"

case "${1:-start}" in
  status)
    for f in "$R"/tte_*.log "$R"/binary_all.log; do
      [ -e "$f" ] || continue
      echo "--- $(basename "$f")"
      grep -viE "warn|^[[:space:]]*$|it/s\]" "$f" | tail -4
    done
    echo "--- running: $(pgrep -fc 'run_all[.]py' || echo 0)"
    ;;
  stop)
    pkill -f 'run_all[.]py' && echo stopped || echo "nothing running"
    ;;
  binary)
    echo "launching XGBoost sweep on GPU $XGB_GPU"
    CUDA_VISIBLE_DEVICES="$XGB_GPU" setsid nohup "$PY" -u "$HERE/run_all.py" \
      --skip survival --n_jobs "$XGB_NJOBS" --device cuda > "$R/binary_all.log" 2>&1 &
    sleep 3; echo "running: $(pgrep -fc 'run_all[.]py')"
    ;;
  tte)
    for i in "${!TASKS[@]}"; do
      t="${TASKS[$i]}"; g="${GPUS[$i]}"
      echo "launching $t on GPU $g"
      CUDA_VISIBLE_DEVICES="$g" setsid nohup "$PY" -u "$HERE/run_all.py" \
        --phenotypes "$t" --skip binary --n_jobs "$DH_NJOBS" --deephit_grid "$DH_GRID" > "$R/tte_$t.log" 2>&1 &
      sleep 1
    done
    sleep 3; echo "running: $(pgrep -fc 'run_all[.]py')"
    ;;
  start)
    for i in "${!TASKS[@]}"; do
      t="${TASKS[$i]}"; g="${GPUS[$i]}"
      echo "launching $t on GPU $g"
      CUDA_VISIBLE_DEVICES="$g" setsid nohup "$PY" -u "$HERE/run_all.py" \
        --phenotypes "$t" --skip binary --n_jobs "$DH_NJOBS" --deephit_grid "$DH_GRID" > "$R/tte_$t.log" 2>&1 &
      sleep 1
    done
    echo "launching XGBoost sweep on GPU $XGB_GPU"
    CUDA_VISIBLE_DEVICES="$XGB_GPU" setsid nohup "$PY" -u "$HERE/run_all.py" \
      --skip survival --n_jobs "$XGB_NJOBS" --device cuda > "$R/binary_all.log" 2>&1 &
    sleep 3
    echo "running: $(pgrep -fc 'run_all[.]py')"
    ;;
  *)
    echo "usage: bash launch.sh [start|binary|tte|status|stop]" >&2; exit 2 ;;
esac
