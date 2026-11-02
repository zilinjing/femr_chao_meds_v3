#!/usr/bin/env bash
# Shared launcher for prepare.py and pretrain.py. Sourced by every <dataset>/<cluster>/<arch>.sh:
#
#   source lib.sh <env name>     # loads env/<env name>.sh (paths, time bins, GPUs)
#   ARCH=transformer             # or mamba
#   ARCH_ARGS=(...)              # backbone hyperparameters for pretrain.py
#   main "$@"                    # tasks [extra prepare.py args]
#                                # | <model> [prepare|pretrain|all|train] [extra args]
#
# Typical use:
#   bash <arch>.sh tasks          # shared tokenizer + all six task pickles (motor first, then the rest in parallel)
#   bash <arch>.sh <model> train  # per model, once its task pickle exists: build batches, then pretrain
#   (per job: GPUS=0,1 NUM_THREADS=16 N_EPOCHS=20 RUN_NAME=...; TIME_BINS/VALUE_BINS apply when a task is fit)
#
# Every setting that must be equal across models for a fair comparison lives here.
# Every run writes a log (with date, host, git commit and the exact command) to <model dir>/logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="$1"
source "$SCRIPT_DIR/env/$ENV_NAME.sh"
cd "$SCRIPT_DIR/.."  # prepare.py and pretrain.py live here
export PYTHONUNBUFFERED=1  # stream python output into the logs as it happens

# ---- Shared settings (identical for every model) ----
VOCAB_SIZE=16384
TOKENS_PER_BATCH=16384
MIN_SUBJECTS_PER_BATCH=2      # transformer windows: 16384 / 2 = 8192 events per subject
MAMBA_MAX_TOKENS=8192         # mamba windows: 8192 events per subject
GLOBAL_TOKENS=65536           # tokens per optimizer step = TOKENS_PER_BATCH x number of GPUs x grad accumulation
VALUE_BINS=${VALUE_BINS:-5}            # MTPP value bins; fixed when the task is fit ("tasks")
VALUE_COUNT_THRESHOLD=1000

MODELS="ntp|motor|tpp|mtpp|icarefm|survivehr"
USAGE="usage: $0 tasks [extra prepare.py args]
       $0 <$MODELS> [prepare|pretrain|all|train] [extra prepare.py args (prepare) or pretrain.py args (pretrain, all, train)]"

# run <log name> <command...>: record the context and exact command, run it, tee the output.
run() {
    local log="$DATA_DIR/logs/$1_${ARCH}_$(date +%Y%m%d_%H%M%S).log"
    shift
    mkdir -p "$DATA_DIR/logs"
    {
        echo "# $(date) host=$(hostname) env=$ENV_NAME git=$(git describe --always --dirty)"
        echo "# python=$(command -v python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} WANDB_PROJECT=${WANDB_PROJECT:-}"
        echo "# command: $(printf '%q ' "$@")"
        local status=0
        "$@" || status=$?
        echo "# finished $(date) exit=$status"
        exit "$status"
    } 2>&1 | tee "$log"
}

prepare() {
    local model="$1"
    run "prepare_$model" python prepare.py \
        --pretraining_data "$DATA_DIR" \
        --ontology_path "$META_DIR/ontology.pkl" \
        --tokenizer_path "$META_DIR/tokenizer" \
        --meds_reader "$MEDS_READER" \
        --hf_cache_path "$HF_CACHE" \
        --num_threads "$NUM_THREADS" \
        --model "$ARCH" \
        --loss_type "$model" \
        --vocab_size "$VOCAB_SIZE" \
        --tokens_per_batch "$TOKENS_PER_BATCH" \
        --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
        --mamba_per_sample_max_tokens "$MAMBA_MAX_TOKENS" \
        --time_bins "$TIME_BINS" \
        --value_bins "$VALUE_BINS" \
        --value_count_threshold "$VALUE_COUNT_THRESHOLD" \
        --banned_properties "${BANNED_PROPERTIES[@]}" \
        "${@:2}"
}

pretrain() {
    local model="$1"
    shift

    local loss="$model"
    if [ "$model" = mtpp ]; then
        loss=mtpp_shared  # MTPP is prepared as "mtpp" and pretrained with its shared head
    fi

    local model_args=()
    if [ "$model" = survivehr ]; then
        model_args=(--survivehr_time_scale_days 1.0)
    fi

    # GPUS lists the GPU ids; their number sets the processes and the gradient accumulation,
    # so every run sees GLOBAL_TOKENS tokens per optimizer step.
    local gpu_ids
    IFS=, read -ra gpu_ids <<< "$GPUS"
    local num_gpus=${#gpu_ids[@]}
    case "$num_gpus" in
        1|2|4) ;;
        *) echo "GPUS must list 1, 2 or 4 GPU ids (got '$GPUS')" >&2; exit 1 ;;
    esac
    export CUDA_VISIBLE_DEVICES="$GPUS"
    local grad_accumulation=$(( GLOBAL_TOKENS / (TOKENS_PER_BATCH * num_gpus) ))

    local launcher=(python)
    if [ "$num_gpus" -gt 1 ]; then
        # A free port per job, so several multi-GPU jobs can run on one node.
        local port
        port=$(python -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')
        launcher=(accelerate launch --num_processes "$num_gpus" --main_process_port "$port")
    fi

    run "pretrain_$model" "${launcher[@]}" pretrain.py \
        --pretraining_data "$DATA_DIR" \
        --meds_reader "$MEDS_READER" \
        --output_dir "$DATA_DIR/$ARCH" \
        --model "$ARCH" \
        --loss_type "$loss" \
        --hf_run_name "${RUN_NAME:-${model}_${ARCH}}" \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps "$grad_accumulation" \
        "${ARCH_ARGS[@]}" \
        "${model_args[@]}" \
        "$@"
}

# use_model_dir <model>: set DATA_DIR and link the shared ontology into it
# (pretrain.py reads the ontology from the model directory).
use_model_dir() {
    DATA_DIR="$OUT_ROOT/$1"
    mkdir -p "$DATA_DIR"
    if [ ! -e "$DATA_DIR/ontology.pkl" ] && [ ! -L "$DATA_DIR/ontology.pkl" ]; then
        ln -s "$META_DIR/ontology.pkl" "$DATA_DIR/ontology.pkl"
    fi
}

# tasks [extra prepare.py args]: shared tokenizer and every model's task pickle, no batches.
# motor runs first and trains the shared tokenizer; once motor has its copy, the other models
# start in parallel and reuse it (started earlier, each would train its own tokenizer).
prepare_tasks() {
    local log_dir="$OUT_ROOT/tasks_logs"
    mkdir -p "$log_dir"
    declare -A pids
    local model
    for model in motor ntp tpp mtpp icarefm survivehr; do
        if [ "$model" != motor ]; then
            until [ -f "$OUT_ROOT/motor/tokenizer/dictionary.msgpack" ]; do
                if ! kill -0 "${pids[motor]}" 2>/dev/null; then
                    echo "motor exited before the shared tokenizer was ready; see $log_dir/motor.out" >&2
                    exit 1
                fi
                sleep 30
            done
        fi
        echo "$(date '+%F %T') start $model (console: $log_dir/$model.out)"
        ( use_model_dir "$model"; prepare "$model" --tasks_only "$@" ) > "$log_dir/$model.out" 2>&1 &
        pids[$model]=$!
    done

    local failed=0
    for model in motor ntp tpp mtpp icarefm survivehr; do
        if wait "${pids[$model]}"; then
            echo "$(date '+%F %T') $model: ok"
        else
            echo "$(date '+%F %T') $model: FAILED, see $log_dir/$model.out" >&2
            failed=1
        fi
    done
    return $failed
}

main() {
    local model="${1:-}"
    if [ "$model" = tasks ]; then
        shift
        prepare_tasks "$@"
        return
    fi
    local stage="${2:-all}"
    if [[ ! "$model" =~ ^($MODELS)$ ]]; then
        echo "$USAGE" >&2
        exit 1
    fi
    shift $(( $# >= 2 ? 2 : $# ))
    use_model_dir "$model"

    case "$stage" in
        prepare) prepare "$model" "$@" ;;
        pretrain) pretrain "$model" "$@" ;;
        all) prepare "$model"; pretrain "$model" "$@" ;;
        train)
            # Batches + pretraining only: never fit the tokenizer or the task here (that is "tasks").
            if [ ! -f "$META_DIR/tokenizer/dictionary.msgpack" ] || [ ! -f "$DATA_DIR/${model}_task.pkl" ]; then
                echo "$model: shared tokenizer or $DATA_DIR/${model}_task.pkl missing; run '$0 tasks' first" >&2
                exit 1
            fi
            prepare "$model"
            pretrain "$model" "$@" ;;
        *) echo "$USAGE" >&2; exit 1 ;;
    esac
}
