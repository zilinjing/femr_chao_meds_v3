#!/usr/bin/env bash
# Shared evaluation launcher. Do not run it directly: it is loaded by the launchers in
# script/<dataset>/..., which only pick the env file (paths) and the backbone:
#
#   source lib.sh <env name>     # loads env/<env name>.sh
#   ARCH=transformer             # or mamba (model launchers only)
#   main "$@"                    # model launchers  (transformer.sh, mamba.sh)
#   baseline_main "$@"           # baseline launcher (baseline.sh)
#   batches_main "$@"            # batch launcher (batches.sh, CUMC only)
#
# Model launchers:    bash <arch>.sh <model> <outcome|phenotype|regression|all> [task1,task2]
#                     bash <arch>.sh collect
# Baseline launcher:  bash baseline.sh <extract|outcome|phenotype|regression> [task1,task2]
# Batch launcher:     bash batches.sh <outcome|phenotype|regression|all> [task1,task2]
#
# Every setting that must be equal across models for a fair comparison lives here.
# See evaluation/README.md for the full guide.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
ENV_NAME="$1"
source "$SCRIPT_DIR/env/$ENV_NAME.sh"
export PYTHONUNBUFFERED=1  # stream python output into the logs as it happens

# ---- Shared settings (identical for every model) ----
TOKENS_PER_BATCH=65536
MIN_SUBJECTS_PER_BATCH=8       # at most 65536 / 8 = 8192 events per patient, as in pretraining
PHENOTYPE_OBS_WINDOW=730       # phenotype features only see the last 730 days before prediction
N_BOOT=1000                    # test-set resamples behind every "mean (std)"
SPLIT="$META_DIR/main_split.csv"

MODELS="ntp|motor|tpp|mtpp|icarefm|survivehr"
FAMILIES="outcome|phenotype|regression"
USAGE="usage: $0 <$MODELS> <$FAMILIES|all> [task1,task2,...]
       $0 collect
   or (baseline.sh): $0 <extract|outcome|phenotype|regression> [task1,task2,...]"

# start_log <name> <args...>: copy everything printed from now on to <RESULT_ROOT>/logs/.
start_log() {
    mkdir -p "$RESULT_ROOT/logs"
    local log="$RESULT_ROOT/logs/$1_$(date +%Y%m%d_%H%M%S).log"
    shift
    exec > >(tee -a "$log") 2>&1
    echo "# $(date) host=$(hostname) env=$ENV_NAME git=$(git -C "$EVAL_DIR" describe --always --dirty)"
    echo "# command: $(printf '%q ' "$0" "$@")"
    echo "# log: $log"
}

# tasks_in <cohort dir> [task1,task2]: the given tasks, or every task folder of the cohort.
tasks_in() {
    if [ -n "${2:-}" ]; then
        tr , '\n' <<< "$2"
    else
        find "$1" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    fi
}

# cohort_of <family>: the folder holding one sub-folder of labels per task.
cohort_of() {
    case "$1" in
        outcome) echo "$OUTCOME_COHORT" ;;
        phenotype) echo "$PHENOTYPE_BINARY_COHORT" ;;  # TTE labels: $PHENOTYPE_TTE_COHORT
        regression) echo "$REGRESSION_COHORT" ;;
    esac
}

# collect: rebuild the summary CSVs (<RESULT_ROOT>/<family>/<family>_<metric>.csv).
# The lock keeps two jobs that finish at the same time from writing a CSV together.
collect() {
    mkdir -p "$RESULT_ROOT"
    flock "$RESULT_ROOT/.collect.lock" \
        python -m femr.evaluation.common.collect --result_root "$RESULT_ROOT"
}

# batch_dir_of <family> <task> <cohort dir> <window>: folder of the saved batches of one task.
# Batches are shared by models with the same tokenizer: they do not depend on the weights.
batch_dir_of() {
    local batch_key
    batch_key=$({ echo "$MEDS_READER $ARCH $TOKENS_PER_BATCH $MIN_SUBJECTS_PER_BATCH $4"
                  cat "$MODEL_PATH/dictionary.msgpack" "$META_DIR/ontology.pkl" "$3"/*.parquet
                } | md5sum | cut -c1-16)
    echo "$HF_CACHE/eval_batches/$1_$2_$batch_key"
}

# eval_task <family> <task>: features from the frozen model (GPU), then the probe (CPU).
eval_task() {
    local family="$1" task="$2"
    local cohort_dir type=binary window=""
    cohort_dir="$(cohort_of "$family")/$task"
    [ "$family" = regression ] && type=regression
    [ "$family" = phenotype ] && window=$PHENOTYPE_OBS_WINDOW
    if [ ! -d "$cohort_dir" ]; then
        echo "no cohort folder $cohort_dir" >&2
        return 1
    fi

    # Features are computed once per model and task, then reused (delete the .pkl to recompute).
    local feat_root="$RESULT_ROOT/$family/_features"
    local features="$feat_root/$TAG/features/${task}_${TAG}${window:+_$window}.pkl"
    local labels="$feat_root/$TAG/labels/$task.parquet"
    local batch_dir
    batch_dir=$(batch_dir_of "$family" "$task" "$cohort_dir" "$window")
    if [ -f "$features" ]; then
        echo "reusing features $features"
    else
        python -m femr.evaluation.common.generate_features \
            --model_path "$MODEL_PATH" \
            --model_name "$TAG" \
            --meds_reader "$MEDS_READER" \
            --ontology_path "$META_DIR/ontology.pkl" \
            --cohort_dir "$cohort_dir/" \
            --task_type "$type" \
            --loss_type labeled_subjects \
            --tokens_per_batch "$TOKENS_PER_BATCH" \
            --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
            --num_proc "$NUM_PROC" \
            --device cuda:0 \
            --cache_path "$HF_CACHE" \
            --batch_dir "$batch_dir" \
            --output_root "$feat_root" \
            ${window:+--observation_window "$window"} || return 1
    fi

    local probe=(python -m "femr.evaluation.$family.probe"
        --task "$task"
        --features "$features"
        --labels "$labels"
        --split "$SPLIT"
        --out_dir "$RESULT_ROOT/$family/$task/$TAG"
        --n_boot "$N_BOOT")
    if [ "$family" = phenotype ]; then
        probe+=(--tte_dir "$PHENOTYPE_TTE_COHORT/$task" --truncation "$TRUNCATION")
    fi
    "${probe[@]}"
}

# run_family <family> [task1,task2]: every task of one family, then the summary CSVs.
run_family() {
    local family="$1" wanted="${2:-}" task
    local failed=()
    if [ -z "$wanted" ] && [ "$family" = regression ]; then
        wanted="$REGRESSION_TASKS"
    fi
    for task in $(tasks_in "$(cohort_of "$family")" "$wanted"); do
        echo "$(date '+%F %T') ===== $family / $task / $TAG ====="
        if eval_task "$family" "$task"; then
            echo "$(date '+%F %T') $family/$task: ok"
        else
            echo "$(date '+%F %T') $family/$task: FAILED" >&2
            failed+=("$family/$task")
        fi
    done
    collect
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED tasks: ${failed[*]} (see the log above)" >&2
        return 1
    fi
}

main() {
    local model="${1:-}" family="${2:-}" tasks="${3:-}"
    if [ "$model" = collect ]; then
        collect
        return
    fi
    if [[ ! "$model" =~ ^($MODELS)$ || ! "$family" =~ ^($FAMILIES|all)$ ]]; then
        echo "$USAGE" >&2
        exit 1
    fi

    # Defaults: the best checkpoint of pretraining, and the model name as the table row.
    MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/$model/$ARCH/best_model}"
    local default_tag="$model"
    [ "$ARCH" = transformer ] || default_tag="${model}_$ARCH"
    TAG="${TAG:-$default_tag}"
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "no trained model at $MODEL_PATH (train it first, or set MODEL_PATH=...)" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"

    start_log "${TAG}_${family}" "$@"
    echo "# model=$MODEL_PATH row=$TAG gpu=$CUDA_VISIBLE_DEVICES results=$RESULT_ROOT"

    local status=0 f
    if [ "$family" = all ]; then
        for f in outcome phenotype regression; do
            run_family "$f" "$tasks" || status=1
        done
    else
        run_family "$family" "$tasks" || status=1
    fi
    exit $status
}

# build_batches <family> [task1,task2]: save the batches of every task of one family, one task
# after the other. CPU only: no model is loaded, the tokenizer is read from $MODEL_PATH.
build_batches() {
    local family="$1" wanted="${2:-}" task cohort_dir batch_dir
    local type=binary window="" failed=()
    [ "$family" = regression ] && type=regression
    [ "$family" = phenotype ] && window=$PHENOTYPE_OBS_WINDOW
    if [ -z "$wanted" ] && [ "$family" = regression ]; then
        wanted="$REGRESSION_TASKS"
    fi
    for task in $(tasks_in "$(cohort_of "$family")" "$wanted"); do
        echo "$(date '+%F %T') ===== batches / $family / $task ====="
        cohort_dir="$(cohort_of "$family")/$task"
        if [ ! -d "$cohort_dir" ]; then
            echo "no cohort folder $cohort_dir" >&2
            failed+=("$family/$task")
            continue
        fi
        batch_dir=$(batch_dir_of "$family" "$task" "$cohort_dir" "$window")
        if [ -d "$batch_dir" ]; then
            echo "already built: $batch_dir"
            continue
        fi
        # The slow step of convert_dataset runs num_proc / 4 workers, hence 4 * BATCH_WORKERS.
        # $batch_dir.hf is a private HF cache: it would otherwise stay in $HF_CACHE as a second
        # copy of the batches, and it keeps the families running at once out of each other's way.
        rm -rf "$batch_dir.hf" "$batch_dir.tmp"
        if python -m femr.evaluation.common.generate_features \
            --batches_only \
            --model_path "$MODEL_PATH" \
            --model_name batches \
            --meds_reader "$MEDS_READER" \
            --ontology_path "$META_DIR/ontology.pkl" \
            --cohort_dir "$cohort_dir/" \
            --task_type "$type" \
            --tokens_per_batch "$TOKENS_PER_BATCH" \
            --min_subjects_per_batch "$MIN_SUBJECTS_PER_BATCH" \
            --num_proc $((4 * BATCH_WORKERS)) \
            --cache_path "$batch_dir.hf" \
            --batch_dir "$batch_dir" \
            --output_root "$batch_dir.hf" \
            ${window:+--observation_window "$window"}; then
            echo "$(date '+%F %T') batches $family/$task: ok"
        else
            echo "$(date '+%F %T') batches $family/$task: FAILED" >&2
            failed+=("$family/$task")
        fi
        rm -rf "$batch_dir.hf" "$batch_dir.tmp"
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED tasks: ${failed[*]} (see the log above)" >&2
        return 1
    fi
}

# batches_main <outcome|phenotype|regression|all> [task1,task2]: build the batches before any model
# is evaluated, so every model loads them. `all` starts the three families at the same time, each
# with BATCH_WORKERS workers and its own log in <RESULT_ROOT>/logs/.
batches_main() {
    local family="${1:-}" tasks="${2:-}"
    if [[ ! "$family" =~ ^($FAMILIES|all)$ ]]; then
        echo "usage: $0 <$FAMILIES|all> [task1,task2,...]" >&2
        exit 1
    fi
    if [ "$family" = all ]; then
        local f pid status=0 pids=()
        for f in outcome phenotype regression; do
            bash "$0" "$f" "$tasks" > /dev/null 2>&1 &
            pids+=($!)
        done
        echo "$(date '+%F %T') building 3 families, logs: $RESULT_ROOT/logs/batches_<family>_*.log"
        for pid in "${pids[@]}"; do
            wait "$pid" || status=1
        done
        echo "$(date '+%F %T') done, status $status"
        exit $status
    fi

    MODEL_PATH="$META_DIR/tokenizer"  # the shared tokenizer: no model has to exist yet
    if [ ! -f "$MODEL_PATH/dictionary.msgpack" ]; then
        echo "no tokenizer at $MODEL_PATH (run the training launcher's tasks step first)" >&2
        exit 1
    fi
    start_log "batches_$family" "$@"
    build_batches "$family" "$tasks"
}

# baseline_main <extract|outcome|phenotype|regression> [task1,task2]
#   extract     MEDS-Tab count/value features for every classification and TTE cohort (CPU)
#   outcome     XGBoost on the MEDS-Tab features
#   phenotype   XGBoost (binary) + DeepHit (TTE) on the MEDS-Tab features (DeepHit uses the GPU)
#   regression  most recent value of the lab before the prediction time
baseline_main() {
    local stage="${1:-}" tasks="${2:-}"
    if [[ ! "$stage" =~ ^(extract|$FAMILIES)$ ]]; then
        echo "$USAGE" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
    start_log "baseline_$stage" "$@"

    # run_all.py writes its own pickles/logs to $MEDS_TAB_DIR/results, and the rows of the
    # summary tables to $RESULT_ROOT (--eval_root). The last three options are set by the env file
    # of a dataset that needs them; without them run_all.py keeps its defaults.
    local run_all=(python run_all.py
        --data_path "$MEDS_TAB_DIR"
        --split_path "$SPLIT"
        --results_dir "$MEDS_TAB_DIR/results"
        --eval_root "$RESULT_ROOT"
        --truncation "$TRUNCATION"
        --n_jobs "$NUM_PROC"
        --device "${XGB_DEVICE:-cpu}"
        ${DEEPHIT_GRID:+--deephit_grid "$DEEPHIT_GRID"}
        ${DEEPHIT_WD:+--wd "$DEEPHIT_WD"}
        ${TTE_COL:+--tte_col "$TTE_COL"})

    case "$stage" in
        extract)
            MEDS_READER="$MEDS_READER" OUTCOME_COHORT="$OUTCOME_COHORT" \
            PHENOTYPE_BINARY_COHORT="$PHENOTYPE_BINARY_COHORT" PHENOTYPE_TTE_COHORT="$PHENOTYPE_TTE_COHORT" \
            MEDS_TAB_DIR="$MEDS_TAB_DIR" NUM_THREADS="$NUM_PROC" \
                bash "$EVAL_DIR/baseline/meds_tab/extract.sh" "${tasks:-all}"
            return ;;  # tasks = all | outcome | binary | tte
        outcome)
            (cd "$EVAL_DIR/baseline/meds_tab" &&
                "${run_all[@]}" --phenotypes --binary_only $(tasks_in "$OUTCOME_COHORT" "$tasks")) ;;
        phenotype)
            (cd "$EVAL_DIR/baseline/meds_tab" &&
                "${run_all[@]}" --binary_only --phenotypes $(tasks_in "$PHENOTYPE_TTE_COHORT" "$tasks")) ;;
        regression)
            python -m femr.evaluation.regression.baseline.baseline \
                --cohort_dir "$REGRESSION_COHORT" \
                --meds_path "$MEDS_READER" \
                --split_path "$SPLIT" \
                --dataset "$DATASET" \
                --output_dir "$RESULT_ROOT/regression" \
                --num_threads "$NUM_PROC" \
                --tasks "$(tasks_in "$REGRESSION_COHORT" "${tasks:-$REGRESSION_TASKS}" | paste -sd,)" ;;
    esac
    collect
}
