> [!CAUTION]
> This tutorial is still a work in progress, use at your own risk!

FEMR Columbia MEDS Tutorial
======================

This tutorial illustrates how to use 

Step 1. Converting into meds_reader
------------------------

The FEMR library uses the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. This requires a second preprocessing step

```bash
pip install meds_reader==0.1.3
```

```bash
export OMOP_MEDS = ""
export OMOP_MEDS_READER = ""
export PRETRAINING_DATA = ""
export ATHENA_DATA = ""

```

```bash
meds_reader_convert $OMOP_MEDS $OMOP_MEDS_READER --num_threads 16
```

Step 2. Downloading Athena
-------------------------

FEMR uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.

You can create an account for free.

Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

Step 3. Installing FEMR
------------------------

Finally, you need to install FEMR.

This code currently uses a branch of FEMR, mimic_tutorial, in order to function. However, soon these changes will be merged upstream into femr 2.4

```bash
git clone https://github.com/reAIM-Lab/EHR_TTE.git
cd EHR_TTE
pip install -e .
pip install xformers
```

Make sure to also install the correct gpu enabled version of PyTorch


Step 4. Preparing For Pretraining
------------------------

## Overview

The pretraining pipeline consists of two main scripts:
1. **prepare.py** - Prepares data, trains tokenizer, creates tasks, and generates training batches
2. **pretrain.py** - Trains the foundation model using the prepared data

## Loss Types Explained

The framework supports multiple pretraining objectives:

| Loss Type | Description | Use Case |
|-----------|-------------|----------|
| `motor` | MOTOR (Multi-task Temporal Risk) - Predicts time until multiple medical events | General purpose foundation model |
| `tpp` | TPP (Temporal Point Process) - Models timing of medical events | Time-focused modeling |
| `mtpp` | MTPP (Multivariate TPP) - Models both timing and lab values | When lab values are important |
| `mtpp_shared` | Shared MTPP - MTPP with shared parameters across tasks | Memory-efficient MTPP variant |
| `ntp` | NTP (Next Token Prediction) - CLMBR-style next code prediction | Language model-style pretraining |

**Note:** `mtpp` and `mtpp_shared` support the `--value_bins` parameter for discretizing continuous lab values.

## Quick Start Workflow

```bash
# Step 1: Prepare data, tokenizer, and tasks
python prepare.py \
  --pretraining_data /path/to/output \
  --ontology_path /path/to/ontology \
  --time_bins discretized_time_bins \
  --value_bins discretized_value_bins \
  --loss_type loss_type \
  --num_final_tasks pretraining_task_num \
  --hf_cache_path cache_path \ 
  --num_threads num_threads \
  --meds_reader /path/to/meds_reader

# Step 2: Train the model
accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data /path/to/output \
  --meds_reader /path/to/meds_reader \
  --output_dir /path/to/output/transformer \
  --model transformer \
  --hf_run_name my_experiment \
  --loss_type mtpp_shared
```

### 4.1. Running prepare.py

The `prepare.py` script performs the following:
- Creates train/validation/test splits
- Generates or loads an ontology from Athena
- Trains a hierarchical tokenizer
- Creates pretraining tasks (MOTOR, TPP, MTPP, or NTP)
- Converts patient data into batches for training

**Example Usage:**

```bash
python prepare.py \
  --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  --ontology_path ontology_path \
  --time_bins 8 \
  --value_bins 5 \
  --loss_type mtpp \
  --num_final_tasks 6100 \
  --num_threads 32 \
  --hf_cache_path /shared/share_mala/zj2398/huggingface/datasets \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--pretraining_data` | Yes | **Output directory** that will store all generated artifacts: train/val splits, tokenizer, task definitions, and training/evaluation batches |
| `--ontology_path` | Yes | Directory path where the ontology file (ontology.pkl) will be saved or loaded from |
| `--meds_reader` | Yes | Path to the MEDS reader database containing the input patient data |
| `--athena_path` | No | Path to Athena ontology data for creating the medical code ontology (optional if ontology already exists) |
| `--time_bins` | No | Number of time bins for temporal discretization (default: 8) |
| `--value_bins` | No | Number of bins for discretizing lab values in MTPP tasks (default: 10) |
| `--loss_type` | Yes | Type of pretraining task: `motor`, `tpp`, `mtpp`, or `ntp` |
| `--num_final_tasks` | No | Number of final prediction tasks for MTPP (default: 8192) |
| `--num_threads` | No | Number of parallel threads for data processing (default: 16) |
| `--hf_cache_path` | No | HuggingFace datasets cache directory |
| `--vocab_size` | No | Tokenizer vocabulary size (default: 16384) |
| `--tokens_per_batch` | No | Maximum tokens per training batch (default: 8192) |
| `--value_count_threshold` | No | Minimum occurrence count for lab values to be included (default: 1000) |
| `--motor_codes_to_skip` | No | Path to parquet file containing medical codes to exclude from pretraining |

**Output Structure:**

After running prepare.py, the `--pretraining_data` directory will contain:
```
pretraining_data/
├── main_split.csv              # Train/test patient splits
├── ontology.pkl                # Medical ontology
├── tokenizer/                  # Trained tokenizer files
├── {loss_type}_task.pkl        # Pretraining task definition
├── train_batches/              # Training data batches (transformer/local attention)
├── val_batches/                # Validation data batches (transformer/local attention)
├── train_batches_mamba/        # Training batches optimized for Mamba (≤8192 per sample, ≤16384 per batch)
└── val_batches_mamba/          # Validation batches optimized for Mamba
```

### 4.2. Running pretrain.py

The `pretrain.py` script trains a foundation model (Transformer or Mamba) using the prepared data. It supports multi-GPU training via Hugging Face Accelerate.

Note: When running prepare.py, pass `--model transformer` to build transformer-style batches or `--model mamba` to build Mamba-optimized batches only. Pretrain automatically loads the correct directory based on `--model`.

**Example Usage (Multi-GPU):**

```bash
# Set visible GPUs (optional)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch training with accelerate
accelerate launch \
  --num_processes 4 \
  pretrain.py \
  --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --output_dir /user/zj2398/cache/mtpp_all_shared_6100_bin_5/transformer \
  --model transformer \
  --hf_run_name mimic_mtpp_shared_aligned_task_bin_5 \
  --loss_type mtpp_shared
```

**Example Usage (Single GPU):**

```bash
python pretrain.py \
  --pretraining_data /user/zj2398/cache/mtpp_all_shared_6100_bin_5 \
  --meds_reader /shared/share_mala/zj2398/mimic/data/meds_v0.6_reader \
  --output_dir /user/zj2398/cache/output \
  --model transformer \
  --hf_run_name my_run \
  --loss_type mtpp \
  --per_device_train_batch_size 1
```

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--pretraining_data` | Yes | **Same directory from prepare.py** containing tokenizer, task, and batches |
| `--meds_reader` | Yes | Path to MEDS reader database (same as prepare.py) |
| `--output_dir` | Yes | Directory where model checkpoints will be saved during training |
| `--model` | Yes | Model architecture: `transformer` or `mamba` |
| `--loss_type` | Yes | Type of pretraining loss: `motor`, `tpp`, `mtpp`, `mtpp_shared`, or `ntp` (must match prepare.py) |
| `--hf_run_name` | Yes | Name for the training run (displayed in WandB) |
| `--checkpoint_dir` | No | Path to checkpoint to resume training from |
| `--learning_rate` | No | Learning rate for optimizer (default: 1e-5) |
| `--n_layers` | No | Number of transformer/mamba layers (default: 11) |
| `--n_epochs` | No | Number of training epochs (default: 50) |
| `--per_device_train_batch_size` | No | Training batch size per GPU (default: 1) |
| `--per_device_eval_batch_size` | No | Evaluation batch size per GPU (default: 1) |
| `--linear_interpolation` | No | Use linear interpolation for time modeling (default: False) |

**Important Notes:**

1. **Loss Type Matching**: The `--loss_type` in pretrain.py must match the task created in prepare.py
   - If you ran prepare.py with `--loss_type mtpp`, you must use `--loss_type mtpp` or `--loss_type mtpp_shared` in pretrain.py

2. **Multi-GPU Training**: Use `accelerate launch` for distributed training:
   ```bash
   accelerate launch --num_processes N pretrain.py [args]
   ```

3. **Checkpoints**: The model automatically saves checkpoints to `--output_dir` after each epoch. The best checkpoint (lowest validation loss) is saved separately.

4. **Monitoring**: Training metrics are logged to Weights & Biases (WandB) using the `--hf_run_name` identifier.

5. **Memory Requirements**: Foundation model pretraining typically requires high-memory GPUs (24GB+ recommended). Adjust `--per_device_train_batch_size` based on your GPU memory.

**Output:**

Training produces checkpoints in the `--output_dir`:
```
output_dir/
├── checkpoint-1000/
├── checkpoint-2000/
├── ...
└── checkpoint-best/          # Best performing checkpoint
```

Step 5. Model Evaluation
------------------------

After pretraining, evaluate the models on the downstream tasks (outcome, phenotype and regression)
with one command per model and task family, e.g.

```bash
GPU=0 bash src/femr/evaluation/script/mimic/gsb/transformer.sh motor all
```

The full guide, including the baselines and how to read the result tables, is
[`src/femr/evaluation/README.md`](../evaluation/README.md).
