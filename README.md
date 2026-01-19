# Cross Model Self Distillation (CMSD) repdroduction repository
Authors: Ryo Bertolissi, Tom Offermann, Roberto Oliveira Pais, Lejs Behric

## Overview
This repository contains the code to reproduce the main experiments (without the ablations in the appendix) from our paper: "Cross Model Self Distillation for Vision-Language Models". This includes Code to:
- Generate our dataset (see `dataset/` folder) as well as splitting into train/val/test sets (canonically without data leakage).
- Train InternVL3.5-4B on both our own GG4x4 and the ChartQA dataset using CMSD as well as the Cross Entropy (CE) baseline.
- Evaluate trained models on the test sets and compute accuracy metrics.

To recreate the results in the paper, run the commands in the "Quick Start" section below.

## Quick Start
### 1. Environment Setup
```bash
# Set up cache directories (recommended for cluster environments)
export CONDA_PKGS_DIRS="$PWD/.conda/pkgs"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export TMPDIR="$PWD/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# Create conda environment
conda create -p ./env python=3.10.*
conda activate ./env

# Install dependencies
conda env update -p ./env -f evaluation-environment.yml

# Install flash-attention (required for some models)
pip install --no-build-isolation flash-attn==2.8.3 --extra-index-url https://download.pytorch.org/whl/cu121

# Clone and install VLMEvalKit
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
cd ..
```

All of our experiments were run on an NVIDIA RTX 4090 GPU with 24GB of VRAM.  
Training requires <10min (as we only do 100 gradient steps resulting in approximately 1500 images seen).  
Evaluation takes ~15min per checkpoint.

### 2. Dataset
The datasets can be found under the `dataset/collections/` folder. 
The currently saved ones are exactly the ones used in the paper. (`graph_dataset_4x4` and `chartqa`).  
You can also generate GG4x4 using the scripts in the `dataset/` folder.
The ChartQA dataset can be downloaded from its official repository. 

### 3. Run Training + Evaluation

To run the full final training + evaluation pipeline with checkpoint saving, use the following commands:
```bash
# Full pipeline on graph dataset, with checkpoints saved and evaluated automatically
python run.py --dataset graph

# Full pipeline on ChartQA, with checkpoints saved and evaluated automatically
python run.py --dataset chartqa
```
the run.py script supports various options for training and evaluation. The mains one needed are:
```bash
--dataset: Dataset to use (`graph` or `chartqa`)
--seed: Random seed for reproducibility
--baseline`: Use CE baseline instead of CMSD
```
Many parameters (such as in what intervals checkpoints should be evaluated) can also be adjusted in `config.py`.

To recreate the data used in our paper, run all the following commands with the original `config.py` and plot the results accordingly.
```bash
# Train CMSD on GridGraph4x4 dataset
python run.py --dataset graph --seed 42
python run.py --dataset graph --seed  6
python run.py --dataset graph --seed 23
python run.py --dataset graph --seed 67
python run.py --dataset graph --seed 99
# Train CE baseline on GridGraph4x4 dataset
pyton run.py --dataset graph --seed 42 --baseline
python run.py --dataset graph --seed 6 --baseline
python run.py --dataset graph --seed 23 --baseline
python run.py --dataset graph --seed 67 --baseline
python run.py --dataset graph --seed 99 --baseline
# Train CMSD on ChartQA dataset
python run.py --dataset chartqa --seed 42
python run.py --dataset chartqa --seed 6
python run.py --dataset chartqa --seed 23
python run.py --dataset chartqa --seed 67
python run.py --dataset chartqa --seed 99
# Train CE baseline on ChartQA dataset
python run.py --dataset chartqa --seed 42 --baseline
python run.py --dataset chartqa --seed 6 --baseline
python run.py --dataset chartqa --seed 23 --baseline
python run.py --dataset chartqa --seed 67 --baseline
python run.py --dataset chartqa --seed 99 --baseline

```

## Extended Usage 

### Project Structure

```
DL-Reproducable/
├── config.py               # All configuration and constants
├── run.py                  # Unified entry point
├── environment.yml         # Conda environment
├── training/
│   ├── train.py            # Clean training script
├── evaluation/
│   ├── evaluate.py         # Clean evaluation script
│   └── loaders/            # Dataset loaders
│       ├── __init__.py
│       └── diagram_mcq.py  # MCQ dataset loader
└── dataset/
    ├── collections/        # Dataset files
    └── *.py                # Dataset generation scripts and helpers
```


### Training Only

```bash
# Train on graph dataset
python run.py --dataset graph --train_only

# Train with custom parameters
python run.py --dataset graph --train_only --kd_beta 0.3 --lr 1e-4 --epochs 5
```

### Evaluation Only

```bash
# Evaluate a checkpoint
python run.py --dataset graph --eval_only --model ./training/model_checkpoints/graph/checkpoint-100

# Evaluate base model (no training at all)
python run.py --dataset graph --eval_only --model InternVL3_5-4B

# Skip inference (compute metrics only)
python run.py --dataset graph --eval_only --model ./checkpoint --skip_inference
```

### Evaluating Multiple Checkpoints

After training, all checkpoints (10, 20, 30, ... 100) are automatically evaluated:

```bash
# Runs training + evaluates ALL checkpoints
python run.py --dataset graph

# Evaluate only specific checkpoints
python run.py --dataset graph --eval_checkpoints 50,100

# Evaluate only the final checkpoint
python run.py --dataset graph --eval_checkpoints last
```

### Direct Script Usage

You can also use the training and evaluation scripts directly:

```bash
# Training
python training/train.py --dataset_type graph --mode kd --output_dir ./output

# Evaluation
python evaluation/evaluate.py --model ./output/checkpoint-100 --dataset_type graph
```

### Configuration

All defaults are in [config.py](config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEED` | 42 | Random seed (fixed) |
| `DEFAULT_LR` | 5e-4 | Learning rate |
| `DEFAULT_LORA_R` | 8 | LoRA rank |
| `DEFAULT_LORA_ALPHA` | 16 | LoRA alpha |
| `TRAIN_FIRST_N_LLM` | 2 | Train first N LLM layers |
| `TRAIN_LAST_N_LLM` | 0 | Train last N LLM layers |
| `TEACHER_EMA` | 1.0 | Teacher EMA decay |
| `KD_TEMPERATURE` | 2.0 | KD temperature |
| `KD_BETA` | 0.25 (graph) / 0.075 (chartqa) | KD loss weight |


### Memory Requirements

- **GPU**: RTX 4090 (24GB) or equivalent
- **Training**: ~20GB VRAM with gradient checkpointing
- **Inference**: ~12GB VRAM

### Results

Checkpoints are saved to `training/model_checkpoints/<dataset>/`:
- `checkpoint-<step>/`: Model checkpoint at step
- `training_info.json`: Training configuration

Evaluation results are saved to `evaluation/results/`:
- `<model_name>_<dataset>_results.jsonl`: Per-sample predictions
- `<model_name>_<dataset>_metrics.json`: Accuracy metrics

