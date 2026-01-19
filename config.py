"""
Configuration file for the Knowledge Distillation VLM training pipeline.

This file contains all constants and default hyperparameters for reproducibility.
"""

import os
from pathlib import Path

# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================
SEED = 42  # Fixed seed for reproducibility

# =============================================================================
# MODEL SETTINGS
# =============================================================================
DEFAULT_MODEL = "OpenGVLab/InternVL3_5-4B"

# =============================================================================
# PATHS
# =============================================================================
# Get the directory where this config file is located
PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset paths
DATASET_ROOT = PROJECT_ROOT / "dataset" / "collections"
GRAPH_DATASET_PATH = DATASET_ROOT / "graph_dataset_4x4"
CHARTQA_DATASET_PATH = DATASET_ROOT / "chartqa"

# Output paths
CHECKPOINT_DIR = PROJECT_ROOT / "training" / "model_checkpoints"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

# =============================================================================
# TRAINING DEFAULTS (Fixed for final model)
# =============================================================================
# These are the optimal hyperparameters found through experimentation

# Learning rate
DEFAULT_LR = 5e-4

# LoRA configuration
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16  # Usually 2x lora_r

# LLM layer training
DEFAULT_TRAIN_FIRST_N_LLM_LAYERS = 2
DEFAULT_TRAIN_LAST_N_LLM_LAYERS = 0

# Knowledge Distillation parameters
DEFAULT_KD_ALPHA = 1.0  # Weight on CE loss
DEFAULT_KD_TEMPERATURE = 2.0  # Temperature for KL divergence

# Dataset-specific KD beta (weight on KL divergence loss)
DATASET_KD_BETA = {
    "graph": 0.25,      # For graph_dataset_4x4
    "chartqa": 0.075,   # For ChartQA
}

# Dataset-specific student initialization seed (FIXED, do not change)
DATASET_STUDENT_INIT_SEED = {
    "graph": 42,
    "chartqa": 67,
}

# Teacher EMA settings
DEFAULT_USE_TEACHER_EMA = True
DEFAULT_TEACHER_EMA_DECAY = 0.0  # 1.0 means no EMA update (frozen teacher) 0.0 means student = teacher

# Representation loss (disabled in final model, was used in previous experiments
DEFAULT_KD_GAMMA = 0.0
DEFAULT_KD_GAMMA_HIDDEN = 0.0
DEFAULT_REP_LOSS_TYPE = "none"
DEFAULT_NUM_HIDDEN_LAYERS_ALIGN = 8

# Other training settings
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 8192
DEFAULT_EPOCHS = 1
DEFAULT_CONST_LR = True  # Use constant learning rate (no scheduler decay)
DEFAULT_TEACHER_CORRECT_ONLY = True # only apply the kd_loss when the teacher is correct

# Checkpoint settings
DEFAULT_CHECKPOINT_STEPS = [5,10,15, 20,25, 30,35, 40,45, 50, 60, 70, 80, 90, 100]

# =============================================================================
# EVALUATION DEFAULTS
# =============================================================================
DEFAULT_EVAL_DEVICE = "cuda"
DEFAULT_NUM_SAMPLES = None  # None means evaluate all samples

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
DATASET_CONFIGS = {
    "graph": {
        "name": "Graph VQA 4x4",
        "path": GRAPH_DATASET_PATH,
        "train_split": "train",
        "val_split": "validation", 
        "test_split": "test",
        "kd_beta": 0.25,
        "student_init_seed": 42,
        "loader_name": "DiagramMCQ",
    },
    "chartqa": {
        "name": "ChartQA",
        "path": CHARTQA_DATASET_PATH,
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
        "kd_beta": 0.075,
        "student_init_seed": 67,
        "loader_name": "ChartQA_TEST",
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_config(dataset_type: str) -> dict:
    """Get configuration for a specific dataset type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_type]


def get_kd_beta(dataset_type: str) -> float:
    """Get the KD beta value for a specific dataset."""
    return DATASET_KD_BETA.get(dataset_type, 0.25)


def get_student_init_seed(dataset_type: str) -> int:
    """Get the FIXED student initialization seed for a specific dataset.
    
    These seeds are FIXED and should NOT be changed - they ensure
    reproducible LoRA adapter initialization across runs.
    """
    return DATASET_STUDENT_INIT_SEED.get(dataset_type, 42)


def get_output_dir(dataset_type: str, run_name: str = None) -> Path:
    """Generate output directory path for checkpoints."""
    base_dir = CHECKPOINT_DIR / dataset_type
    if run_name:
        return base_dir / run_name
    return base_dir


def get_results_dir(dataset_type: str, model_name: str = None) -> Path:
    """Generate results directory path for evaluation."""
    base_dir = RESULTS_DIR / dataset_type
    if model_name:
        # Sanitize model name for directory
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return base_dir / safe_name
    return base_dir
