"""
Evaluation script for VLMEvalKit.

This script provides a complete evaluation pipeline that:
1. Works with any VLMEvalKit-supported vision-language model
2. Uses VLMEvalKit's standard inference & evaluation pipeline (infer_data_job + dataset.evaluate)

Architecture:
- Leverages VLMEvalKit's built-in inference loop (infer_data_job) instead of custom iteration
- Uses dataset's inherited evaluate() method from ImageMCQDataset for standardized accuracy computation
- VLMEvalKit handles: checkpointing, resume, distributed inference, answer extraction
"""

import os
import argparse
import logging
import random
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from typing import Optional, Any, Dict
import importlib

# Suppress VLMEvalKit's warning about custom datasets
warnings.filterwarnings('ignore', message='.*is a custom one and not annotated as.*')

# Import VLMEvalKit components
from vlmeval.config import supported_VLM
from vlmeval.smp import dump, load, get_intermediate_file_path


def compute_gap_split_metrics(predictions_file, dataset_path=None):
    """
    Compute metrics split by gap/non-gap samples.
    
    Args:
        predictions_file: Path to predictions.jsonl file with 'hit' column
        dataset_path: Path to dataset directory (to find gap_split_metadata.json)
        
    Returns:
        Dictionary with gap metrics or None if metadata not found
    """
    from pathlib import Path
    import json
    import pandas as pd
    
    predictions_file = Path(predictions_file)
    
    # Find gap_split_metadata.json
    if dataset_path:
        dataset_path = Path(dataset_path)
        # Look in parent directory (e.g., test -> graph_dataset_4x4/)
        metadata_file = dataset_path.parent / "gap_split_metadata.json"
    else:
        # Try to infer from predictions file path
        return None
    
    if not metadata_file.exists():
        logging.info(f"Gap metadata not found at {metadata_file}, skipping gap metrics")
        return None
    
    # Load gap metadata
    with open(metadata_file, 'r') as f:
        gap_metadata = json.load(f)
    
    gap_question_ids = set(gap_metadata.get('gap_question_ids', []))
    
    if not gap_question_ids:
        logging.warning("No gap question IDs found in metadata")
        return None
    
    # Load predictions
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    df = pd.DataFrame(predictions)
    
    if 'question_id' not in df.columns or 'hit' not in df.columns:
        logging.warning("Predictions file missing 'question_id' or 'hit' column")
        return None
    
    # Split by gap/non-gap
    gap_mask = df['question_id'].isin(gap_question_ids)
    
    overall_acc = df['hit'].mean() * 100
    gap_acc = df[gap_mask]['hit'].mean() * 100 if gap_mask.sum() > 0 else 0.0
    non_gap_acc = df[~gap_mask]['hit'].mean() * 100 if (~gap_mask).sum() > 0 else 0.0
    
    return {
        'overall_acc': overall_acc,
        'overall_count': len(df),
        'gap_acc': gap_acc,
        'gap_count': int(gap_mask.sum()),
        'non_gap_acc': non_gap_acc,
        'non_gap_count': int((~gap_mask).sum())
    }



from vlmeval.smp import *  # Includes: load, dump, osp, get_pred_file_path, get_intermediate_file_path
from vlmeval.dataset import build_dataset as vlmeval_build_dataset

# Import custom components
# from probing import ProbingModelWrapper
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """
    Sanitize a name to be safe for use as a directory name.
    Takes the last part after splitting by forward slash.
    
    For checkpoint paths, includes the parent directory name to avoid collisions.
    Examples:
        - "Qwen/Qwen2-VL-2B-Instruct" -> "Qwen2-VL-2B-Instruct"
        - "/path/to/model/checkpoint-200" -> "model_checkpoint-200"
        - "/path/to/model_20251223_123456" -> "model_20251223_123456"
    
    Args:
        name: The name to sanitize (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        
    Returns:
        Last part of the name (e.g., "Qwen2-VL-2B-Instruct")
    """
    # Split by forward slash and take the last part
    last_part = name.split('/')[-1]
    
    # If it's a checkpoint directory, include parent directory name to avoid collisions
    if last_part.startswith('checkpoint-'):
        parts = name.rstrip('/').split('/')
        if len(parts) >= 2:
            # Use format: parent_checkpoint-XXX
            return f"{parts[-2]}_{last_part}"
    
    return last_part


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def build_model(model_name: str, device: str = 'cuda') -> Any:
    """
    Build a VLMEvalKit model.
    
    Args:
        model_name: Name of the model (must be in supported_VLM) OR a local path to a fine-tuned checkpoint.
                   For local paths, we detect the base model from the checkpoint and load with custom weights.
        device: Device to load model on
        
    Returns:
        Initialized model instance
    """
    import os
    
    # Check if this is a local path to a fine-tuned model
    is_local_path = os.path.isdir(model_name)
    
    if is_local_path:
        logger.info(f"Detected local checkpoint path: {model_name}")
        # Try to detect the base model from the checkpoint
        # First check for training_info.json which we save during training
        training_info_path = os.path.join(model_name, "training_info.json")
        base_model_key = None
        
        if os.path.exists(training_info_path):
            import json
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
            base_model_name = training_info.get("base_model_name", "")
            logger.info(f"Found training_info.json - base model: {base_model_name}")
            
            # Map base model name to supported key
            if "InternVL3_5-4B" in base_model_name or "InternVL3.5-4B" in base_model_name:
                base_model_key = "InternVL3_5-4B"
            elif "InternVL3_5-8B" in base_model_name:
                base_model_key = "InternVL3_5-8B"
            elif "InternVL3-8B" in base_model_name:
                base_model_key = "InternVL3-8B"
            elif "InternVL2_5" in base_model_name:
                base_model_key = "InternVL2_5-8B"
        
        # Fallback: Look for config.json to identify the model architecture
        if base_model_key is None:
            config_path = os.path.join(model_name, "config.json")
            
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Detect InternVL models
                architectures = config.get("architectures", [])
                model_type = config.get("model_type", "")
                
                if "InternVLChatModel" in architectures or "internvl" in model_type.lower():
                    # Determine which InternVL version based on config or path
                    if "InternVL3_5" in model_name or "InternVL3.5" in model_name:
                        # Check size from hidden_size or path
                        if "4B" in model_name or "4b" in model_name:
                            base_model_key = "InternVL3_5-4B"
                        elif "8B" in model_name or "8b" in model_name:
                            base_model_key = "InternVL3_5-8B"
                        elif "2B" in model_name or "2b" in model_name:
                            base_model_key = "InternVL3_5-2B"
                        elif "1B" in model_name or "1b" in model_name:
                            base_model_key = "InternVL3_5-1B"
                        else:
                            base_model_key = "InternVL3_5-4B"  # default
                    elif "InternVL3" in model_name:
                        if "8B" in model_name or "8b" in model_name:
                            base_model_key = "InternVL3-8B"
                    else:
                        base_model_key = "InternVL3-8B"  # default
                elif "InternVL2_5" in model_name or "InternVL2.5" in model_name:
                    if "4B" in model_name or "4b" in model_name:
                        base_model_key = "InternVL2_5-4B"
                    elif "8B" in model_name or "8b" in model_name:
                        base_model_key = "InternVL2_5-8B"
                    else:
                        base_model_key = "InternVL2_5-8B"  # default
                else:
                    # Generic InternVL, try to guess
                    base_model_key = "InternVL3_5-4B"
                    
            # Add more model family detection here as needed
            # elif "Qwen" in str(architectures) or "qwen" in model_type.lower():
            #     ...
        
        if base_model_key is None:
            # Fallback: try to guess from directory name
            dir_name = os.path.basename(model_name)
            # Also check parent directory name if it's a checkpoint folder
            parent_dir_name = os.path.basename(os.path.dirname(model_name))
            
            if "InternVL3_5" in dir_name or "InternVL3.5" in dir_name or "InternVL3_5" in parent_dir_name:
                base_model_key = "InternVL3_5-4B"
            elif "InternVL3" in dir_name or "InternVL3" in parent_dir_name:
                base_model_key = "InternVL3-8B"
            elif "InternVL2_5" in dir_name or "InternVL2_5" in parent_dir_name:
                base_model_key = "InternVL2_5-8B"
            elif "kd_" in dir_name or "kd_" in parent_dir_name:
                # Most of our KD runs use InternVL3_5-4B
                base_model_key = "InternVL3_5-4B"
            else:
                logger.error(f"Could not detect base model for checkpoint: {model_name}")
                logger.error("Please specify a known model name or ensure config.json exists")
                raise ValueError(f"Cannot detect base model for local path: {model_name}")
        
        logger.info(f"Using base model class: {base_model_key} for fine-tuned checkpoint")
        model_class = supported_VLM[base_model_key]
        
        try:
            # Load the fine-tuned model using the base model class but with local path
            # Note: LMDeploy disabled - VLMEvalKit processes samples sequentially anyway
            # Multi-GPU via torchrun provides better parallelization
            model = model_class(model_path=model_name)
            logger.info(f"Successfully loaded fine-tuned model from {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model {model_name}: {e}")
            raise
    
    # Original logic for standard model names
    if model_name.split('/')[-1] not in supported_VLM and "google" not in model_name:
        logger.error(f"Model '{model_name.split('/')[-1]}' not supported.")
        logger.error(f"Supported models: {list(supported_VLM.keys())}")
        raise ValueError(f"Unsupported model: {model_name}")
    
    logger.info(f"Building model: {model_name.split('/')[-1]}")
    
    # Get model class
    if "google" in model_name:
        if "4b" in model_name:
            model_class = supported_VLM['Gemma3-4B']
        elif "12b" in model_name:
            model_class = supported_VLM['Gemma3-12B']
        elif "27b" in model_name:
            model_class = supported_VLM['Gemma3-27B']
    else:
        model_class = supported_VLM[model_name.split('/')[-1]]
    
    # Initialize model
    # Note: LMDeploy disabled - adds overhead without benefit since VLMEvalKit
    # processes samples one-at-a-time in a loop. Multi-GPU via torchrun is better.
    try:
        if "Qwen3" in model_name:
            model = model_class(model_path=model_name, use_vllm=False)
        else:
            model = model_class(model_path=model_name)
        logger.info(f"Successfully loaded {model_name} with standard transformers")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise


def build_dataset(
    loader_name: str, 
    dataset_path: Optional[str] = None,
    clevr_split: str = 'val',
    clevr_mode: str = 'image_only',
    arc_split: str = 'evaluation',
    arc_reasoning_mode: bool = False,
    arc_representation: str = 'simple',
    gui360_mode: str = 'visual',
    gui360_domain: str = 'all',
    gui360_category: str = 'all',
    num_samples: Optional[int] = None
) -> Any:
    """
    Build a dataset instance dynamically.
    
    Args:
        loader_name: Name of the loader (e.g., 'DiagramMCQ', 'CLEVR', 'ARCAGI', 'GUI360') OR standard dataset name (e.g., 'ChartQA_TEST')
        dataset_path: Path to dataset directory (not the file itself). Optional for standard datasets.
        clevr_split: CLEVR split to use ('val', 'train', 'test')
        clevr_mode: CLEVR mode ('image_only' or 'image_text')
        arc_split: ARC-AGI split to use ('evaluation' or 'training')
        arc_reasoning_mode: Enable extended reasoning prompts for ARC-AGI
        arc_representation: Grid representation format ('simple', 'spreadsheet', 'json')
        gui360_mode: GUI-360 evaluation mode ('visual' or 'a11y')
        gui360_domain: GUI-360 domain filter ('excel', 'word', 'ppt', or 'all')
        gui360_category: GUI-360 category filter ('in_app', 'search', 'online', 'wikihow', or 'all')
        num_samples: Limit number of samples (for debugging)
        
    Returns:
        Dataset instance
    """
    logger.info(f"Building dataset: {loader_name}")
    
    if dataset_path is None:
        # Try to load as a standard VLMEvalKit dataset
        logger.info(f"No dataset_path provided. Attempting to load as standard VLMEvalKit dataset: {loader_name}")
        try:
            dataset = vlmeval_build_dataset(loader_name)
            if dataset is None:
                raise ValueError(f"Failed to build standard dataset: {loader_name}")
            logger.info(f"Successfully loaded standard dataset: {loader_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load standard dataset {loader_name}: {e}")
            logger.error("If using a custom dataset, please provide --dataset_path")
            raise

    # Convert to Path object and get dataset name from directory
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_dir():
        raise ValueError(f"dataset_path must be a directory, got: {dataset_path}")
    
    # Dataset name is the directory name
    # Special case: if the directory is a split name (test/train/validation),
    # use the parent directory name instead
    dir_name = dataset_dir.name
    if dir_name in ['test', 'train', 'validation', 'val']:
        dataset_name = dataset_dir.parent.name
        logger.info(f"Detected split directory '{dir_name}', using parent directory name: {dataset_name}")
    else:
        dataset_name = dir_name
    
    try:
        # Import from loaders
        from loaders import diagram_mcq, chartqa_custom, clevr, arc_agi, gui_360
        
        # Get the dataset class and instantiate with appropriate arguments
        loader_lower = loader_name.lower()
        
        if loader_lower == 'diagrammcq':
            dataset = diagram_mcq.DiagramMCQDataset(dataset=dataset_name, dataset_path=str(dataset_dir))
        elif loader_lower == 'chartqacustom':
            dataset = chartqa_custom.ChartQACustomDataset(dataset=dataset_name, dataset_path=str(dataset_dir))
        elif loader_lower == 'chartqawithcsv':
            dataset = chartqa_custom.ChartQAWithCSV(dataset=dataset_name, dataset_path=str(dataset_dir))
        elif loader_lower == 'chartqawithannotations':
            dataset = chartqa_custom.ChartQAWithAnnotations(dataset=dataset_name, dataset_path=str(dataset_dir))
        elif loader_lower == 'chartqawithboth':
            dataset = chartqa_custom.ChartQAWithBoth(dataset=dataset_name, dataset_path=str(dataset_dir))
        elif loader_lower == 'clevr':
            # Full CLEVR loader with mode and split options
            dataset = clevr.CLEVRDataset(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                split=clevr_split,
                mode=clevr_mode,
                num_samples=num_samples
            )
        elif loader_lower == 'clevrtest':
            # CLEVR test split (no scene descriptions available)
            dataset = clevr.CLEVRTestDataset(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'clevrimageonly':
            # Convenience loader for image-only mode
            dataset = clevr.CLEVRImageOnlyDataset(
                dataset_path=str(dataset_dir),
                split=clevr_split,
                num_samples=num_samples
            )
        elif loader_lower == 'clevrimagetext':
            # Convenience loader for image+text mode
            dataset = clevr.CLEVRImageTextDataset(
                dataset_path=str(dataset_dir),
                split=clevr_split,
                num_samples=num_samples
            )
        elif loader_lower == 'arcagi':
            # ARC-AGI abstract reasoning benchmark
            dataset = arc_agi.ARCAGIDataset(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                split=arc_split,
                reasoning_mode=arc_reasoning_mode,
                representation=arc_representation,
                num_samples=num_samples,
                max_train_examples=3  # Limit to 3 training examples to reduce prompt length
            )
        elif loader_lower == 'arcagireasoning':
            # ARC-AGI with reasoning mode enabled
            dataset = arc_agi.ARCAGIReasoningDataset(
                dataset_path=str(dataset_dir),
                split=arc_split,
                representation=arc_representation,
                num_samples=num_samples
            )
        elif loader_lower == 'arcagitraining':
            # ARC-AGI training split
            dataset = arc_agi.ARCAGITrainingDataset(
                dataset_path=str(dataset_dir),
                reasoning_mode=arc_reasoning_mode,
                representation=arc_representation,
                num_samples=num_samples
            )
        # GUI-360 Action Prediction Benchmark
        elif loader_lower == 'gui360':
            # GUI-360 flexible loader with mode, domain, and category options
            dataset = gui_360.GUI360Dataset(
                dataset_path=str(dataset_dir),
                mode=gui360_mode,
                domain=gui360_domain,
                category=gui360_category,
                num_samples=num_samples
            )
        elif loader_lower == 'gui360visual':
            # GUI-360 visual-only mode (convenience loader)
            dataset = gui_360.GUI360VisualDataset(
                dataset_path=str(dataset_dir),
                domain=gui360_domain,
                category=gui360_category,
                num_samples=num_samples
            )
        elif loader_lower == 'gui360a11y':
            # GUI-360 with accessibility info (convenience loader)
            dataset = gui_360.GUI360A11yDataset(
                dataset_path=str(dataset_dir),
                domain=gui360_domain,
                category=gui360_category,
                num_samples=num_samples
            )
        elif loader_lower == 'gui360excel':
            # GUI-360 Excel-only (convenience loader)
            dataset = gui_360.GUI360ExcelDataset(
                dataset_path=str(dataset_dir),
                mode=gui360_mode,
                category=gui360_category,
                num_samples=num_samples
            )
        elif loader_lower == 'gui360word':
            # GUI-360 Word-only (convenience loader)
            dataset = gui_360.GUI360WordDataset(
                dataset_path=str(dataset_dir),
                mode=gui360_mode,
                category=gui360_category,
                num_samples=num_samples
            )
        elif loader_lower == 'gui360ppt':
            # GUI-360 PowerPoint-only (convenience loader)
            dataset = gui_360.GUI360PPTDataset(
                dataset_path=str(dataset_dir),
                mode=gui360_mode,
                category=gui360_category,
                num_samples=num_samples
            )
        # ChartInsights Benchmark
        elif loader_lower == 'chartinsights':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsDataset(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsmcq':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsMCQ(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsmcqwithcsv':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsMCQWithCSV(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsjudgement':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsJudgement(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsjudgementwithcsv':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsJudgementWithCSV(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsfillblank':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsFillBlank(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsfillblankwithcsv':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsFillBlankWithCSV(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightscorrective':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsCorrective(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightscorrectivewithcsv':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsCorrectiveWithCSV(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        # Overall Evaluation loaders
        elif loader_lower == 'chartinsightsoverallmcq':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallMCQ(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsoverallmcqwithcsvfiltered':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallMCQWithCSVFiltered(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsoveralljudgement':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallJudgement(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsoveralljudgementwithcsvfiltered':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallJudgementWithCSVFiltered(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsoverallfillblank':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallFillBlank(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        elif loader_lower == 'chartinsightsoverallfillblankwithcsvfiltered':
            from loaders import chartinsights
            dataset = chartinsights.ChartInsightsOverallFillBlankWithCSVFiltered(
                dataset=dataset_name,
                dataset_path=str(dataset_dir),
                num_samples=num_samples
            )
        else:
            # Fallback for other loaders - try dynamic import
            dataset_module = importlib.import_module(f'loaders.{loader_name.lower()}')
            dataset_class = getattr(dataset_module, f'{loader_name}Dataset')
            dataset = dataset_class(dataset=dataset_name, dataset_path=str(dataset_dir))
        
        logger.info(f"Successfully loaded dataset '{dataset.dataset_name}' from {dataset_dir}")
        return dataset
        
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load loader {loader_name}: {e}")
        logger.error(f"Make sure loaders/{loader_name.lower()}.py exists with {loader_name}Dataset class")
        raise ValueError(f"Loader {loader_name} not found or invalid")


def evaluate_model_with_probing(
    model: Any,
    dataset: Any,
    model_name: str,
    dataset_name: str,
    dataset_display_name: Optional[str] = None,  # For filenames (without slashes)
    output_subdir: Optional[str] = None,
    num_samples: Optional[int] = None,
    skip_inference: bool = False,
    compute_gap_metrics: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Uses VLMEvalKit's built-in inference pipeline (infer_data_job) and evaluation method.
    
    Args:
        model: Model instance (can be wrapped or unwrapped, or None if skip_inference=True)
        dataset: Dataset instance
        model_name: Name of the model
        dataset_name: Name/path for directory structure (can include slashes like "graph_dataset_4x4/test")
        dataset_display_name: Name for filenames (without slashes, defaults to last part of dataset_name)
        output_subdir: Optional subdirectory within results/ (e.g., "bidirectional" -> results/bidirectional/{dataset}/{model})
        num_samples: Limit the number of samples to evaluate (None = evaluate all)
        skip_inference: If True, only load cached results and run evaluation (no model inference)
        compute_gap_metrics: If True, compute separate metrics for gap samples
        
    Returns:
        Dictionary of evaluation results
    """
    from vlmeval.inference import infer_data_job
    from vlmeval.smp import get_pred_file_path
    from datetime import datetime
    import json
    
    # Use display name for filenames (without slashes)
    if dataset_display_name is None:
        # Extract last part of dataset_name for display (e.g., "test" from "graph_dataset_4x4/test")
        dataset_display_name = dataset_name.split('/')[-1]
    
    # Construct output directory: results/{subdir}/{dataset}/{model}
    script_dir = Path(__file__).parent.resolve()
    safe_model_name = sanitize_name(model_name)
    
    # Build path: results/ or results/{subdir}/{dataset}/{model}
    if output_subdir:
        work_dir = script_dir / "results" / output_subdir / dataset_name / safe_model_name
    else:
        work_dir = script_dir / "results" / dataset_name / safe_model_name
    
    work_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation: {model_name} on {dataset_name}")
    logger.info(f"Work directory: {work_dir}")
    
    # Limit dataset samples if specified
    if num_samples is not None and num_samples > 0:
        logger.info(f"Limiting evaluation to {num_samples} samples")
        original_data = dataset.data
        dataset.data = dataset.data.head(num_samples)
    
    # Get prediction file path using VLMEvalKit's naming convention
    # Use display_name for filename to avoid slashes in filenames
    result_file = get_pred_file_path(str(work_dir), safe_model_name, dataset_display_name, use_env_format=True)
    
    # Check if we need to run inference
    need_inference = True
    if osp.exists(result_file) and not skip_inference:
        logger.info(f"Found existing result file: {result_file}")
        data = load(result_file)
        existing_indices = set(data['index'])
        required_indices = set(dataset.data['index'])
        
        if required_indices.issubset(existing_indices):
            logger.info("All required samples already evaluated! Skipping inference.")
            need_inference = False
        else:
            missing = required_indices - existing_indices
            logger.info(f"Found {len(missing)} new samples to evaluate")
    
    # Run inference using VLMEvalKit's pipeline
    if need_inference and not skip_inference:
        if model is None:
            raise ValueError("Model is required to run inference but was not provided")
        
        logger.info("Running inference using VLMEvalKit's pipeline...")
        
        # Enable verbose mode for ARC-AGI to see prompts and responses
        dataset_class_name = dataset.__class__.__name__
        verbose_mode = 'ARCAGI' in dataset_class_name or 'ArcAgi' in dataset_class_name
        
        if verbose_mode:
            logger.info("Verbose mode enabled for ARC-AGI - will print each prompt and response")
        
        # VLMEvalKit's infer_data_job handles:
        # - Iterating through dataset
        # - Calling model.generate() 
        # - Saving predictions to JSONL/TSV
        # - Resume from checkpoints
        # - Distributed inference (if using torchrun)
        model = infer_data_job(
            model=model,
            work_dir=str(work_dir),
            model_name=safe_model_name,
            dataset=dataset,
            verbose=verbose_mode,
            api_nproc=4,
            ignore_failed=False,
            use_vllm=False
        )
        
        logger.info(f"Inference complete. Results saved to: {result_file}")
    elif skip_inference:
        logger.info("Skipping inference (--skip-inference flag set)")
    
    # Restore original dataset if we limited samples
    if num_samples is not None and num_samples > 0:
        dataset.data = original_data
    
    # Run evaluation using dataset's built-in evaluate method
    eval_results = {}
    if osp.exists(result_file):
        # Check for ground truth availability:
        # - Standard VQA datasets have 'answer' column
        # - GUI-360 has 'gt_function', 'gt_args', 'gt_status' columns
        has_standard_gt = 'answer' in dataset.data.columns
        has_gui360_gt = all(col in dataset.data.columns for col in ['gt_function', 'gt_status'])
        
        if has_standard_gt or has_gui360_gt:
            logger.info("Evaluating predictions using dataset's evaluation method...")
            
            # VLMEvalKit's evaluate method handles:
            # - Loading predictions
            # - Extracting answer letters (A/B/C/D) for MCQ
            # - Computing accuracy by category/split
            # - Saving intermediate results
            # GUI-360's evaluate method handles:
            # - Parsing action predictions
            # - Computing function/args/status accuracy
            # - Computing overall success rate
            eval_results = dataset.evaluate(str(result_file), model='exact_matching')
            
            # After evaluation, create a comprehensive result file with ground truth + predictions + hit
            logger.info("Creating comprehensive predictions file with ground truth + prediction + hit...")
            comprehensive_file = work_dir / "predictions.jsonl"
            
            # Start with full original dataset
            result_with_gt = dataset.data.copy()
            # Reset index to avoid ambiguity between index level and column
            if 'index' in result_with_gt.columns:
                result_with_gt = result_with_gt.reset_index(drop=True)
            
            # Load prediction and hit data from VLMEvalKit's result files
            # Try to load the evaluated result file (with 'hit' column)
            eval_result_file = get_intermediate_file_path(result_file, '_exact_matching_result')
            if osp.exists(eval_result_file):
                eval_data = load(eval_result_file)
                # Reset index to avoid ambiguity
                if eval_data.index.name == 'index' or 'index' in eval_data.index.names:
                    eval_data = eval_data.reset_index(drop=False)
                # Extract only prediction and hit columns
                pred_hit_data = eval_data[['index', 'prediction', 'hit']].copy()
            else:
                # Fallback: load from prediction file (no hit column yet)
                pred_data = load(result_file)
                # Reset index to avoid ambiguity
                if pred_data.index.name == 'index' or 'index' in pred_data.index.names:
                    pred_data = pred_data.reset_index(drop=False)
                pred_hit_data = pred_data[['index', 'prediction']].copy()
                # Add empty hit column
                pred_hit_data['hit'] = None
            
            # Merge prediction and hit columns into original dataset
            result_with_gt = result_with_gt.merge(
                pred_hit_data,
                on='index',
                how='left',
                suffixes=('', '_pred')
            )
            
            # Reorder columns: index, question, image, options (A/B/C/D), answer, prediction, hit, other metadata
            priority_cols = ['index', 'question', 'image', 'A', 'B', 'C', 'D', 'answer', 'prediction', 'hit']
            col_order = [c for c in priority_cols if c in result_with_gt.columns]
            other_cols = [c for c in result_with_gt.columns if c not in col_order]
            result_with_gt = result_with_gt[col_order + other_cols]
            
            # Save as JSONL
            import json
            with open(comprehensive_file, 'w') as f:
                for _, row in result_with_gt.iterrows():
                    # Convert row to dict, handling NaN values
                    row_dict = row.to_dict()
                    # Replace NaN with None for JSON serialization, handle scalars and arrays
                    cleaned_dict = {}
                    for k, v in row_dict.items():
                        try:
                            # Check if it's a scalar NaN
                            if pd.isna(v):
                                cleaned_dict[k] = None
                            else:
                                cleaned_dict[k] = v
                        except (ValueError, TypeError):
                            # If pd.isna() fails (e.g., for lists/arrays), keep original value
                            cleaned_dict[k] = v
                    f.write(json.dumps(cleaned_dict) + '\n')
            
            logger.info(f"Comprehensive predictions saved to: {comprehensive_file}")
            logger.debug(f"  - Includes: {len(result_with_gt)} samples")
            logger.debug(f"  - Columns: {', '.join(result_with_gt.columns.tolist())}")
            
            # Log results summary
            logger.info("="*50)
            logger.info(f"Evaluation Results for {model_name} on {dataset_name}")
            logger.info("="*50)
            if isinstance(eval_results, pd.DataFrame):
                # Only log first few rows and summary stats
                logger.info(f"Accuracy by category:")
                for idx, row in eval_results.iterrows():
                    if 'split' in eval_results.columns and 'Accuracy' in eval_results.columns:
                        logger.info(f"  {row.get('split', 'Overall')}: {row.get('Accuracy', 'N/A'):.2f}%")
                    else:
                        logger.debug(f"  {row}")
                logger.debug(f"Full results:\n{eval_results.to_string(index=False)}")
            else:
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                    else:
                        logger.debug(f"{key}: {value}")
            logger.info("="*50)
            
            # Compute gap-specific metrics if requested
            if compute_gap_metrics:
                logger.info("\nComputing gap-specific metrics...")
                gap_metrics = compute_gap_split_metrics(
                    comprehensive_file, 
                    dataset.dataset_path if hasattr(dataset, 'dataset_path') else None
                )
                if gap_metrics:
                    logger.info("="*50)
                    logger.info("Gap Split Metrics:")
                    logger.info("="*50)
                    logger.info(f"  Overall:     {gap_metrics['overall_acc']:.2f}% ({gap_metrics['overall_count']} samples)")
                    logger.info(f"  Gap:         {gap_metrics['gap_acc']:.2f}% ({gap_metrics['gap_count']} samples)")
                    logger.info(f"  Non-Gap:     {gap_metrics['non_gap_acc']:.2f}% ({gap_metrics['non_gap_count']} samples)")
                    logger.info("="*50)
                    
                    # Add to eval_results
                    if isinstance(eval_results, dict):
                        eval_results['gap_metrics'] = gap_metrics
        else:
            logger.info("No ground truth available. Skipping evaluation.")
            eval_results = {'note': 'No ground truth available', 'num_samples': len(dataset.data)}
    else:
        logger.warning(f"Result file not found: {result_file}")
        eval_results = {'error': 'No predictions found'}
    
    # Save metrics summary
    if output_subdir:
        dataset_work_dir = script_dir / "results" / output_subdir / dataset_name
    else:
        dataset_work_dir = script_dir / "results" / dataset_name
    
    metrics_file = dataset_work_dir / "metrics.jsonl"
    
    # Convert eval_results to JSON-serializable format
    def make_serializable(obj):
        """Convert numpy/pandas types to native Python types."""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    if isinstance(eval_results, pd.DataFrame):
        serializable_metrics = eval_results.to_dict(orient='list')
        serializable_metrics = make_serializable(serializable_metrics)
    else:
        serializable_metrics = {}
        for key, value in eval_results.items():
            if isinstance(value, pd.DataFrame):
                serializable_metrics[key] = value.to_dict(orient='records')
            elif isinstance(value, pd.Series):
                serializable_metrics[key] = value.tolist() if len(value) > 1 else value.iloc[0]
            else:
                serializable_metrics[key] = value
        serializable_metrics = make_serializable(serializable_metrics)
    
    metrics_entry = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'dataset': dataset_name,
        'num_samples': len(dataset.data),
        'result_file': str(result_file),
        'metrics': serializable_metrics
    }
    
    # Append to metrics file
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics_entry) + '\n')
    logger.info(f"Metrics appended to {metrics_file}")
    
    return eval_results


def main():
    """Main evaluation pipeline with attention probing."""
    parser = argparse.ArgumentParser(
        description='Evaluate VLMs with attention probing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Model name (must be supported by VLMEvalKit)'
    )
    
    parser.add_argument(
        '--dataset_loader',
        type=str,
        default='DiagramMCQ',
        help='Dataset loader name (e.g., DiagramMCQ, CLEVR, CLEVRImageOnly, CLEVRImageText)'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=False,
        help='Path to dataset directory (e.g., datasets/VLQA_testmini_test/). The loader will automatically find the data file (data.jsonl, data.csv, etc.) inside.'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Subdirectory within results/ for saving outputs (e.g., "bidirectional" -> results/bidirectional/{dataset}/{model})'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run model on (cuda/cpu)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--no_probing',
        action='store_true',
        help='Disable attention probing (only run evaluation)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Limit the number of samples to evaluate (for debugging)'
    )
    
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Skip inference and only run evaluation on existing results'
    )
    
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List all supported models and exit'
    )
    
    parser.add_argument(
        '--compute_gap_metrics',
        action='store_true',
        help='Compute separate metrics for gap samples (requires gap_split_metadata.json in dataset parent dir)'
    )
    
    # CLEVR-specific arguments
    parser.add_argument(
        '--clevr_split',
        type=str,
        default='val',
        choices=['val', 'train', 'test'],
        help='CLEVR split to evaluate (val has scene descriptions, test does not)'
    )
    
    parser.add_argument(
        '--clevr_mode',
        type=str,
        default='image_only',
        choices=['image_only', 'image_text'],
        help='CLEVR evaluation mode: image_only (VLM) or image_text (VLM+scene description)'
    )
    
    # ARC-AGI-specific arguments
    parser.add_argument(
        '--arc_split',
        type=str,
        default='evaluation',
        choices=['evaluation', 'training'],
        help='ARC-AGI split to evaluate (evaluation=400 tasks, training=400 tasks)'
    )
    
    parser.add_argument(
        '--arc_reasoning_mode',
        action='store_true',
        help='Enable extended reasoning prompts for ARC-AGI (encourages step-by-step reasoning)'
    )
    
    parser.add_argument(
        '--arc_representation',
        type=str,
        default='simple',
        choices=['simple', 'spreadsheet', 'json'],
        help='ARC-AGI grid representation format: simple (rows of numbers), spreadsheet (with row/col headers), json (JSON arrays)'
    )
    
    # GUI-360-specific arguments
    parser.add_argument(
        '--gui360_mode',
        type=str,
        default='visual',
        choices=['visual', 'a11y'],
        help='GUI-360 evaluation mode: visual (screenshot only) or a11y (screenshot + accessibility info)'
    )
    
    parser.add_argument(
        '--gui360_domain',
        type=str,
        default='all',
        choices=['all', 'excel', 'word', 'ppt'],
        help='GUI-360 application domain filter: excel, word, ppt, or all'
    )
    
    parser.add_argument(
        '--gui360_category',
        type=str,
        default='all',
        choices=['all', 'in_app', 'search', 'online'],
        help='GUI-360 task category filter: in_app, search, online, or all'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\nSupported VLMEvalKit Models:")
        print("="*50)
        for model_name in sorted(supported_VLM.keys()):
            print(f"  - {model_name}")
        print("="*50)
        return
    
    # Validate required arguments for evaluation
    if not args.model:
        parser.error("--model is required (unless using --list_models)")
    # dataset_path is now optional for standard datasets
    # if not args.dataset_path:
    #     parser.error("--dataset_path is required (unless using --list_models)")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build dataset first to get the actual dataset name
    logger.info("Step 1: Loading dataset...")
    dataset = build_dataset(
        args.dataset_loader, 
        dataset_path=args.dataset_path,
        clevr_split=args.clevr_split,
        clevr_mode=args.clevr_mode,
        arc_split=args.arc_split,
        arc_reasoning_mode=args.arc_reasoning_mode,
        arc_representation=args.arc_representation,
        gui360_mode=args.gui360_mode,
        gui360_domain=args.gui360_domain,
        gui360_category=args.gui360_category,
        num_samples=args.num_samples
    )
    dataset_name = dataset.dataset_name
    
    # Print a sample from the dataset
    if hasattr(dataset, 'data') and len(dataset.data) > 0:
        logger.info("="*50)
        logger.info(f"Dataset sample (first row):")
        sample_row = dataset.data.iloc[0].to_dict()
        for k, v in sample_row.items():
            # Truncate long values for display
            v_str = str(v)
            if len(v_str) > 200:
                v_str = v_str[:200] + "..."
            logger.info(f"  {k}: {v_str}")
        logger.info("="*50)

    # Check if dataset_path ends with a split directory (test/train/validation)
    # If so, we want to preserve it in the output path
    dataset_subpath = dataset_name
    if args.dataset_path:
        dataset_path_obj = Path(args.dataset_path)
        split_name = dataset_path_obj.name
        if split_name in ['test', 'train', 'validation', 'val']:
            # Use dataset_name/split structure
            dataset_subpath = f"{dataset_name}/{split_name}"
    
    # Sanitize model name for use in directory paths
    safe_model_name = sanitize_name(args.model)
    
    # Construct output directory: results/{subdir}/{dataset}/{split}/{model}
    script_dir = Path(__file__).parent.resolve()
    if args.output_dir:
        output_dir = script_dir / "results" / args.output_dir / dataset_subpath / safe_model_name
    else:
        output_dir = script_dir / "results" / dataset_subpath / safe_model_name
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting evaluation pipeline")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Dataset loader: {args.dataset_loader}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if we need to build the model (check for cached results first)
    result_file = output_dir / "predictions.jsonl"
    need_inference = True
    
    if result_file.exists():
        logger.info(f"Found existing result file: {result_file}")
        # Load existing results to check if we need more inference
        import json
        existing_results = []
        with open(result_file, 'r') as f:
            for line in f:
                existing_results.append(json.loads(line))
        
        existing_indices = {r['index'] for r in existing_results}
        logger.info(f"Found {len(existing_indices)} existing predictions")
        
        # Check what samples we need
        dataset_df_check = dataset.data
        if args.num_samples is not None and args.num_samples > 0:
            dataset_df_check = dataset_df_check.head(args.num_samples)
        
        required_indices = set(dataset_df_check['index'].tolist())
        missing_indices = required_indices - existing_indices
        
        if not missing_indices:
            logger.info("All required samples already evaluated! Skipping model loading and inference.")
            need_inference = False
        else:
            logger.info(f"Found {len(missing_indices)} new samples to evaluate. Will load model.")
    
    # Build model only if we need to run inference
    model = None
    if need_inference:
        logger.info("Step 2: Building model...")
        model = build_model(args.model, device=args.device)
    else:
        logger.info("Step 2: Skipping model loading (using cached results)")
    
    # Run evaluation
    logger.info("Step 4: Running evaluation...")
    results = evaluate_model_with_probing(
        model=model,
        dataset=dataset,
        model_name=args.model,
        dataset_name=dataset_subpath,  # Use full path including split for directory
        dataset_display_name=dataset_name,  # Use dataset name (e.g. graph_dataset_4x4) for filenames to match infer_data_job
        output_subdir=args.output_dir,
        num_samples=args.num_samples,
        skip_inference=not need_inference,
        compute_gap_metrics=args.compute_gap_metrics
    )
    
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
