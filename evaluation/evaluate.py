"""
Evaluation script for trained VLM models.

This script evaluates vision-language models on graph VQA and other datasets.
It supports both standard VLMEvalKit models and locally fine-tuned checkpoints.

Usage:
    # Evaluate a fine-tuned checkpoint on graph dataset
    python evaluate.py --model ./path/to/checkpoint --dataset_type graph
    
    # Evaluate base model on graph dataset
    python evaluate.py --model InternVL3_5-4B --dataset_type graph
    
    # Evaluate on ChartQA
    python evaluate.py --model ./path/to/checkpoint --dataset_type chartqa
    
    # Skip inference (only evaluate existing predictions)
    python evaluate.py --model ./path/to/checkpoint --skip_inference
"""

import os
import sys
import argparse
import logging
import random
import json
import warnings
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

# Suppress VLMEvalKit's warning about custom datasets
warnings.filterwarnings('ignore', message='.*is a custom one and not annotated as.*')

# =============================================================================
# InternVL Image Processing (for LoRA model evaluation)
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image_for_internvl(image, input_size=448, max_num=1):
    """Process a PIL image for InternVL. Returns pixel_values tensor."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert('RGB')
    else:
        image = image.convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SEED,
    DEFAULT_MODEL,
    GRAPH_DATASET_PATH,
    RESULTS_DIR,
    DATASET_CONFIGS,
    get_results_dir,
)

# Import VLMEvalKit components
from vlmeval.config import supported_VLM
from vlmeval.smp import dump, load, get_intermediate_file_path, osp, get_pred_file_path
from vlmeval.dataset import build_dataset as vlmeval_build_dataset
from vlmeval.inference import infer_data_job

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_name(name: str) -> str:
    """
    Sanitize a name for use as a directory name.
    
    Examples:
        - "Qwen/Qwen2-VL-2B-Instruct" -> "Qwen2-VL-2B-Instruct"
        - "/path/to/model/checkpoint-200" -> "model_checkpoint-200"
    """
    last_part = name.split('/')[-1]
    
    if last_part.startswith('checkpoint-'):
        parts = name.rstrip('/').split('/')
        if len(parts) >= 2:
            return f"{parts[-2]}_{last_part}"
    
    return last_part


def is_lora_checkpoint(path: str) -> bool:
    """Check if a directory contains a LoRA/PEFT checkpoint (adapter_model files).
    
    Checks both the root directory and the 'student/' subdirectory,
    since our KD training saves adapters under a named adapter directory.
    """
    p = Path(path)
    # PEFT saves adapter_model.safetensors or adapter_model.bin
    # Check root directory
    if (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists():
        return True
    # Check student/ subdirectory (KD training uses named adapters)
    if (p / "student" / "adapter_model.safetensors").exists() or (p / "student" / "adapter_model.bin").exists():
        return True
    return False


def build_model(model_name: str, device: str = 'cuda') -> Any:
    """
    Build a VLMEvalKit model.
    
    Args:
        model_name: Name of the model (in supported_VLM) or path to fine-tuned checkpoint
        device: Device to load model on
        
    Returns:
        Initialized model instance
    """
    is_local_path = os.path.isdir(model_name)
    
    if is_local_path:
        logger.info(f"Loading checkpoint: {model_name}")
        
        # Check if this is a LoRA checkpoint (only adapter weights)
        if is_lora_checkpoint(model_name):
            logger.info("Detected LoRA checkpoint - will load base model + adapters")
            return build_lora_model(model_name, device)
        
        # Try to detect base model from training_info.json
        training_info_path = os.path.join(model_name, "training_info.json")
        base_model_key = None
        
        if os.path.exists(training_info_path):
            with open(training_info_path, 'r') as f:
                training_info = json.load(f)
            base_model_name = training_info.get("base_model_name", "")
            logger.info(f"Base model from training_info: {base_model_name}")
            
            # Map to VLMEvalKit key
            if "InternVL3_5-4B" in base_model_name or "InternVL3.5-4B" in base_model_name:
                base_model_key = "InternVL3_5-4B"
            elif "InternVL3_5-8B" in base_model_name:
                base_model_key = "InternVL3_5-8B"
            elif "InternVL3-8B" in base_model_name:
                base_model_key = "InternVL3-8B"
        
        # Fallback detection from config.json
        if base_model_key is None:
            config_path = os.path.join(model_name, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                architectures = config.get("architectures", [])
                if "InternVLChatModel" in architectures:
                    if "4B" in model_name or "4b" in model_name:
                        base_model_key = "InternVL3_5-4B"
                    elif "8B" in model_name or "8b" in model_name:
                        base_model_key = "InternVL3_5-8B"
                    else:
                        base_model_key = "InternVL3_5-4B"
        
        # Ultimate fallback
        if base_model_key is None:
            base_model_key = "InternVL3_5-4B"
            logger.warning(f"Could not detect base model, using default: {base_model_key}")
        
        logger.info(f"Using base model class: {base_model_key}")
        model_class = supported_VLM[base_model_key]
        
        try:
            model = model_class(model_path=model_name)
            logger.info(f"Successfully loaded fine-tuned model")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    # Standard model name
    model_key = model_name.split('/')[-1]
    if model_key not in supported_VLM:
        logger.error(f"Model '{model_key}' not supported.")
        logger.error(f"Supported models: {list(supported_VLM.keys())[:10]}...")
        raise ValueError(f"Unsupported model: {model_name}")
    
    logger.info(f"Building standard model: {model_key}")
    model_class = supported_VLM[model_key]
    
    try:
        model = model_class(model_path=model_name)
        logger.info(f"Successfully loaded {model_key}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def build_lora_model(checkpoint_path: str, device: str = 'cuda') -> Any:
    """
    Build a model from a LoRA checkpoint.
    
    This loads the base model and applies the LoRA adapters from the checkpoint.
    Returns a VLMEvalKit-compatible wrapper.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint directory
        device: Device to load model on
        
    Returns:
        Model wrapper compatible with VLMEvalKit evaluation
    """
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel
    
    checkpoint_path = Path(checkpoint_path)
    
    # Find training_info.json in checkpoint or parent directory
    training_info = None
    for search_path in [checkpoint_path, checkpoint_path.parent]:
        info_path = search_path / "training_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                training_info = json.load(f)
            break
    
    # Determine base model
    if training_info:
        base_model_name = training_info.get("base_model_name", DEFAULT_MODEL)
    else:
        base_model_name = DEFAULT_MODEL
        logger.warning(f"No training_info.json found, using default base model: {base_model_name}")
    
    logger.info(f"Loading base model: {base_model_name}")
    
    # Load base model
    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device != 'cpu' else torch.float32,
        trust_remote_code=True,
        device_map='auto' if device != 'cpu' else None,
        low_cpu_mem_usage=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    
    # Determine the actual adapter path
    # KD training saves adapters in a 'student/' subdirectory (named adapter)
    adapter_path = checkpoint_path
    if (checkpoint_path / "student" / "adapter_model.safetensors").exists() or \
       (checkpoint_path / "student" / "adapter_model.bin").exists():
        adapter_path = checkpoint_path / "student"
        logger.info(f"Found adapters in 'student/' subdirectory")
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    
    # Merge adapters for faster inference (optional but recommended)
    logger.info("Merging LoRA adapters for inference...")
    model = model.merge_and_unload()
    
    if device != 'cpu':
        model = model.cuda()
    model.eval()
    
    logger.info("Successfully loaded LoRA model")
    
    # Wrap in a VLMEvalKit-compatible interface
    return LoRAModelWrapper(model, tokenizer, base_model_name)


class LoRAModelWrapper:
    """
    Wrapper to make a LoRA-loaded model compatible with VLMEvalKit evaluation.
    
    Implements the VLMEvalKit BaseModel interface with required methods:
    - set_dump_image / dump_image for image handling
    - generate / generate_inner for inference
    """
    
    INTERLEAVE = False
    allowed_types = ['text', 'image']
    
    def __init__(self, model, tokenizer, model_name: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.dump_image_func = None
        
        # Try to get generation config
        if hasattr(model, 'generation_config'):
            self.generation_config = model.generation_config
        else:
            self.generation_config = None
    
    def set_dump_image(self, dump_image_func):
        """Set the image dump function (required by VLMEvalKit)."""
        self.dump_image_func = dump_image_func
    
    def dump_image(self, line, dataset=None):
        """Dump image using the provided function (required by VLMEvalKit)."""
        if self.dump_image_func is not None:
            return self.dump_image_func(line)
        return None
    
    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset."""
        return False
    
    def check_content(self, msgs):
        """Check the content type of the input."""
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'
    
    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts."""
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                # Simple check if it's an image path
                if s.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    res.append(dict(type='image', value=s))
                else:
                    res.append(dict(type='text', value=s))
            return res
        elif self.check_content(inputs) == 'listdict':
            return inputs
        return None
    
    def generate(self, message, dataset=None):
        """Generate output - VLMEvalKit interface."""
        message = self.preproc_content(message)
        if message is None:
            return ""
        return self.generate_inner(message, dataset)
    
    def generate_inner(self, message, dataset=None):
        """
        Generate a response for the given message.
        
        This method matches VLMEvalKit's expected interface.
        For InternVL models, we need to preprocess images to pixel_values tensors.
        """
        # Extract image and text from message
        image = None
        prompt_parts = []
        
        for item in message:
            if item['type'] == 'image':
                img_path = item['value']
                if isinstance(img_path, str) and os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                elif isinstance(img_path, Image.Image):
                    image = img_path.convert('RGB')
            elif item['type'] == 'text':
                prompt_parts.append(item['value'])
        
        prompt = '\n'.join(prompt_parts)
        
        # Build conversation and generate
        if image is not None:
            # Use InternVL's chat method if available
            if hasattr(self.model, 'chat'):
                # Preprocess image to pixel_values tensor (InternVL expects this)
                pixel_values = load_image_for_internvl(image, input_size=448, max_num=1)
                device = next(self.model.parameters()).device
                dtype = next(self.model.parameters()).dtype
                pixel_values = pixel_values.to(device=device, dtype=dtype)
                
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config=dict(max_new_tokens=512, do_sample=False),
                )
                return response
        
        # Fallback: use standard generation (text-only)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response


def build_dataset(
    dataset_type: str,
    dataset_path: Optional[str] = None,
    split: str = "test",
    num_samples: Optional[int] = None
) -> Any:
    """
    Build a dataset for evaluation.
    
    Args:
        dataset_type: Type of dataset ('graph' or 'chartqa')
        dataset_path: Optional custom path to dataset
        split: Dataset split to evaluate
        num_samples: Limit number of samples (for debugging)
        
    Returns:
        Dataset instance
    """
    config = DATASET_CONFIGS.get(dataset_type)
    if config is None:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # For ChartQA, use VLMEvalKit's built-in dataset
    if dataset_type == "chartqa":
        logger.info("Loading ChartQA from VLMEvalKit")
        dataset = vlmeval_build_dataset("ChartQA_TEST")
        if dataset is None:
            raise ValueError("Failed to build ChartQA dataset")
        return dataset
    
    # For graph dataset, use custom loader
    path = Path(dataset_path) if dataset_path else config["path"]
    if path is None:
        raise ValueError(f"Dataset path not specified for {dataset_type}")
    
    # Determine split directory
    split_dir = path / split if (path / split).exists() else path
    
    logger.info(f"Loading {dataset_type} dataset from: {split_dir}")
    
    # Import custom loader - handle both running from project root and evaluation dir
    from loaders.diagram_mcq import DiagramMCQDataset
    from loaders.chartqa_custom import ChartQACustomDataset
    
    dataset_name = path.name if path.name not in ['test', 'train', 'validation'] else path.parent.name
    dataset = DiagramMCQDataset(dataset=dataset_name, dataset_path=str(split_dir))
    
    logger.info(f"Loaded {len(dataset.data)} samples")
    return dataset


def evaluate_model(
    model: Any,
    dataset: Any,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    num_samples: Optional[int] = None,
    skip_inference: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model instance (or None if skip_inference=True)
        dataset: Dataset instance
        model_name: Name of the model for output files
        dataset_name: Name of the dataset
        output_dir: Directory for results
        num_samples: Limit samples (for debugging)
        skip_inference: Only evaluate existing predictions
        
    Returns:
        Dictionary of evaluation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_model_name = sanitize_name(model_name)
    
    logger.info(f"Evaluating {model_name} on {dataset_name}")
    logger.info(f"Output directory: {output_dir}")
    
    # Limit samples if specified
    if num_samples is not None and num_samples > 0:
        logger.info(f"Limiting to {num_samples} samples")
        original_data = dataset.data
        dataset.data = dataset.data.head(num_samples)
    
    # Get prediction file path
    result_file = get_pred_file_path(str(output_dir), safe_model_name, dataset_name, use_env_format=True)
    
    # Check if inference needed
    need_inference = True
    if osp.exists(result_file) and not skip_inference:
        logger.info(f"Found existing predictions: {result_file}")
        data = load(result_file)
        existing_indices = set(data['index'])
        required_indices = set(dataset.data['index'])
        
        if required_indices.issubset(existing_indices):
            logger.info("All samples already evaluated, skipping inference")
            need_inference = False
        else:
            missing = required_indices - existing_indices
            logger.info(f"Found {len(missing)} new samples to evaluate")
    
    # Run inference
    if need_inference and not skip_inference:
        if model is None:
            raise ValueError("Model required for inference")
        
        logger.info("Running inference...")
        model = infer_data_job(
            model=model,
            work_dir=str(output_dir),
            model_name=safe_model_name,
            dataset=dataset,
            verbose=False,
            api_nproc=4,
            ignore_failed=False,
            use_vllm=False
        )
        logger.info(f"Inference complete: {result_file}")
    elif skip_inference:
        logger.info("Skipping inference (--skip_inference)")
    
    # Restore dataset if limited
    if num_samples is not None and num_samples > 0:
        dataset.data = original_data
    
    # Run evaluation
    eval_results = {}
    if osp.exists(result_file):
        has_gt = 'answer' in dataset.data.columns
        
        if has_gt:
            logger.info("Running evaluation...")
            eval_results = dataset.evaluate(str(result_file), model='exact_matching')
            
            # Create comprehensive results file
            comprehensive_file = output_dir / "predictions.jsonl"
            result_df = dataset.data.copy()
            if 'index' in result_df.columns:
                result_df = result_df.reset_index(drop=True)
            
            # Load predictions
            eval_result_file = get_intermediate_file_path(result_file, '_exact_matching_result')
            if osp.exists(eval_result_file):
                eval_data = load(eval_result_file)
                if eval_data.index.name == 'index':
                    eval_data = eval_data.reset_index(drop=False)
                pred_hit = eval_data[['index', 'prediction', 'hit']].copy()
            else:
                pred_data = load(result_file)
                if pred_data.index.name == 'index':
                    pred_data = pred_data.reset_index(drop=False)
                pred_hit = pred_data[['index', 'prediction']].copy()
                pred_hit['hit'] = None
            
            result_df = result_df.merge(pred_hit, on='index', how='left')
            
            # Save
            with open(comprehensive_file, 'w') as f:
                for _, row in result_df.iterrows():
                    row_dict = {}
                    for k, v in row.to_dict().items():
                        try:
                            if pd.isna(v):
                                row_dict[k] = None
                            else:
                                row_dict[k] = v
                        except (ValueError, TypeError):
                            row_dict[k] = v
                    f.write(json.dumps(row_dict) + '\n')
            
            logger.info(f"Results saved to: {comprehensive_file}")
            
            # Print results
            logger.info("=" * 50)
            logger.info("Evaluation Results")
            logger.info("=" * 50)
            if isinstance(eval_results, pd.DataFrame):
                for idx, row in eval_results.iterrows():
                    if 'Accuracy' in eval_results.columns:
                        logger.info(f"Accuracy: {row.get('Accuracy', 'N/A'):.2f}%")
            else:
                for key, value in eval_results.items():
                    if isinstance(value, float):
                        logger.info(f"{key}: {value:.4f}")
                    else:
                        logger.info(f"{key}: {value}")
            logger.info("=" * 50)
        else:
            logger.info("No ground truth available")
            eval_results = {'note': 'No ground truth', 'num_samples': len(dataset.data)}
    else:
        logger.warning(f"No predictions found: {result_file}")
        eval_results = {'error': 'No predictions'}
    
    # Save metrics
    metrics_file = output_dir / "metrics.json"
    
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'dataset': dataset_name,
        'num_samples': len(dataset.data),
        'results': make_serializable(eval_results),
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to: {metrics_file}")
    
    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate VLM models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (from VLMEvalKit) or path to fine-tuned checkpoint'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='graph',
        choices=['graph', 'chartqa'],
        help='Dataset type to evaluate on'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Custom dataset path (overrides default)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to evaluate'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Limit samples (for debugging)'
    )
    
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Only evaluate existing predictions'
    )
    
    parser.add_argument(
        '--list_models',
        action='store_true',
        help='List supported models and exit'
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list_models:
        print("\nSupported VLMEvalKit Models:")
        print("=" * 50)
        for name in sorted(supported_VLM.keys())[:30]:
            print(f"  - {name}")
        print("  ...")
        print("=" * 50)
        return
    
    # Set seed
    set_seed()
    
    # Build dataset
    logger.info("Loading dataset...")
    dataset = build_dataset(
        args.dataset_type,
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=args.num_samples
    )
    dataset_name = dataset.dataset_name if hasattr(dataset, 'dataset_name') else args.dataset_type
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_results_dir(args.dataset_type, sanitize_name(args.model))
    
    # Check if we need to build model
    need_model = not args.skip_inference
    
    # Build model if needed
    model = None
    if need_model:
        logger.info("Loading model...")
        model = build_model(args.model, device=args.device)
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        dataset=dataset,
        model_name=args.model,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_samples=args.num_samples,
        skip_inference=args.skip_inference,
    )
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
