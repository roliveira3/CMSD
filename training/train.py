"""
Training script for Graph VQA with Knowledge Distillation using LoRA.

This script trains a Vision-Language Model (VLM) using knowledge distillation:
- Student: Model with LoRA adapters enabled (receives image-only input)
- Teacher: Same model with LoRA disabled (receives image + text description)

The approach is memory-efficient as we only load ONE model and toggle LoRA on/off.

Usage:
    # Train on graph dataset (default)
    python train.py
    
    # Train on graph dataset with custom output directory
    python train.py --output_dir ./my_checkpoints
    
    # Train on ChartQA dataset
    python train.py --dataset_type chartqa

    # Train without KD (image-only baseline)
    python train.py --mode image
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import PrinterCallback
from peft import LoraConfig, get_peft_model

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SEED,
    DEFAULT_MODEL,
    GRAPH_DATASET_PATH,
    CHARTQA_DATASET_PATH,
    DEFAULT_LR,
    DEFAULT_LORA_R,
    DEFAULT_TRAIN_FIRST_N_LLM_LAYERS,
    DEFAULT_TRAIN_LAST_N_LLM_LAYERS,
    DEFAULT_KD_ALPHA,
    DEFAULT_KD_TEMPERATURE,
    DEFAULT_USE_TEACHER_EMA,
    DEFAULT_TEACHER_EMA_DECAY,
    DEFAULT_KD_GAMMA,
    DEFAULT_KD_GAMMA_HIDDEN,
    DEFAULT_REP_LOSS_TYPE,
    DEFAULT_NUM_HIDDEN_LAYERS_ALIGN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_EPOCHS,
    DEFAULT_CONST_LR,
    DEFAULT_TEACHER_CORRECT_ONLY,
    DEFAULT_CHECKPOINT_STEPS,
    get_kd_beta,
    get_student_init_seed,
    get_output_dir,
    DATASET_CONFIGS,
)


# =============================================================================
# RNG State Management for Deterministic Adapter Creation
# =============================================================================

def seed_all(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> Dict[str, Any]:
    """Capture current RNG state from all sources."""
    state = {
        'python_random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG state from a previously captured state."""
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


# =============================================================================
# InternVL Image Processing
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


# =============================================================================
# Dataset
# =============================================================================

class GraphMCQDataset(Dataset):
    """Dataset for graph MCQ task with both image and text prompts."""
    
    def __init__(self, data_path: str, tokenizer, max_length=1024, split: Optional[str] = None, max_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        data_path = Path(data_path)
        jsonl_path = data_path / "data.jsonl" if data_path.is_dir() else data_path
        self.base_dir = jsonl_path.parent
        
        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    if split is not None and sample.get("split") != split:
                        continue
                    self.samples.append(sample)
                    if max_samples is not None and len(self.samples) >= max_samples:
                        break
        
        suffix = f" (split={split})" if split is not None else ""
        print(f"Loaded {len(self.samples)} samples{suffix}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        def _split_graph_and_question(text_prompt: str, question: str) -> Dict[str, str]:
            """Parse graph representation and question from text_prompt."""
            if not isinstance(text_prompt, str) or not text_prompt.strip():
                return {"graph_text": "", "question_text": question.strip() if isinstance(question, str) else ""}
            if isinstance(question, str) and question.strip():
                text_trimmed = text_prompt.rstrip()
                q_trimmed = question.strip()
                if text_trimmed.endswith(q_trimmed):
                    graph_part = text_trimmed[: -len(q_trimmed)].rstrip()
                    if graph_part.endswith("\n\n"):
                        graph_part = graph_part[:-2].rstrip()
                    elif graph_part.endswith("\n"):
                        graph_part = graph_part[:-1].rstrip()
                    return {"graph_text": graph_part, "question_text": q_trimmed}
            sep = "\n\n"
            pos = text_prompt.rfind(sep)
            if pos == -1:
                return {"graph_text": text_prompt.strip(), "question_text": ""}
            return {
                "graph_text": text_prompt[:pos].strip(),
                "question_text": text_prompt[pos + len(sep):].strip(),
            }
        
        # Format options
        options = sample.get("options", [])
        options_prompt = "Options:\n"
        for i, opt in enumerate(options):
            options_prompt += f"{chr(65+i)}. {opt}\n"

        answer_idx = sample.get("answer_idx", None)
        if answer_idx is None:
            answer_text = sample.get("answer", None)
            if answer_text is not None and answer_text in options:
                answer_idx = options.index(answer_text)
            else:
                answer_idx = 0
        answer = chr(65 + int(answer_idx))
        
        # Build prompt
        question = sample['question']
        student_question_text = f"Question: {question}\n"
        student_question_text += options_prompt
        student_question_text += "Do not explain. Answer with a single uppercase letter (A, B, C, or D) and nothing else.\nAnswer:\n"
        
        # Student prompt (image-based)
        student_prompt = f"<image>\n{student_question_text}"
        student_text = f"{student_prompt}{answer}"
        
        # Teacher prompt (image + text description)
        teacher_text_prompt = sample.get("text_prompt", "").strip()
        split_prompt = _split_graph_and_question(teacher_text_prompt, question)
        graph_text = split_prompt["graph_text"]

        teacher_context = f"{teacher_text_prompt}\n"
        teacher_context += options_prompt
        teacher_context += "Do not explain. Answer with a single uppercase letter (A, B, C, or D) and nothing else.\nAnswer:\n"

        teacher_prompt = f"<image>\n{teacher_context}"
        teacher_text = f"{teacher_prompt}{answer}"
        
        # Load image
        image_path = sample.get("image", "")
        if image_path:
            try:
                image = Image.open(self.base_dir / image_path).convert("RGB")
            except:
                image = Image.new("RGB", (448, 448), (255, 255, 255))
        else:
            image = Image.new("RGB", (448, 448), (255, 255, 255))
        
        return {
            "idx": idx,
            "answer": answer,
            "image": image,
            "question": question,
            "graph_text": graph_text,
            "student_text": student_text,
            "student_prompt": student_prompt,
            "teacher_text": teacher_text,
            "teacher_prompt": teacher_prompt,
        }


class KDDataCollator:
    """Data collator that prepares both student and teacher inputs."""
    
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    
    def __init__(self, tokenizer, max_length=1024, mode="kd", num_image_token=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.num_image_token = num_image_token
    
    def __call__(self, features):
        batch = {}
        
        # Process student inputs (image mode)
        if self.mode in ["image", "kd"]:
            images = [f["image"] for f in features]
            pixel_values_list = []
            num_patches_list = []
            
            for img in images:
                pv = load_image_for_internvl(img, input_size=448, max_num=1)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
            
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            
            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                image_tokens = (
                    self.IMG_START_TOKEN + 
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + 
                    self.IMG_END_TOKEN
                )
                prompt = f["student_prompt"].replace("<image>", image_tokens)
                answer = f["answer"]
                
                prompt_encoding = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length-5
                )
                answer_encoding = self.tokenizer(
                    answer, add_special_tokens=False, truncation=False
                )
                
                input_ids = prompt_encoding["input_ids"] + answer_encoding["input_ids"]
                attention_mask = prompt_encoding["attention_mask"] + answer_encoding["attention_mask"]
                labels = [-100] * len(prompt_encoding["input_ids"]) + answer_encoding["input_ids"]
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)
            
            max_len = max(len(ids) for ids in all_input_ids)
            padded_input_ids = []
            padded_attention_masks = []
            padded_labels = []
            
            for input_ids, attention_mask, labels in zip(all_input_ids, all_attention_masks, all_labels):
                pad_len = max_len - len(input_ids)
                padded_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_attention_masks.append(attention_mask + [0] * pad_len)
                padded_labels.append(labels + [-100] * pad_len)
            
            batch["input_ids"] = torch.tensor(padded_input_ids, dtype=torch.long)
            batch["attention_mask"] = torch.tensor(padded_attention_masks, dtype=torch.long)
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            
            pixel_values = torch.cat(pixel_values_list, dim=0)
            batch["pixel_values"] = pixel_values
            batch["image_flags"] = torch.ones(pixel_values.shape[0], dtype=torch.long)
        
        # Process teacher inputs (text + image mode) - only for KD
        if self.mode == "kd":
            teacher_texts = []
            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                image_tokens = (
                    self.IMG_START_TOKEN + 
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + 
                    self.IMG_END_TOKEN
                )
                text = f["teacher_text"].replace("<image>", image_tokens)
                teacher_texts.append(text)
            
            teacher_encodings = self.tokenizer(
                teacher_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            
            batch["teacher_input_ids"] = teacher_encodings["input_ids"]
            batch["teacher_attention_mask"] = teacher_encodings["attention_mask"]
            batch["teacher_pixel_values"] = pixel_values.clone()
            batch["teacher_image_flags"] = torch.ones(pixel_values.shape[0], dtype=torch.long)
        
        # Text-only mode
        if self.mode == "text":
            texts = [f["teacher_text"] for f in features]
            prompts = [f["teacher_prompt"] for f in features]
            
            encodings = self.tokenizer(
                texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            
            batch["input_ids"] = encodings["input_ids"]
            batch["attention_mask"] = encodings["attention_mask"]
            
            labels = encodings["input_ids"].clone()
            for i, prompt in enumerate(prompts):
                prompt_tokens = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length
                )["input_ids"]
                labels[i, :min(len(prompt_tokens), labels.shape[1])] = -100
            batch["labels"] = labels
        
        return batch


# =============================================================================
# KD Trainer
# =============================================================================

class KDTrainer(Trainer):
    """
    Trainer with Knowledge Distillation support.
    
    Uses a single model with LoRA adapters:
    - Student forward: LoRA ENABLED (trainable adapters active)
    - Teacher forward: LoRA DISABLED (frozen pretrained weights)
    """
    
    def __init__(
        self,
        *args,
        kd_alpha=1.0,
        kd_temperature=2.0,
        kd_beta: float = 0.25,
        enable_kd: bool = True,
        teacher_correct_only: bool = True,
        use_teacher_ema: bool = True,
        teacher_ema_decay: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.use_kd = enable_kd
        self.kd_beta = kd_beta
        self.teacher_correct_only = teacher_correct_only
        self.use_teacher_ema = use_teacher_ema
        self.teacher_ema_decay = teacher_ema_decay
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        peft_model = model.module if hasattr(model, 'module') else model
        base_model = peft_model
        if hasattr(base_model, 'base_model'):
            base_model = base_model.base_model.model
        
        model_dtype = next(base_model.parameters()).dtype
        
        # Student forward pass (with LoRA ENABLED)
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "pixel_values" in inputs:
            student_inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
        if "image_flags" in inputs:
            student_inputs["image_flags"] = inputs["image_flags"]
        
        labels = inputs["labels"]
        outputs = base_model(**student_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # CE Loss
        valid_targets = (shift_labels != -100).view(-1)
        if valid_targets.sum() == 0:
            ce_loss = 0.0 * shift_logits.sum()
        else:
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        # KD Loss
        use_teacher = ("teacher_input_ids" in inputs) and self.use_kd
        if use_teacher:
            with torch.no_grad():
                batch_size = inputs["teacher_input_ids"].shape[0]
                
                if self.use_teacher_ema:
                    peft_model.set_adapter("teacher")
                else:
                    peft_model.disable_adapter_layers()
                
                try:
                    teacher_outputs = base_model(
                        input_ids=inputs["teacher_input_ids"],
                        attention_mask=inputs["teacher_attention_mask"],
                        pixel_values=inputs["teacher_pixel_values"].to(model_dtype),
                        image_flags=inputs["teacher_image_flags"],
                    )
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs[0]
                finally:
                    peft_model.set_adapter("student")
            
            # Compute KD loss for answer tokens
            kd_loss = ce_loss.new_zeros(())
            T = self.kd_temperature
            kd_loss_accum = 0.0
            num_valid = 0
            
            for i in range(batch_size):
                student_valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                if len(student_valid_positions) == 0:
                    continue
                
                student_answer_pos = student_valid_positions[0].item()
                student_pred_pos = student_answer_pos - 1
                
                if student_pred_pos < 0:
                    continue
                
                teacher_seq_len = inputs["teacher_attention_mask"][i].sum().item()
                teacher_pred_pos = teacher_seq_len - 2
                
                if teacher_pred_pos < 0:
                    continue
                
                student_logits_i = logits[i, student_pred_pos, :]
                teacher_logits_i = teacher_logits[i, teacher_pred_pos, :]
                
                # Check teacher correctness if required
                if self.teacher_correct_only:
                    gt_answer_token = labels[i, student_answer_pos].item()
                    teacher_pred_token = teacher_logits_i.argmax().item()
                    if teacher_pred_token != gt_answer_token:
                        continue
                
                student_soft = F.log_softmax(student_logits_i / T, dim=-1)
                teacher_soft = F.softmax(teacher_logits_i / T, dim=-1)
                
                kd_loss_i = F.kl_div(student_soft, teacher_soft, reduction="sum")
                kd_loss_accum = kd_loss_accum + kd_loss_i
                num_valid += 1
            
            if num_valid > 0:
                kd_loss = (kd_loss_accum / num_valid) * (T ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
        
        if self.use_kd:
            loss = self.kd_alpha * ce_loss + self.kd_beta * kd_loss
        else:
            loss = ce_loss
        
        # Log losses
        if self.state.global_step > 0 and self.state.global_step % 10 == 0:
            log_payload = {"ce_loss": ce_loss.item()}
            if kd_loss is not None:
                log_payload["kd_loss"] = kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss
            self.log(log_payload)
            print(
                f"[step {self.state.global_step}] "
                f"ce_loss={log_payload.get('ce_loss', -1):.6f} "
                f"kd_loss={log_payload.get('kd_loss', -1):.6f}",
                flush=True,
            )
        
        return (loss, outputs) if return_outputs else loss

    def _update_teacher_ema(self):
        """Update teacher adapter with EMA of student adapter weights."""
        if not self.use_teacher_ema or self.teacher_ema_decay >= 1.0:
            return
        
        peft_model = self.model.module if hasattr(self.model, 'module') else self.model
        student_params = {n: p for n, p in peft_model.named_parameters() if 'lora' in n and 'student' in n}
        teacher_params = {n: p for n, p in peft_model.named_parameters() if 'lora' in n and 'teacher' in n}
        
        with torch.no_grad():
            for name in student_params.keys():
                teacher_name = name.replace('.student.', '.teacher.')
                if teacher_name in teacher_params:
                    teacher_params[teacher_name].mul_(self.teacher_ema_decay).add_(
                        student_params[name], alpha=1 - self.teacher_ema_decay
                    )
    
    def training_step(self, model, inputs, num_items_in_batch):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.use_teacher_ema and self.state.global_step > 0:
            self._update_teacher_ema()
        return loss


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train VLM with Knowledge Distillation')
    
    # Dataset selection
    parser.add_argument('--dataset_type', type=str, default='graph', choices=['graph', 'chartqa'],
                        help='Dataset type to train on')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Custom dataset path (overrides default)')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='kd', choices=['image', 'text', 'kd'],
                        help='Training mode: image (student only), text (text only), kd (knowledge distillation)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for checkpoints')
    
    # Hyperparameters (with sensible defaults)
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size per device')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=DEFAULT_LORA_R,
                        help='LoRA rank')
    parser.add_argument('--kd_beta', type=float, default=None,
                        help='KD loss weight (default depends on dataset)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit training samples (for debugging)')
    parser.add_argument('--checkpoint_steps', type=str, default=None,
                        help='Comma-separated list of steps to save checkpoints')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for reproducibility (training data order, dropout, etc.)')
    
    args = parser.parse_args()
    
    # Get FIXED student_init_seed based on dataset (do NOT change these!)
    # graph: 42, chartqa: 67
    args.student_init_seed = get_student_init_seed(args.dataset_type)
    
    # Set random seed
    seed_all(args.seed)
    
    # Determine KD beta based on dataset if not specified
    if args.kd_beta is None:
        args.kd_beta = get_kd_beta(args.dataset_type)
    
    # Parse checkpoint steps
    if args.checkpoint_steps:
        checkpoint_steps = [int(s.strip()) for s in args.checkpoint_steps.split(',')]
    else:
        checkpoint_steps = DEFAULT_CHECKPOINT_STEPS
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(get_output_dir(args.dataset_type, f"{args.mode}_seed{args.seed}"))
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"Training Configuration")
    print(f"=" * 60)
    print(f"Dataset:          {args.dataset_type}")
    print(f"Mode:             {args.mode}")
    print(f"Seed:             {args.seed}")
    print(f"Student Init Seed:{args.student_init_seed}")
    print(f"Learning Rate:    {args.lr}")
    print(f"LoRA Rank:        {args.lora_r}")
    print(f"KD Beta:          {args.kd_beta}")
    print(f"Checkpoints at:   {checkpoint_steps}")
    print(f"Output:           {args.output_dir}")
    print(f"=" * 60)
    
    # Device setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        print(f"Distributed training: rank {local_rank}/{world_size}")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Single GPU training: {torch.cuda.get_device_name(0)}")
        else:
            print("CPU training (this will be slow!)")
    
    # ========================================================================
    # Prepare deterministic student adapter initialization
    # ========================================================================
    # Create a clean RNG snapshot for student adapter initialization
    # This ensures reproducible adapter weights regardless of earlier randomness
    # IMPORTANT: student_init_seed is FIXED per dataset (graph=42, chartqa=67)
    print(f"Creating clean RNG snapshot with student_init_seed={args.student_init_seed}")
    seed_all(args.student_init_seed)
    student_rng_state = capture_rng_state()
    
    # Restore program's training seed for other randomness (data loading, dropout, etc.)
    seed_all(args.seed)
    
    # Load model
    print(f"Loading model: {args.model}")
    
    # Determine dtype - use float32 on CPU for compatibility
    use_cpu = not torch.cuda.is_available()
    model_dtype = torch.float32 if use_cpu else torch.bfloat16
    
    load_common = dict(torch_dtype=model_dtype, trust_remote_code=True)
    
    # Only use device_map="auto" on GPU (causes meta tensor issues on CPU)
    if torch.cuda.is_available() and not is_distributed:
        load_common["device_map"] = "auto"

    try:
        model = AutoModel.from_pretrained(
            args.model, **load_common, attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Flash attention not available, using eager attention: {e}")
        model = AutoModel.from_pretrained(
            args.model, **load_common, attn_implementation="eager"
        )
    
    # Move to CPU explicitly if no GPU
    if use_cpu:
        model = model.to("cpu")
    
    if is_distributed:
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # Set image context token ID
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    
    # Freeze vision tower
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("Frozen: vision_model")
    
    # Setup LoRA targets
    mlp1_targets = []
    for name, module in model.named_modules():
        if "mlp1" in name and hasattr(module, 'weight') and len(module.weight.shape) == 2:
            mlp1_targets.append(name)
    
    llm_layer_targets = []
    if DEFAULT_TRAIN_FIRST_N_LLM_LAYERS > 0 or DEFAULT_TRAIN_LAST_N_LLM_LAYERS > 0:
        layer_indices = set()
        for name, module in model.named_modules():
            if 'language_model' in name and 'layers.' in name:
                parts = name.split('layers.')
                if len(parts) > 1:
                    layer_idx_str = parts[1].split('.')[0]
                    if layer_idx_str.isdigit():
                        layer_indices.add(int(layer_idx_str))
        
        if layer_indices:
            total_layers = max(layer_indices) + 1
            target_layer_indices = set()
            
            if DEFAULT_TRAIN_FIRST_N_LLM_LAYERS > 0:
                first_n = min(DEFAULT_TRAIN_FIRST_N_LLM_LAYERS, total_layers)
                target_layer_indices.update(range(first_n))
                print(f"Training first {first_n} LLM layers")
            
            if DEFAULT_TRAIN_LAST_N_LLM_LAYERS > 0:
                last_n = min(DEFAULT_TRAIN_LAST_N_LLM_LAYERS, total_layers)
                target_layer_indices.update(range(total_layers - last_n, total_layers))
                print(f"Training last {last_n} LLM layers")
            
            for name, module in model.named_modules():
                if 'language_model' in name and 'layers.' in name and hasattr(module, 'weight') and len(module.weight.shape) == 2:
                    parts = name.split('layers.')
                    if len(parts) > 1:
                        layer_idx_str = parts[1].split('.')[0]
                        if layer_idx_str.isdigit() and int(layer_idx_str) in target_layer_indices:
                            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']):
                                llm_layer_targets.append(name)
    
    all_targets = mlp1_targets + llm_layer_targets
    print(f"LoRA targets: {len(all_targets)} modules")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=all_targets,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Create adapters with deterministic initialization
    restore_rng_state(student_rng_state)
    model = get_peft_model(model, lora_config, "student")
    model.add_adapter("teacher", lora_config)
    
    # Copy student weights to teacher
    student_state = model.state_dict()
    teacher_state = {}
    for key, value in student_state.items():
        if '.student.' in key:
            teacher_key = key.replace('.student.', '.teacher.')
            teacher_state[teacher_key] = value.clone()
    model.load_state_dict(teacher_state, strict=False)
    
    # Freeze teacher
    for name, param in model.named_parameters():
        if '.teacher.' in name:
            param.requires_grad = False
    
    model.set_adapter("student")
    model.img_context_token_id = img_context_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model.base_model.model.img_context_token_id = img_context_token_id
    
    if hasattr(model, 'vision_model'):
        model.vision_model.gradient_checkpointing = False
        if hasattr(model.vision_model, 'encoder'):
            model.vision_model.encoder.gradient_checkpointing = False
    
    model.print_trainable_parameters()
    
    # Load dataset
    dataset_config = DATASET_CONFIGS[args.dataset_type]
    dataset_path = args.dataset_path or dataset_config["path"]
    
    if dataset_path is None:
        raise ValueError(f"Dataset path not specified for {args.dataset_type}")
    
    dataset_path = Path(dataset_path)
    
    # Dataset loading depends on dataset type
    if args.dataset_type == 'chartqa':
        # Import and use ChartQA KD dataset
        from chartqa_kd_dataset import create_chartqa_dataset, ChartQAKDDataCollator
        
        train_dataset = create_chartqa_dataset(
            data_path=str(dataset_path),
            split='train',
            include_csv=True,
            include_annotations=True,
            max_samples=args.max_samples,
        )
        eval_dataset = create_chartqa_dataset(
            data_path=str(dataset_path),
            split='validation',
            include_csv=True,
            include_annotations=True,
            max_samples=args.max_samples // 10 if args.max_samples else 100,
        )
        
        # Use KDDataCollator (same format as graph)
        data_collator = KDDataCollator(
            tokenizer=tokenizer,
            max_length=DEFAULT_MAX_LENGTH,
            mode=args.mode,
            num_image_token=model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token,
        )
    else:
        # Default: Graph MCQ Dataset
        if dataset_path.is_dir() and (dataset_path / "train").exists():
            train_dataset = GraphMCQDataset(
                str(dataset_path / "train"),
                tokenizer=tokenizer,
                max_length=DEFAULT_MAX_LENGTH,
                max_samples=args.max_samples,
            )
            eval_dataset = GraphMCQDataset(
                str(dataset_path / "validation"),
                tokenizer=tokenizer,
                max_length=DEFAULT_MAX_LENGTH,
                max_samples=args.max_samples // 10 if args.max_samples else None,
            )
        else:
            train_dataset = GraphMCQDataset(
                str(dataset_path),
                tokenizer=tokenizer,
                max_length=DEFAULT_MAX_LENGTH,
                split="train",
                max_samples=args.max_samples,
            )
            eval_dataset = GraphMCQDataset(
                str(dataset_path),
                tokenizer=tokenizer,
                max_length=DEFAULT_MAX_LENGTH,
                split="validation",
                max_samples=args.max_samples // 10 if args.max_samples else None,
            )
        
        data_collator = KDDataCollator(
            tokenizer=tokenizer,
            max_length=DEFAULT_MAX_LENGTH,
            mode=args.mode,
            num_image_token=model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token,
        )
    
    # Training setup
    grad_accum_steps = max(1, 16 // (args.batch_size * world_size))
    use_cpu = not torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=max(checkpoint_steps) if checkpoint_steps else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="constant" if DEFAULT_CONST_LR else "cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        bf16=not use_cpu, 
        fp16=False,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=checkpoint_steps[0] if checkpoint_steps else 10,
        save_total_limit=None,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,  
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0 if use_cpu else 1,  # No multiprocessing on CPU/Windows
        dataloader_pin_memory=not use_cpu,  # pin_memory only useful with GPU
    )
    
    if args.mode == "kd":
        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            kd_alpha=DEFAULT_KD_ALPHA,
            kd_temperature=DEFAULT_KD_TEMPERATURE,
            kd_beta=args.kd_beta,
            teacher_correct_only=DEFAULT_TEACHER_CORRECT_ONLY,
            use_teacher_ema=DEFAULT_USE_TEACHER_EMA,
            teacher_ema_decay=DEFAULT_TEACHER_EMA_DECAY,
        )
    else:
        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            kd_alpha=1.0,
            kd_temperature=DEFAULT_KD_TEMPERATURE,
            kd_beta=0.0,
            enable_kd=False,
        )
    
    trainer.remove_callback(PrinterCallback)
    
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info for evaluation
    training_info = {
        "base_model_name": args.model,
        "training_mode": args.mode,
        "dataset_type": args.dataset_type,
        "lora_r": args.lora_r,
        "kd_beta": args.kd_beta,
        "seed": args.seed,
        "student_init_seed": args.student_init_seed,
    }
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
