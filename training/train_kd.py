"""
Simple training script for graph VQA with Knowledge Distillation using LoRA.

The KD setup (memory-efficient single model approach):
- Teacher: Same model with LoRA DISABLED → uses frozen pretrained weights
- Student: Same model with LoRA ENABLED → uses trainable adapters
- Loss: alpha * CE_loss + KD_loss (KD coefficient fixed to 1.0)

This approach is memory efficient as we only load ONE model and toggle LoRA on/off.

Trainable components:
- mlp1 (vision-language projector): Via LoRA adapters
- LLM: Via LoRA adapters  
- Vision tower: Frozen

Usage:
    # Single GPU - Image-only training (no KD)
    python train_kd.py --mode image
    
    # Multi-GPU (2 GPUs) - Image-only training
    torchrun --nproc_per_node=2 train_kd.py --mode image
    
    # Text-only training (baseline)
    python train_kd.py --mode text
    
    # Training with Knowledge Distillation (student=image, teacher=text+image)
    python train_kd.py --mode kd --kd_alpha 0.7

sbatch examples:
    # Single GPU
    sbatch --gpus=rtx_3090:1 --mem=32G --time=6:00:00 --wrap="... python train_kd.py --mode image"
    
    # Multi-GPU (2 GPUs)
    sbatch --gpus=rtx_3090:2 --ntasks=2 --mem-per-cpu=8G --time=4:00:00 --wrap="... torchrun --nproc_per_node=2 train_kd.py --mode image"
    
    # Multi-GPU (4 GPUs) 
    sbatch --gpus=rtx_3090:4 --ntasks=4 --mem-per-cpu=8G --time=2:00:00 --wrap="... torchrun --nproc_per_node=4 train_kd.py --mode image"
"""

import argparse
import json
import os
import random
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

# Import OT losses from separate module
from ot_losses import UnbalancedOTLoss, PartialOTLoss, test_unbalanced_ot_loss, test_partial_ot_loss

# Import GUI-360 dataset for KD training
from gui360_kd_dataset import GUI360KDDatasetCompatible, create_gui360_dataset, GUI360KDDataCollator
# Import ChartInsights dataset for KD training
from chartinsights_kd_dataset import create_chartinsights_dataset, ChartInsightsKDDataCollator
from peft import LoraConfig, get_peft_model


# ============================================================================
# RNG State Management for Deterministic Adapter Creation
# ============================================================================

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


# ============================================================================
# Configuration
# ============================================================================

DATASET_PATH = "/cluster/home/rbertolissi/DL-Project/dataset/collections/graph_dataset_4x4"
GUI360_DATASET_PATH = "/cluster/scratch/rbertolissi/datasets/GUI-360"
CHARTINSIGHTS_DATASET_PATH = "/cluster/scratch/rbertolissi/datasets/HKUSTDial ChartInsights master Dataset"
SAVE_DIR = "/cluster/scratch/rbertolissi/graph_vqa_model_new_losses"
DEFAULT_MODEL = "OpenGVLab/InternVL3_5-4B"

# ============================================================================
# InternVL Image Processing (from HuggingFace example)
# ============================================================================

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
    # For training, use max_num=1 to keep it simple (single tile)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ============================================================================
# Dataset
# ============================================================================

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
            """
            Graph benchmark-specific parsing.

            In this dataset, `text_prompt` contains:
              (graph representation text) + "\\n\\n" + (the natural-language question)

            We use the explicit `question` field (when available) to split robustly.
            """
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
        
        # Format options (matching diagram_mcq.py format)
        options = sample.get("options", [])
        options_prompt = "Options:\n"
        for i, opt in enumerate(options):
            options_prompt += f"{chr(65+i)}. {opt}\n"

        answer_idx = sample.get("answer_idx", None)
        if answer_idx is None:
            # Fallback: infer the index from the answer text (dataset provides both in many cases).
            answer_text = sample.get("answer", None)
            if answer_text is not None and answer_text in options:
                answer_idx = options.index(answer_text)
            else:
                answer_idx = 0
        answer = chr(65 + int(answer_idx))
        
        # Build prompt matching diagram_mcq.py format
        question = sample['question']
        student_question_text = f"Question: {question}\n"
        student_question_text += options_prompt
        student_question_text += "Do not explain. Answer with a single uppercase letter (A, B, C, or D) and nothing else.\nAnswer:\n"
        
        # Student prompt (image-based) - image token will be prepended
        student_prompt = f"<image>\n{student_question_text}"
        # Answer will be tokenized separately (just the character)
        student_text = f"{student_prompt}{answer}"
        
        # Teacher prompt (image + graph-text benchmark prompt).
        # `text_prompt` already contains the graph representation and the question (graph first, then question).
        # We still parse out `graph_text` to build a mask for representation alignment.
        teacher_text_prompt = sample.get("text_prompt", "").strip()
        split_prompt = _split_graph_and_question(teacher_text_prompt, question)
        graph_text = split_prompt["graph_text"]

        teacher_context = f"{teacher_text_prompt}\n"
        teacher_context += options_prompt
        teacher_context += "Do not explain. Answer with a single uppercase letter (A, B, C, or D) and nothing else.\nAnswer:\n"

        # Teacher prompt (text + image)
        teacher_prompt = f"<image>\n{teacher_context}"
        # Answer will be tokenized separately (just the character)
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
    
    # InternVL special tokens
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    
    def __init__(self, tokenizer, max_length=1024, mode="kd", num_image_token=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        # InternVL uses 256 tokens per image patch (for 448x448 with patch size 14)
        self.num_image_token = num_image_token
    
    def __call__(self, features):
        batch = {}
        
        # ====================================================================
        # Student inputs (image mode)
        # ====================================================================
        if self.mode in ["image", "kd"]:
            # Process images first to get num_patches
            images = [f["image"] for f in features]
            pixel_values_list = []
            num_patches_list = []
            
            for img in images:
                # Use max_num=1 for single tile
                pv = load_image_for_internvl(img, input_size=448, max_num=1)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])  # number of patches for this image
            
            # Replace <image> with actual image tokens
            # Format: <img><IMG_CONTEXT>...<IMG_CONTEXT></img>
            # Tokenize prompt and answer separately to avoid tokenizer merging issues
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            
            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                # Number of IMG_CONTEXT tokens = num_patches * num_image_token
                image_tokens = (
                    self.IMG_START_TOKEN + 
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + 
                    self.IMG_END_TOKEN
                )
                # Replace <image> placeholder with actual tokens
                prompt = f["student_prompt"].replace("<image>", image_tokens)
                answer = f["answer"]
                
                # Tokenize prompt and answer separately
                prompt_encoding = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length-5
                )
                # Tokenize answer (just the character, no leading space)
                answer_encoding = self.tokenizer(
                    answer, add_special_tokens=False, truncation=False
                )
                
                # Concatenate tokens
                input_ids = prompt_encoding["input_ids"] + answer_encoding["input_ids"]
                attention_mask = prompt_encoding["attention_mask"] + answer_encoding["attention_mask"]
                
                # Create labels: mask prompt, keep answer
                labels = [-100] * len(prompt_encoding["input_ids"]) + answer_encoding["input_ids"]
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)
            
            # Pad sequences to max length in batch
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
            
            # Stack pixel values: [total_patches, 3, 448, 448]
            pixel_values = torch.cat(pixel_values_list, dim=0)
            batch["pixel_values"] = pixel_values
            
            # image_flags: one per patch (not per sample!)
            batch["image_flags"] = torch.ones(pixel_values.shape[0], dtype=torch.long)
        
        # ====================================================================
        # Teacher inputs (text + image mode) - only for KD
        # ====================================================================
        if self.mode == "kd":
            # Teacher gets both image AND text description
            # Process images for teacher (same images as student)
            teacher_texts = []
            teacher_prompts = []
            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                image_tokens = (
                    self.IMG_START_TOKEN + 
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + 
                    self.IMG_END_TOKEN
                )
                # Replace <image> placeholder with actual tokens for teacher
                text = f["teacher_text"].replace("<image>", image_tokens)
                prompt = f["teacher_prompt"].replace("<image>", image_tokens)
                teacher_texts.append(text)
                teacher_prompts.append(prompt)
            
            teacher_encodings = self.tokenizer(
                teacher_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            
            batch["teacher_input_ids"] = teacher_encodings["input_ids"]
            batch["teacher_attention_mask"] = teacher_encodings["attention_mask"]

            # Mask selecting only the graph representation tokens inside the teacher prompt.
            # This is used by the trainer to compute rep-loss only on the graph representation
            # (not on question/options/instruction tokens).
            teacher_graph_mask = torch.zeros_like(batch["teacher_input_ids"], dtype=torch.bool)

            def _common_prefix_len(a: List[int], b: List[int]) -> int:
                n = min(len(a), len(b))
                for j in range(n):
                    if a[j] != b[j]:
                        return j
                return n

            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                image_tokens = (
                    self.IMG_START_TOKEN +
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) +
                    self.IMG_END_TOKEN
                )
                graph_text = f.get("graph_text", "")
                if not isinstance(graph_text, str) or not graph_text.strip():
                    continue
                # Graph benchmark-specific span:
                # teacher prompt starts with "<image>\\n" followed by `text_prompt`,
                # and `graph_text` is the prefix of `text_prompt` (before the final "\\n\\n{question}").
                prefix = f"{image_tokens}\n"
                prefix_ids = self.tokenizer(
                    prefix, add_special_tokens=True, truncation=True, max_length=self.max_length
                )["input_ids"]
                prefix_plus_graph_ids = self.tokenizer(
                    prefix + graph_text, add_special_tokens=True, truncation=True, max_length=self.max_length
                )["input_ids"]
                start = _common_prefix_len(prefix_ids, prefix_plus_graph_ids)

                # Compute end boundary robustly (tokenization is not perfectly additive for byte-level BPE).
                graph_plus_delim_ids = self.tokenizer(
                    prefix + graph_text + "\n\n", add_special_tokens=True, truncation=True, max_length=self.max_length
                )["input_ids"]
                end = _common_prefix_len(prefix_plus_graph_ids, graph_plus_delim_ids)
                if end <= start:
                    continue
                start = min(start, teacher_graph_mask.shape[1])
                end = min(end, teacher_graph_mask.shape[1])
                if end > start:
                    teacher_graph_mask[i, start:end] = True
            teacher_graph_mask = teacher_graph_mask & batch["teacher_attention_mask"].bool()
            batch["teacher_graph_mask"] = teacher_graph_mask
            
            # Teacher uses the same pixel_values as student
            batch["teacher_pixel_values"] = pixel_values.clone()
            batch["teacher_image_flags"] = torch.ones(pixel_values.shape[0], dtype=torch.long)
        
        # ====================================================================
        # Text-only mode
        # ====================================================================
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
                # Use truncation here as well for the same reason as above
                prompt_tokens = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length
                )["input_ids"]
                labels[i, :min(len(prompt_tokens), labels.shape[1])] = -100
            batch["labels"] = labels
        
        return batch


# ============================================================================
# KD Trainer
# ============================================================================

class KDTrainer(Trainer):
    """
    Trainer with Knowledge Distillation support.
    
    Uses a single model with LoRA adapters:
    - Student forward: LoRA ENABLED (trainable adapters active)
    - Teacher forward: LoRA DISABLED (frozen pretrained weights)
    
    This is memory efficient as we only load one model.
    """
    
    def __init__(
        self,
        *args,
        kd_alpha=0.5,
        kd_temperature=4.0,
        kd_beta: float = 1.0,
        kd_gamma: float = 0.0,
        kd_gamma_hidden: float = 0.0,
        rep_loss_type: str = "none",
        enable_kd: bool = True,
        target_hidden_layer: int = 0,
        num_hidden_layers_align: int = 4,
        teacher_correct_only: bool = False,
        student_incorrect_only: bool = False,
        use_teacher_ema: bool = False,
        teacher_ema_decay: float = 0.999,
        # UOT (Unbalanced OT) loss hyperparameters
        uot_weight: float = 0.0,
        uot_eps: float = 0.1,
        uot_iters: int = 50,
        uot_tau_a: float = 1.0,
        uot_tau_b: float = 1.0,
        uot_lam_d: float = 1.0,
        uot_lam_c: float = 0.0,
        uot_cost_weight: float = 1.0,
        uot_reg_weight: float = 1.0,
        uot_use_teacher_attention: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.use_kd = enable_kd
        self.kd_beta = kd_beta
        self.kd_gamma = kd_gamma
        self.kd_gamma_hidden = kd_gamma_hidden
        self.rep_loss_type = rep_loss_type
        self.target_hidden_layer = target_hidden_layer
        self.num_hidden_layers_align = num_hidden_layers_align
        self.teacher_correct_only = teacher_correct_only
        self.student_incorrect_only = student_incorrect_only
        self.use_teacher_ema = use_teacher_ema
        self.teacher_ema_decay = teacher_ema_decay
        self._attn_impl_configured = False
        
        # UOT loss setup
        self.uot_weight = uot_weight
        self.uot_cost_weight = uot_cost_weight
        self.uot_reg_weight = uot_reg_weight
        self.uot_use_teacher_attention = uot_use_teacher_attention
        if uot_weight > 0:
            self.uot_loss_fn = UnbalancedOTLoss(
                eps=uot_eps,
                iters=uot_iters,
                tau_a=uot_tau_a,
                tau_b=uot_tau_b,
                lam_d=uot_lam_d,
                lam_c=uot_lam_c,
            )
        else:
            self.uot_loss_fn = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the PEFT model for adapter control
        peft_model = model.module if hasattr(model, 'module') else model  # Handle DDP
        
        # For InternVL with PEFT, we need to call the base model directly
        # because InternVL doesn't accept inputs_embeds which PEFT tries to pass
        base_model = peft_model
        if hasattr(base_model, 'base_model'):
            # PEFT model - get the actual InternVL model
            base_model = base_model.base_model.model
        
        # Get model dtype for pixel values conversion
        model_dtype = next(base_model.parameters()).dtype
        
        need_rep_loss = (
            (
                (self.kd_gamma > 0 and self.rep_loss_type in {"mu_sigma", "cka", "attn"})
                or (self.kd_gamma_hidden > 0)
            )
            and "teacher_input_ids" in inputs
        )
        need_uot_loss = (
            self.uot_weight > 0
            and self.uot_loss_fn is not None
            and "teacher_input_ids" in inputs
        )
        need_hidden_states = need_rep_loss or need_uot_loss
        need_attentions = need_rep_loss and self.rep_loss_type == "attn"
        if need_attentions:
            self._ensure_eager_attention(base_model)

        # ================================================================
        # Student forward pass (with LoRA ENABLED)
        # ================================================================
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "pixel_values" in inputs:
            # Convert pixel_values to model dtype (bfloat16)
            student_inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
        if "image_flags" in inputs:
            student_inputs["image_flags"] = inputs["image_flags"]
        if need_hidden_states:
            student_inputs["output_hidden_states"] = True
        if need_attentions:
            student_inputs["output_attentions"] = True
        
        labels = inputs["labels"]
        
        # LoRA is enabled by default
        outputs = base_model(**student_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        # For representation alignment we want the *input* embeddings (pre-LLM), not the last layer.
        # For hidden_state loss, we need all hidden states to access the last k layers
        student_hidden = outputs.hidden_states[self.target_hidden_layer] if need_hidden_states and outputs.hidden_states is not None else None
        student_all_hidden_states = outputs.hidden_states if (need_hidden_states and self.kd_gamma_hidden > 0) else None
        student_attentions = outputs.attentions if need_attentions else None
        
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
        
        # ================================================================
        # KD Loss (Teacher forward with LoRA DISABLED)
        # ================================================================
        rep_loss = None
        use_teacher = ("teacher_input_ids" in inputs) and (self.use_kd or need_hidden_states)
        if use_teacher:
            with torch.no_grad():
                batch_size = inputs["teacher_input_ids"].shape[0]
                
                # Teacher forward: Use EMA adapter or disable LoRA
                if self.use_teacher_ema:
                    # Switch to teacher adapter (EMA weights)
                    peft_model.set_adapter("teacher")
                else:
                    # DISABLE LoRA adapters for teacher forward pass
                    # This makes the model use frozen pretrained weights
                    peft_model.disable_adapter_layers()
                
                try:
                    teacher_outputs = base_model(
                        input_ids=inputs["teacher_input_ids"],
                        attention_mask=inputs["teacher_attention_mask"],
                        pixel_values=inputs["teacher_pixel_values"].to(model_dtype),
                        image_flags=inputs["teacher_image_flags"],
                        output_hidden_states=need_hidden_states,
                        output_attentions=need_attentions,
                    )
                    teacher_logits = teacher_outputs.logits if hasattr(teacher_outputs, "logits") else teacher_outputs[0]
                    # For representation alignment we want the *input* embeddings (pre-LLM), not the last layer.
                    teacher_hidden = teacher_outputs.hidden_states[self.target_hidden_layer] if need_hidden_states and teacher_outputs.hidden_states is not None else None
                    # For hidden_state loss, we need all hidden states to access the last k layers
                    teacher_all_hidden_states = teacher_outputs.hidden_states if (need_hidden_states and self.kd_gamma_hidden > 0) else None
                    teacher_attentions = teacher_outputs.attentions if need_attentions else None
                finally:
                    # Switch back to student adapter or RE-ENABLE LoRA
                    peft_model.set_adapter("student")
            
            # For KD in MCQ tasks, we only care about the answer token prediction.
            # which is the last non-padded position in the input.
            
            kd_loss = ce_loss.new_zeros(())
            kd_prob_l2 = ce_loss.new_zeros(())
            # Track teacher correctness per sample for teacher_correct_only mode
            teacher_correct_mask = []
            # Track student correctness per sample for student_incorrect_only mode
            student_correct_mask = []
            # Track answer positions for hidden_state loss
            student_answer_positions = []
            teacher_answer_positions = []
            
            if self.use_kd:
                T = self.kd_temperature
                kd_loss_accum = 0.0
                num_valid = 0
                prob_l2_accum = 0.0
                
                for i in range(batch_size):
                    # Find the position of the answer token in student (last non -100 label)
                    student_valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(student_valid_positions) == 0:
                        student_answer_positions.append(-1)
                        teacher_answer_positions.append(-1)
                        teacher_correct_mask.append(False)
                        continue
                    
                    # The answer token position - we want the logits that predict this token
                    # In causal LM, position t predicts token t+1, so we need position (answer_pos - 1)
                    student_answer_pos = student_valid_positions[0].item()  # First valid = answer position
                    student_pred_pos = student_answer_pos - 1  # Position that predicts the answer
                    
                    if student_pred_pos < 0:
                        student_answer_positions.append(-1)
                        teacher_answer_positions.append(-1)
                        teacher_correct_mask.append(False)
                        continue
                    
                    # For teacher, find the last token position (before padding)
                    teacher_seq_len = inputs["teacher_attention_mask"][i].sum().item()
                    teacher_pred_pos = teacher_seq_len - 2  # -1 for 0-indexing, -1 more for prediction position
                    
                    if teacher_pred_pos < 0:
                        student_answer_positions.append(-1)
                        teacher_answer_positions.append(-1)
                        teacher_correct_mask.append(False)
                        continue
                    
                    # Store answer positions for hidden_state loss
                    student_answer_positions.append(student_pred_pos)
                    teacher_answer_positions.append(teacher_pred_pos)
                    
                    # Extract logits at the answer prediction positions
                    student_logits_i = logits[i, student_pred_pos, :]  # [vocab_size]
                    teacher_logits_i = teacher_logits[i, teacher_pred_pos, :]  # [vocab_size]
                    
                    # Get the ground truth answer token
                    gt_answer_token = labels[i, student_answer_pos].item()
                    
                    # Check if student prediction is correct (for student_incorrect_only mode)
                    student_pred_token = student_logits_i.argmax().item()
                    is_student_correct = (student_pred_token == gt_answer_token)
                    student_correct_mask.append(is_student_correct)
                    
                    # Check if teacher prediction is correct (for teacher_correct_only mode)
                    teacher_pred_token = teacher_logits_i.argmax().item()
                    is_teacher_correct = (teacher_pred_token == gt_answer_token)
                    teacher_correct_mask.append(is_teacher_correct)
                    
                    # If teacher_correct_only is enabled and teacher is wrong, skip KL-div for this sample
                    if self.teacher_correct_only and not is_teacher_correct:
                        continue
                    
                    # If student_incorrect_only is enabled and student is correct, skip KL-div for this sample
                    if self.student_incorrect_only and is_student_correct:
                        continue
                    
                    # Compute KL divergence for this sample
                    student_soft = F.log_softmax(student_logits_i / T, dim=-1)
                    teacher_soft = F.softmax(teacher_logits_i / T, dim=-1)
                    
                    kd_loss_i = F.kl_div(student_soft, teacher_soft, reduction="sum")
                    kd_loss_accum = kd_loss_accum + kd_loss_i
                    student_probs = student_soft.exp()
                    prob_l2_accum = prob_l2_accum + torch.norm(student_probs - teacher_soft, p=2)
                    num_valid += 1
                
                if num_valid > 0:
                    kd_loss = (kd_loss_accum / num_valid) * (T ** 2)
                    kd_prob_l2 = prob_l2_accum / num_valid
            else:
                # Fill in positions even when not using KD (for potential rep loss)
                for i in range(batch_size):
                    student_valid_positions = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if len(student_valid_positions) == 0:
                        student_answer_positions.append(-1)
                        teacher_answer_positions.append(-1)
                        teacher_correct_mask.append(False)
                        continue
                    student_answer_pos = student_valid_positions[0].item()
                    student_pred_pos = student_answer_pos - 1
                    teacher_seq_len = inputs["teacher_attention_mask"][i].sum().item()
                    teacher_pred_pos = teacher_seq_len - 2
                    student_answer_positions.append(student_pred_pos if student_pred_pos >= 0 else -1)
                    teacher_answer_positions.append(teacher_pred_pos if teacher_pred_pos >= 0 else -1)
                    # Check student and teacher correctness
                    if student_pred_pos >= 0 and teacher_pred_pos >= 0:
                        gt_answer_token = labels[i, student_answer_pos].item()
                        student_pred_token = logits[i, student_pred_pos, :].argmax().item()
                        student_correct_mask.append(student_pred_token == gt_answer_token)
                        teacher_pred_token = teacher_logits[i, teacher_pred_pos, :].argmax().item()
                        teacher_correct_mask.append(teacher_pred_token == gt_answer_token)
                    else:
                        student_correct_mask.append(False)
                        teacher_correct_mask.append(False)
        else:
            kd_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
            kd_prob_l2 = torch.tensor(0.0, device=labels.device)
            teacher_correct_mask = []
            student_correct_mask = []
            student_answer_positions = []
            teacher_answer_positions = []
        
        if self.use_kd:
            # KD coefficient fixed to 1.0, kd_alpha scales CE term, kd_beta scales KL_div
            base_loss = self.kd_alpha * ce_loss + self.kd_beta * kd_loss
        else:
            base_loss = ce_loss
        
        # ================================================================
        # Representation Alignment Losses
        # ================================================================
        rep_loss = None  # For CKA, attn, moment losses (weighted by gamma)
        rep_loss_hidden = None  # For hidden_state loss (weighted by gamma_hidden)
        
        # Compute hidden_state loss if gamma_hidden > 0 (independent of rep_loss_type)
        if need_rep_loss and use_teacher and self.kd_gamma_hidden > 0:
            # Use the new hidden_state_cosine_loss for last k layers alignment
            # Filter positions based on teacher_correct_only and student_incorrect_only if enabled
            if (self.teacher_correct_only or self.student_incorrect_only) and teacher_correct_mask and student_correct_mask:
                filtered_student_pos = [
                    pos if (
                        (not self.teacher_correct_only or teacher_correct_mask[idx]) and
                        (not self.student_incorrect_only or not student_correct_mask[idx])
                    ) else -1
                    for idx, pos in enumerate(student_answer_positions)
                ]
                filtered_teacher_pos = [
                    pos if (
                        (not self.teacher_correct_only or teacher_correct_mask[idx]) and
                        (not self.student_incorrect_only or not student_correct_mask[idx])
                    ) else -1
                    for idx, pos in enumerate(teacher_answer_positions)
                ]
            else:
                filtered_student_pos = student_answer_positions
                filtered_teacher_pos = teacher_answer_positions
            
            rep_loss_hidden = self._hidden_state_cosine_loss(
                student_hidden_states=student_all_hidden_states,
                teacher_hidden_states=teacher_all_hidden_states if 'teacher_all_hidden_states' in locals() else None,
                student_answer_positions=filtered_student_pos,
                teacher_answer_positions=filtered_teacher_pos,
                num_layers=self.num_hidden_layers_align,
            )
        
        # Compute other rep losses if gamma > 0
        if need_rep_loss and use_teacher and self.kd_gamma > 0 and self.rep_loss_type in ["cka", "attn", "mu_sigma"]:
            if student_hidden is not None:
                img_token_id = getattr(base_model, "img_context_token_id", None)
                # Fallback to PEFT wrapper attribute if needed
                if img_token_id is None:
                    img_token_id = getattr(peft_model, "img_context_token_id", None)
                
                # For other rep loss types with teacher_correct_only, we need to handle per-sample
                # For now, these methods use masked tokens, so teacher_correct_only mainly affects hidden_state loss
                rep_loss = self._compute_representation_loss(
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden if 'teacher_hidden' in locals() else None,
                    student_ids=inputs["input_ids"],
                    teacher_ids=inputs.get("teacher_input_ids"),
                    student_attention_mask=inputs["attention_mask"],
                    teacher_attention_mask=inputs.get("teacher_attention_mask"),
                    teacher_subset_mask=inputs.get("teacher_graph_mask"),
                    img_context_token_id=img_token_id,
                    student_attentions=student_attentions,
                    teacher_attentions=teacher_attentions if 'teacher_attentions' in locals() else None,
                    teacher_correct_mask=teacher_correct_mask if self.teacher_correct_only else None,
                )
        
        # Default to zero if not computed
        if rep_loss is None:
            rep_loss = base_loss.new_zeros(())
        if rep_loss_hidden is None:
            rep_loss_hidden = base_loss.new_zeros(())
        
        # ================================================================
        # UOT Loss (Unbalanced Optimal Transport for representation alignment)
        # ================================================================
        uot_loss_cost = base_loss.new_zeros(())
        uot_loss_reg = base_loss.new_zeros(())
        
        if self.uot_weight > 0 and self.uot_loss_fn is not None and use_teacher:
            img_token_id = getattr(base_model, "img_context_token_id", None)
            if img_token_id is None:
                img_token_id = getattr(peft_model, "img_context_token_id", None)
            
            if student_hidden is not None and 'teacher_hidden' in locals() and teacher_hidden is not None:
                uot_result = self._compute_uot_loss(
                    student_hidden=student_hidden,
                    teacher_hidden=teacher_hidden,
                    student_ids=inputs["input_ids"],
                    teacher_ids=inputs.get("teacher_input_ids"),
                    student_attention_mask=inputs["attention_mask"],
                    teacher_attention_mask=inputs.get("teacher_attention_mask"),
                    teacher_subset_mask=inputs.get("teacher_graph_mask"),
                    img_context_token_id=img_token_id,
                    teacher_attentions=teacher_attentions if 'teacher_attentions' in locals() else None,
                )
                if uot_result is not None:
                    uot_loss_cost = uot_result["loss_uot_cost"]
                    uot_loss_reg = uot_result["loss_uot_reg"]
        
        # Combine UOT losses
        uot_loss_total = self.uot_cost_weight * uot_loss_cost + self.uot_reg_weight * uot_loss_reg
        
        # Combine all losses: base + gamma*rep + gamma_hidden*rep_hidden + uot
        loss = base_loss + self.kd_gamma * rep_loss + self.kd_gamma_hidden * rep_loss_hidden + self.uot_weight * uot_loss_total
        
        # Log losses occasionally
        should_log = self.state.global_step > 0 and self.state.global_step % 10 == 0
        if should_log:
            log_payload = {"ce_loss": ce_loss.item()}
            if kd_loss is not None:
                log_payload["kd_loss"] = kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss
            if 'kd_prob_l2' in locals() and kd_prob_l2 is not None:
                log_payload["kd_prob_l2"] = kd_prob_l2.item() if isinstance(kd_prob_l2, torch.Tensor) else kd_prob_l2
            if rep_loss is not None and rep_loss.item() > 0:
                log_payload["rep_loss"] = rep_loss.item() if isinstance(rep_loss, torch.Tensor) else rep_loss
            if rep_loss_hidden is not None and rep_loss_hidden.item() > 0:
                log_payload["rep_loss_hidden"] = rep_loss_hidden.item() if isinstance(rep_loss_hidden, torch.Tensor) else rep_loss_hidden
            # UOT loss logging
            if self.uot_weight > 0:
                log_payload["uot_loss_cost"] = uot_loss_cost.item() if isinstance(uot_loss_cost, torch.Tensor) else uot_loss_cost
                log_payload["uot_loss_reg"] = uot_loss_reg.item() if isinstance(uot_loss_reg, torch.Tensor) else uot_loss_reg
            # Teacher correctness stats (for teacher_correct_only mode)
            if self.teacher_correct_only and teacher_correct_mask:
                num_correct = sum(teacher_correct_mask)
                num_total = len(teacher_correct_mask)
                log_payload["teacher_correct_ratio"] = num_correct / max(num_total, 1)
            # Student correctness stats (for student_incorrect_only mode)
            if self.student_incorrect_only and student_correct_mask:
                num_incorrect = sum(1 for sc in student_correct_mask if not sc)
                num_total = len(student_correct_mask)
                log_payload["student_incorrect_ratio"] = num_incorrect / max(num_total, 1)
            self.log(log_payload)
            teacher_correct_str = ""
            if self.teacher_correct_only and teacher_correct_mask:
                teacher_correct_str = f" teacher_correct={log_payload.get('teacher_correct_ratio', -1):.2f}"
            student_incorrect_str = ""
            if self.student_incorrect_only and student_correct_mask:
                student_incorrect_str = f" student_incorrect={log_payload.get('student_incorrect_ratio', -1):.2f}"
            rep_hidden_str = f" rep_hidden={log_payload.get('rep_loss_hidden', -1):.6f}" if 'rep_loss_hidden' in log_payload else ""
            print(
                f"[step {self.state.global_step}] "
                f"ce_loss={log_payload.get('ce_loss', -1):.6f} "
                f"kd_loss={log_payload.get('kd_loss', -1):.6f} "
                f"rep_loss={log_payload.get('rep_loss', -1):.6f}" +
                rep_hidden_str +
                f" kd_prob_l2={log_payload.get('kd_prob_l2', -1):.6f}" +
                (f" uot_cost={log_payload.get('uot_loss_cost', -1):.6f} uot_reg={log_payload.get('uot_loss_reg', -1):.6f}" if self.uot_weight > 0 else "") +
                teacher_correct_str +
                student_incorrect_str,
                flush=True,
            )
        
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _moment_cov_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.numel() == 0 or Y.numel() == 0:
            return X.new_zeros(())
        mu_x = X.mean(dim=0)
        mu_y = Y.mean(dim=0)
        mean_diff = (mu_x - mu_y).pow(2).sum()
        
        xc = X - mu_x
        yc = Y - mu_y
        denom_x = float(max(X.shape[0] - 1, 1))
        denom_y = float(max(Y.shape[0] - 1, 1))
        cov_x = xc.t().matmul(xc) / denom_x
        cov_y = yc.t().matmul(yc) / denom_y
        cov_diff = (cov_x - cov_y).pow(2).sum()
        return mean_diff + cov_diff

    @staticmethod
    def _cka_loss(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if X.numel() == 0 or Y.numel() == 0:
            return X.new_zeros(())
        Xc = X - X.mean(dim=0, keepdim=True)
        Yc = Y - Y.mean(dim=0, keepdim=True)
        xy = Xc.matmul(Yc.t())
        numerator = (xy.pow(2)).sum()
        xx = Xc.matmul(Xc.t())
        yy = Yc.matmul(Yc.t())
        denom = torch.sqrt((xx.pow(2).sum() * yy.pow(2).sum()).clamp(min=eps))
        return 1 - numerator / denom.clamp(min=eps)

    @staticmethod
    def _hidden_state_cosine_loss(
        student_hidden_states: Tuple[torch.Tensor, ...],
        teacher_hidden_states: Tuple[torch.Tensor, ...],
        student_answer_positions: List[int],
        teacher_answer_positions: List[int],
        num_layers: int,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Compute hidden state alignment loss using cosine similarity.
        
        Aligns the last `num_layers` hidden states at the answer prediction position
        between student and teacher, using cosine similarity pooled with mean.
        
        Args:
            student_hidden_states: Tuple of student hidden states from all layers (layer, batch, seq, dim)
            teacher_hidden_states: Tuple of teacher hidden states from all layers (layer, batch, seq, dim)
            student_answer_positions: List of answer prediction positions for each sample in student
            teacher_answer_positions: List of answer prediction positions for each sample in teacher
            num_layers: Number of last hidden layers to align
            eps: Small constant for numerical stability
        
        Returns:
            Mean cosine distance (1 - cosine_similarity) across layers and samples
        """
        if student_hidden_states is None or teacher_hidden_states is None:
            return student_hidden_states[0].new_zeros(()) if student_hidden_states else torch.tensor(0.0)
        
        batch_size = len(student_answer_positions)
        num_student_layers = len(student_hidden_states)
        num_teacher_layers = len(teacher_hidden_states)
        
        # Use the last num_layers from both student and teacher
        # Make sure we don't exceed available layers
        k = min(num_layers, num_student_layers, num_teacher_layers)
        if k <= 0:
            return student_hidden_states[0].new_zeros(())
        
        cosine_losses = []
        
        for layer_idx in range(k):
            # Index from the end: -1 is last layer, -2 is second-to-last, etc.
            student_layer_idx = -(layer_idx + 1)
            teacher_layer_idx = -(layer_idx + 1)
            
            student_layer_hidden = student_hidden_states[student_layer_idx]  # (batch, seq, dim)
            teacher_layer_hidden = teacher_hidden_states[teacher_layer_idx]  # (batch, seq, dim)
            
            for i in range(batch_size):
                student_pos = student_answer_positions[i]
                teacher_pos = teacher_answer_positions[i]
                
                if student_pos < 0 or teacher_pos < 0:
                    continue
                if student_pos >= student_layer_hidden.shape[1] or teacher_pos >= teacher_layer_hidden.shape[1]:
                    continue
                
                # Extract hidden states at answer prediction position
                student_h = student_layer_hidden[i, student_pos, :]  # (dim,)
                teacher_h = teacher_layer_hidden[i, teacher_pos, :]  # (dim,)
                
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(student_h.unsqueeze(0), teacher_h.unsqueeze(0), dim=1, eps=eps)
                # Convert to loss: 1 - cos_sim (so perfect alignment = 0 loss)
                cosine_losses.append(1 - cos_sim.squeeze())
        
        if len(cosine_losses) == 0:
            return student_hidden_states[0].new_zeros(())
        
        return torch.stack(cosine_losses).mean()

    @staticmethod
    def _attention_pool(
        embeddings: torch.Tensor,
        attn_matrix: torch.Tensor,
        subset_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> Optional[torch.Tensor]:
        if attn_matrix is None or subset_mask.sum() == 0:
            return None
        # attn_matrix: [num_heads, seq, seq]
        attn = attn_matrix.mean(dim=0)  # [seq, seq]
        key_mask = attention_mask.float()
        query_mask = attention_mask.float()
        attn = attn * query_mask.unsqueeze(-1)
        denom_queries = query_mask.sum().clamp(min=1.0)
        weights_per_key = attn.sum(dim=0) / denom_queries
        weights_per_key = weights_per_key * key_mask
        subset_weights = weights_per_key * subset_mask.float()
        weight_sum = subset_weights.sum().clamp(min=eps)
        return (subset_weights.unsqueeze(-1) * embeddings).sum(dim=0) / weight_sum

    def _attention_alignment_loss(
        self,
        student_emb: torch.Tensor,
        teacher_emb: torch.Tensor,
        student_attn: Optional[torch.Tensor],
        teacher_attn: Optional[torch.Tensor],
        student_subset_mask: torch.Tensor,
        teacher_subset_mask: torch.Tensor,
        student_attn_mask: torch.Tensor,
        teacher_attn_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if student_attn is None or teacher_attn is None:
            return None
        pooled_student = self._attention_pool(student_emb, student_attn, student_subset_mask, student_attn_mask)
        pooled_teacher = self._attention_pool(teacher_emb, teacher_attn, teacher_subset_mask, teacher_attn_mask)
        if pooled_student is None or pooled_teacher is None:
            return None
        return (pooled_student - pooled_teacher).pow(2).sum()

    def _compute_representation_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: Optional[torch.Tensor],
        student_ids: torch.Tensor,
        teacher_ids: Optional[torch.Tensor],
        student_attention_mask: torch.Tensor,
        teacher_attention_mask: Optional[torch.Tensor],
        teacher_subset_mask: Optional[torch.Tensor],
        img_context_token_id: Optional[int],
        student_attentions: Optional[Any] = None,
        teacher_attentions: Optional[Any] = None,
        teacher_correct_mask: Optional[List[bool]] = None,
    ) -> Optional[torch.Tensor]:
        if teacher_hidden is None or img_context_token_id is None:
            return None
        batch_size = student_hidden.shape[0]
        losses: List[torch.Tensor] = []
        for i in range(batch_size):
            # Skip samples where teacher is incorrect (if teacher_correct_only is enabled)
            if teacher_correct_mask is not None and i < len(teacher_correct_mask) and not teacher_correct_mask[i]:
                continue
            student_valid_mask = student_attention_mask[i].bool()
            vision_mask = (student_ids[i] == img_context_token_id) & student_valid_mask
            if teacher_ids is None or teacher_attention_mask is None:
                continue
            teacher_valid_mask = teacher_attention_mask[i].bool()
            if teacher_subset_mask is not None:
                # Prefer an explicit subset mask provided by the data collator (e.g., graph tokens only).
                text_mask = teacher_subset_mask[i].bool() & teacher_valid_mask
            else:
                # Fallback: all non-image tokens.
                text_mask = (teacher_ids[i] != img_context_token_id) & teacher_valid_mask
            if vision_mask.sum() == 0 or text_mask.sum() == 0:
                continue
            X = student_hidden[i][vision_mask]
            Y = teacher_hidden[i][text_mask]
            if self.rep_loss_type == "mu_sigma":
                loss_i = self._moment_cov_loss(X, Y)
            elif self.rep_loss_type == "cka":
                loss_i = self._cka_loss(X, Y)
            elif self.rep_loss_type == "attn":
                student_attn_last = student_attentions[self.target_hidden_layer][i] if student_attentions else None
                teacher_attn_last = teacher_attentions[self.target_hidden_layer][i] if teacher_attentions else None
                loss_i = self._attention_alignment_loss(
                    student_emb=student_hidden[i],
                    teacher_emb=teacher_hidden[i],
                    student_attn=student_attn_last,
                    teacher_attn=teacher_attn_last,
                    student_subset_mask=vision_mask,
                    teacher_subset_mask=text_mask,
                    student_attn_mask=student_valid_mask,
                    teacher_attn_mask=teacher_valid_mask,
                )
            else:
                loss_i = None
            if loss_i is not None:
                losses.append(loss_i)
        if len(losses) == 0:
            return None
        return torch.stack(losses).mean()

    def _ensure_eager_attention(self, base_model: torch.nn.Module) -> None:
        if self._attn_impl_configured:
            return
        set_impl = getattr(base_model, "set_attn_implementation", None)
        if callable(set_impl):
            try:
                set_impl("eager")
                self._attn_impl_configured = True
                return
            except Exception:
                pass
        config = getattr(base_model, "config", None)
        if config is not None and hasattr(config, "attn_implementation"):
            config.attn_implementation = "eager"
            self._attn_impl_configured = True

    def _compute_uot_loss(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        student_ids: torch.Tensor,
        teacher_ids: Optional[torch.Tensor],
        student_attention_mask: torch.Tensor,
        teacher_attention_mask: Optional[torch.Tensor],
        teacher_subset_mask: Optional[torch.Tensor],
        img_context_token_id: Optional[int],
        teacher_attentions: Optional[Any] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Compute Unbalanced Optimal Transport loss between student vision tokens
        and teacher text tokens.
        
        Args:
            student_hidden: Student hidden states (B, seq_len, D)
            teacher_hidden: Teacher hidden states (B, seq_len, D)
            student_ids: Student input IDs (B, seq_len)
            teacher_ids: Teacher input IDs (B, seq_len)
            student_attention_mask: Student attention mask (B, seq_len)
            teacher_attention_mask: Teacher attention mask (B, seq_len)
            teacher_subset_mask: Optional mask for teacher subset (e.g., graph tokens)
            img_context_token_id: ID of the image context token
            teacher_attentions: Optional teacher attention weights for token weighting
        
        Returns:
            Dictionary with loss_uot_cost and loss_uot_reg, or None if not computable
        """
        if self.uot_loss_fn is None or img_context_token_id is None:
            return None
        if teacher_hidden is None or teacher_ids is None or teacher_attention_mask is None:
            return None
        
        batch_size = student_hidden.shape[0]
        device = student_hidden.device
        
        # Collect vision tokens (X) and text tokens (Y) per batch
        X_list = []
        Y_list = []
        Y_weights_list = []
        valid_batch_indices = []
        
        for i in range(batch_size):
            # Student: extract vision tokens (IMG_CONTEXT tokens)
            student_valid_mask = student_attention_mask[i].bool()
            vision_mask = (student_ids[i] == img_context_token_id) & student_valid_mask
            
            # Teacher: extract text tokens (using subset mask if provided)
            teacher_valid_mask = teacher_attention_mask[i].bool()
            if teacher_subset_mask is not None:
                text_mask = teacher_subset_mask[i].bool() & teacher_valid_mask
            else:
                # Fallback: all non-image tokens
                text_mask = (teacher_ids[i] != img_context_token_id) & teacher_valid_mask
            
            if vision_mask.sum() == 0 or text_mask.sum() == 0:
                continue
            
            X_i = student_hidden[i][vision_mask]  # (V_i, D)
            Y_i = teacher_hidden[i][text_mask]    # (T_i, D)
            
            # Compute teacher token weights from attention if enabled
            if self.uot_use_teacher_attention and teacher_attentions is not None:
                # Use attention from target_hidden_layer
                # teacher_attentions is a tuple of (B, num_heads, seq_len, seq_len) tensors
                teacher_attn_layer = teacher_attentions[self.target_hidden_layer][i]  # (num_heads, seq_len, seq_len)
                # Average attention across heads: (seq_len, seq_len)
                attn_avg = teacher_attn_layer.mean(dim=0)
                # For each token, compute its importance as the sum of attention it receives
                # (average attention paid TO this token from all other tokens)
                token_importance = attn_avg.sum(dim=0)  # (seq_len,)
                # Extract importance for text tokens only
                Y_importance = token_importance[text_mask]  # (T_i,)
                Y_weights_list.append(Y_importance)
            else:
                Y_weights_list.append(None)
            
            X_list.append(X_i)
            Y_list.append(Y_i)
            valid_batch_indices.append(i)
        
        if len(X_list) == 0:
            return None
        
        # For efficiency, we need to batch the UOT computation
        # Since V and T can vary per sample, we'll pad to max sizes
        V_max = max(x.shape[0] for x in X_list)
        T_max = max(y.shape[0] for y in Y_list)
        D = X_list[0].shape[-1]
        B_valid = len(X_list)
        
        # Pad tensors
        X_padded = torch.zeros(B_valid, V_max, D, device=device, dtype=student_hidden.dtype)
        Y_padded = torch.zeros(B_valid, T_max, D, device=device, dtype=student_hidden.dtype)
        
        # Create masks for actual tokens (not padding)
        X_mask = torch.zeros(B_valid, V_max, device=device, dtype=torch.bool)
        Y_mask = torch.zeros(B_valid, T_max, device=device, dtype=torch.bool)
        
        for idx, (X_i, Y_i) in enumerate(zip(X_list, Y_list)):
            V_i, T_i = X_i.shape[0], Y_i.shape[0]
            X_padded[idx, :V_i] = X_i
            Y_padded[idx, :T_i] = Y_i
            X_mask[idx, :V_i] = True
            Y_mask[idx, :T_i] = True
        
        # Compute UOT loss using the UnbalancedOTLoss module
        # We need to handle variable-length sequences by computing per-sample
        uot_cost_losses = []
        uot_reg_losses = []
        
        for idx in range(B_valid):
            V_i = X_mask[idx].sum().item()
            T_i = Y_mask[idx].sum().item()
            
            # Extract actual tokens (no padding)
            X_i = X_padded[idx, :V_i].unsqueeze(0)  # (1, V_i, D)
            Y_i = Y_padded[idx, :T_i].unsqueeze(0)  # (1, T_i, D)
            
            # Get attention-based weights if available
            b_weights_i = None
            if Y_weights_list[idx] is not None:
                b_weights_i = Y_weights_list[idx][:T_i].unsqueeze(0)  # (1, T_i)
            
            # Compute UOT loss for this sample
            result = self.uot_loss_fn(X_i, Y_i, b_weights=b_weights_i, return_debug=False)
            uot_cost_losses.append(result["loss_uot_cost"])
            uot_reg_losses.append(result["loss_uot_reg"])
        
        # Average across valid samples
        loss_uot_cost = torch.stack(uot_cost_losses).mean()
        loss_uot_reg = torch.stack(uot_reg_losses).mean()
        
        return {
            "loss_uot_cost": loss_uot_cost,
            "loss_uot_reg": loss_uot_reg,
        }

    def _update_teacher_ema(self):
        """Update teacher adapter with EMA of student adapter weights."""
        if not self.use_teacher_ema:
            return
        
        peft_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Get named parameters for both adapters
        student_params = {n: p for n, p in peft_model.named_parameters() if 'lora' in n and 'student' in n}
        teacher_params = {n: p for n, p in peft_model.named_parameters() if 'lora' in n and 'teacher' in n}
        
        # EMA update: teacher = decay * teacher + (1 - decay) * student
        with torch.no_grad():
            for name in student_params.keys():
                # Convert adapter name from student to teacher
                teacher_name = name.replace('.student.', '.teacher.')
                if teacher_name in teacher_params:
                    teacher_params[teacher_name].mul_(self.teacher_ema_decay).add_(
                        student_params[name], alpha=1 - self.teacher_ema_decay
                    )
    
    def training_step(self, model, inputs, num_items_in_batch):
        """Override training step to add EMA update after gradient step."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Update teacher EMA after each training step
        if self.use_teacher_ema and self.state.global_step > 0:
            self._update_teacher_ema()
        
        return loss


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Graph VQA Training with KD')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL)
    parser.add_argument('--mode', type=str, default='image', choices=['image', 'text', 'kd'])
    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5) 
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--train_last_n_llm_layers', type=int, default=0, help='Number of last LLM layers to train alongside mlp1 (0 to train only mlp1)')
    parser.add_argument('--train_first_n_llm_layers', type=int, default=0, help='Number of first LLM layers to train alongside mlp1 (0 to skip)')
    parser.add_argument('--kd_alpha', type=float, default=0.5, help='Weight on the CE (soft) loss when KD is enabled')
    parser.add_argument('--kd_temperature', type=float, default=1.5)
    parser.add_argument('--kd_beta', type=float, default=1.0, help='Weight on the current CE/KD loss (L_curr)')
    parser.add_argument('--kd_gamma', type=float, default=0.0, help='Weight on the representation alignment loss (CKA/attn/moment)')
    parser.add_argument('--kd_gamma_hidden', type=float, default=0.0, help='Weight on the hidden_state representation alignment loss')
    parser.add_argument('--rep_loss_type', type=str, default='none', choices=['none', 'mu_sigma', 'cka', 'attn', 'hidden_state'], help='Representation alignment loss: mu_sigma, cka, attn, hidden_state')
    parser.add_argument('--num_hidden_layers_align', type=int, default=4, help='Number of last hidden layers to align for hidden_state rep loss')
    parser.add_argument('--teacher_correct_only', action='store_true', help='Only apply KL-div and rep loss when teacher prediction is correct; otherwise use CE loss only')
    parser.add_argument('--student_incorrect_only', action='store_true', help='Only apply KD loss (beta term) when student prediction is incorrect; compatible with teacher_correct_only')
    parser.add_argument('--max_length', type=int, default=8192)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--target_hidden_layer', type=int, default=0)
    
    # UOT (Unbalanced Optimal Transport) loss arguments
    parser.add_argument('--uot_weight', type=float, default=0.0, help='Weight for UOT loss (0 to disable)')
    parser.add_argument('--uot_eps', type=float, default=0.1, help='UOT entropy regularization')
    parser.add_argument('--uot_iters', type=int, default=50, help='UOT Sinkhorn iterations')
    parser.add_argument('--uot_tau_a', type=float, default=1.0, help='UOT row marginal KL penalty (lower = more relaxed)')
    parser.add_argument('--uot_tau_b', type=float, default=1.0, help='UOT col marginal KL penalty (lower = more relaxed)')
    parser.add_argument('--uot_lam_d', type=float, default=1.0, help='UOT weight for L2^2 distance')
    parser.add_argument('--uot_lam_c', type=float, default=0.0, help='UOT weight for cosine distance')
    parser.add_argument('--uot_cost_weight', type=float, default=1.0, help='UOT transport cost loss weight')
    parser.add_argument('--uot_reg_weight', type=float, default=1.0, help='UOT barycentric regression loss weight')
    parser.add_argument('--uot_use_teacher_attention', action='store_true', help='Use teacher attention as weights for UOT target tokens')
    
    # Training checkpoint arguments
    parser.add_argument('--save_steps', type=str, default='200', help='Comma-separated list of steps at which to save checkpoints (e.g., "200,400,800")')
    
    # Dataset selection arguments
    parser.add_argument('--dataset_type', type=str, default='graph', choices=['graph', 'gui360', 'chartinsights'],
                        help='Dataset type: graph (Graph VQA), gui360 (GUI-360), or chartinsights (ChartInsights)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Custom dataset path (overrides defaults)')
    parser.add_argument('--gui360_domain', type=str, default='all', choices=['all', 'excel', 'word', 'ppt'],
                        help='GUI-360 domain filter')
    parser.add_argument('--gui360_category', type=str, default='all', 
                        choices=['all', 'in_app', 'search', 'online'],
                        help='GUI-360 category filter')
    
    # ChartInsights dataset arguments
    parser.add_argument('--chartinsights_subset', type=str, default='Overall Evaluation',
                        choices=['Overall Evaluation', 'Textual Prompt'],
                        help='ChartInsights subset to use')
    parser.add_argument('--chartinsights_question_type', type=str, default='Multiple_choice',
                        choices=['Multiple_choice', 'Judgement_question', 'fill_the_blank'],
                        help='ChartInsights question type')
    parser.add_argument('--chartinsights_task_type', type=str, default='all',
                        help='ChartInsights task type filter (e.g., "determine range", "all")')
    parser.add_argument('--chartinsights_image_type', type=str, default='all',
                        help='ChartInsights image type filter (e.g., "line", "all")')
    parser.add_argument('--chartinsights_question_level', type=str, default='all',
                        choices=['all', 'common', 'complex'],
                        help='ChartInsights question difficulty level')
    
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Limit number of training samples (for debugging)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Teacher EMA arguments
    parser.add_argument('--use_teacher_ema', action='store_true',
                        help='Use EMA of student adapter as teacher (instead of frozen base model)')
    parser.add_argument('--teacher_ema_decay', type=float, default=0.9,
                        help='EMA decay rate for teacher adapter (0.999 or 0.9999 recommended)')
    
    # Deterministic adapter initialization
    parser.add_argument('--student_init_seed', type=int, default=42,
                        help='Seed for deterministic student adapter initialization (default: 42)')
    
    # Learning rate scheduler
    parser.add_argument('--const_lr', action='store_true',
                        help='Use constant learning rate instead of cosine schedule')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    # Model initialization and dropout use args.seed (varies across runs)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # IMPORTANT: Use a fixed seed for dataloader to ensure consistent sample ordering
    # This ensures all runs train on the same samples in the same order,
    # only model init/dropout vary with args.seed
    DATA_SEED = 42
    
    # Parse save_steps into a list of integers
    try:
        save_steps = [int(s.strip()) for s in args.save_steps.split(',')]
    except ValueError:
        print(f"Error: Invalid save_steps format: {args.save_steps}", flush=True)
        print("Expected comma-separated integers, e.g., '200,400,800'", flush=True)
        exit(1)
    
    print(f"Args: {args}", flush=True)
    print(f"Checkpoints will be saved at steps: {save_steps}", flush=True)
    
    # Detect distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1
    
    if is_distributed:
        print(f"Distributed training: rank {local_rank}/{world_size}", flush=True)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single GPU training: {torch.cuda.get_device_name(0)}", flush=True)
    
    # Output dir
    if args.output_dir is None:
        model_name = args.model.split("/")[-1]
        args.output_dir = os.path.join(SAVE_DIR, f"{model_name}_{args.mode}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Wandb (only on main process)
    if args.use_wandb and (not is_distributed or local_rank == 0):
        os.environ["WANDB_PROJECT"] = "graph-vqa-4x4-sample-efficiency-seeded-final"
        os.environ["WANDB_DIR"] = "/cluster/scratch/rbertolissi/wandb"
        os.environ["WANDB_CACHE_DIR"] = "/cluster/scratch/rbertolissi/wandb_cache"
        os.environ["WANDB_DATA_DIR"] = "/cluster/scratch/rbertolissi/wandb"
        os.environ["WANDB_ARTIFACT_LOCATION"] = "/cluster/scratch/rbertolissi/wandb"
        os.environ["WANDB_ARTIFACT_DIR"] = "/cluster/scratch/rbertolissi/wandb"
        os.environ["WANDB_CONFIG_DIR"] = "/cluster/scratch/rbertolissi/wandb"
        os.environ["WANDB_LOG_MODEL"] = "false" 
    
    # ========================================================================
    # Prepare deterministic student adapter initialization
    # ========================================================================
    # Create a clean RNG snapshot for student adapter initialization
    # This ensures reproducible adapter weights regardless of earlier randomness
    if not is_distributed or local_rank == 0:
        print(f"Creating clean RNG snapshot with student_init_seed={args.student_init_seed}")
    seed_all(args.student_init_seed)
    student_rng_state = capture_rng_state()
    
    # Restore program's training seed for other randomness (data loading, dropout, etc.)
    seed_all(args.seed)
    
    # ========================================================================
    # Load Model (single model - teacher/student via LoRA toggling)
    # ========================================================================
    print(f"Loading model: {args.model}", flush=True)
    print("Note: Same model used for both teacher (LoRA disabled) and student (LoRA enabled)")
    
    # For multi-GPU: don't use device_map="auto", let Trainer handle distribution
    # For single GPU: use device_map="auto" for easier memory management
    load_common = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if not is_distributed:
        load_common["device_map"] = "auto"

    def load_model():
        try:
            return AutoModel.from_pretrained(
                args.model,
                **load_common,
                attn_implementation="flash_attention_2",
            )
        except Exception as e:
            print(f"[WARN] flash_attention_2 not usable ({type(e).__name__}: {e}). Falling back to eager.", flush=True)
            return AutoModel.from_pretrained(
                args.model,
                **load_common,
                attn_implementation="eager",
            )

    model = load_model()
    if is_distributed:
        model = model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    # Set the IMG_CONTEXT token id for InternVL (critical for image embedding injection)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    print(f"Set img_context_token_id = {img_context_token_id}")
    
    print(f"Model loaded. Type: {type(model).__name__}", flush=True)
    
    # ========================================================================
    # Freeze vision tower (it's frozen for both teacher and student passes)
    # ========================================================================
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("Frozen: vision_model (vision tower)")
    
    # ========================================================================
    # Apply LoRA to both LLM and mlp1 (vision-language projector)
    # This allows us to use the same model for both teacher and student:
    #   - Teacher: LoRA disabled (base pretrained weights)
    #   - Student: LoRA enabled (adapters active)
    # ========================================================================
    # First, find mlp1 layer names for LoRA targeting
    # InternVL's mlp1 is a Sequential with Linear layers, we need to find them
    mlp1_targets = []
    for name, module in model.named_modules():
        if "mlp1" in name and hasattr(module, 'weight') and len(module.weight.shape) == 2:
            # This is a Linear layer inside mlp1
            # Extract the layer name pattern (e.g., "mlp1.0", "mlp1.2")
            mlp1_targets.append(name)
    
    print(f"Found mlp1 Linear layers: {mlp1_targets}")
    
    # Find first N and/or last N LLM layers if requested
    llm_layer_targets = []
    if args.train_first_n_llm_layers > 0 or args.train_last_n_llm_layers > 0:
        # Find all LLM layers (typically in language_model.model.layers)
        layer_indices = set()
        for name, module in model.named_modules():
            if 'language_model' in name and 'layers.' in name:
                # Extract layer index from name like 'language_model.model.layers.31.self_attn.q_proj'
                parts = name.split('layers.')
                if len(parts) > 1:
                    layer_idx_str = parts[1].split('.')[0]
                    if layer_idx_str.isdigit():
                        layer_indices.add(int(layer_idx_str))
        
        if layer_indices:
            total_layers = max(layer_indices) + 1
            target_layer_indices = set()
            
            # Add first N layers
            if args.train_first_n_llm_layers > 0:
                first_n = min(args.train_first_n_llm_layers, total_layers)
                first_layers = set(range(first_n))
                target_layer_indices.update(first_layers)
                print(f"LLM has {total_layers} layers, training first {first_n} layers: {sorted(first_layers)}")
            
            # Add last N layers
            if args.train_last_n_llm_layers > 0:
                last_n = min(args.train_last_n_llm_layers, total_layers)
                last_layers = set(range(total_layers - last_n, total_layers))
                target_layer_indices.update(last_layers)
                print(f"LLM has {total_layers} layers, training last {last_n} layers: {sorted(last_layers)}")
            
            print(f"Total target layers: {sorted(target_layer_indices)}")
            
            # Add all attention and FFN modules from the target layers
            # CRITICAL: We need FULL module paths, not just short names like 'q_proj'
            # Otherwise LoRA will apply to ALL layers, not just the target ones
            for name, module in model.named_modules():
                if 'language_model' in name and 'layers.' in name and hasattr(module, 'weight') and len(module.weight.shape) == 2:
                    # Extract layer index
                    parts = name.split('layers.')
                    if len(parts) > 1:
                        layer_idx_str = parts[1].split('.')[0]
                        if layer_idx_str.isdigit() and int(layer_idx_str) in target_layer_indices:
                            # Check if it's an attention or FFN layer
                            if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']):
                                # Use the FULL module path for precise targeting
                                llm_layer_targets.append(name)
            
            print(f"LLM layer modules to train (showing first 5): {llm_layer_targets[:5]}")
            print(f"Total LLM modules: {len(llm_layer_targets)}")
        else:
            print(f"Warning: Could not find LLM layers, training only mlp1")
    
    # Combine mlp1 and LLM layer targets
    all_targets = mlp1_targets + llm_layer_targets
    print(f"Final LoRA target modules: {all_targets}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=all_targets,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # ========================================================================
    # Deterministic Adapter Creation
    # ========================================================================
    print("Creating adapters with deterministic initialization...")
    
    # Restore the clean RNG state for student adapter initialization
    # This ensures student weights are deterministic regardless of prior randomness
    restore_rng_state(student_rng_state)
    
    # Create student adapter with deterministic weights
    model = get_peft_model(model, lora_config, "student")
    print("Student adapter created with deterministic initialization")
    
    # Add teacher adapter (will be initialized randomly, but we'll copy student weights)
    model.add_adapter("teacher", lora_config)
    print("Teacher adapter added")
    
    # Copy student weights to teacher adapter (exact copy, no RNG dependence)
    student_state = model.state_dict()
    teacher_state = {}
    for key, value in student_state.items():
        if '.student.' in key:
            teacher_key = key.replace('.student.', '.teacher.')
            teacher_state[teacher_key] = value.clone()
    
    # Load teacher state (strict=False since we only update adapter params)
    model.load_state_dict(teacher_state, strict=False)
    print("Teacher adapter initialized as exact copy of student")
    
    # Freeze teacher adapter parameters (teacher is never trained by optimizer)
    for name, param in model.named_parameters():
        if '.teacher.' in name:
            param.requires_grad = False
    print("Teacher adapter parameters frozen")
    
    # Set student as active adapter for training
    model.set_adapter("student")
    
    if args.use_teacher_ema:
        print(f"Teacher EMA enabled with decay={args.teacher_ema_decay}")
    
    # Restore training seed for subsequent randomness (dropout, data shuffling, etc.)
    seed_all(args.seed)
        
    model.img_context_token_id = img_context_token_id
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model.base_model.model.img_context_token_id = img_context_token_id

    # IMPORTANT: Disable gradient checkpointing on vision model
    # but keep it frozen - gradients still need to flow through for mlp1
    if hasattr(model, 'vision_model'):
        model.vision_model.gradient_checkpointing = False
        if hasattr(model.vision_model, 'encoder'):
            model.vision_model.encoder.gradient_checkpointing = False
    
    model.print_trainable_parameters()
    
    # ========================================================================
    # Data - Select dataset based on dataset_type
    # ========================================================================
    if args.dataset_type == 'gui360':
        # GUI-360 Action Prediction Dataset for KD
        # Teacher: image + a11y info, Student: clean image only
        dataset_base = args.dataset_path if args.dataset_path else GUI360_DATASET_PATH
        
        if not is_distributed or local_rank == 0:
            print(f"Using GUI-360 dataset from: {dataset_base}")
            print(f"  Domain filter: {args.gui360_domain}")
            print(f"  Category filter: {args.gui360_category}")
        
        train_dataset = create_gui360_dataset(
            data_path=os.path.join(dataset_base, "train"),
            domain=args.gui360_domain,
            category=args.gui360_category,
            max_samples=args.max_samples,
        )
        
        # For evaluation, use test set (smaller)
        eval_dataset = create_gui360_dataset(
            data_path=os.path.join(dataset_base, "test"),
            domain=args.gui360_domain,
            category=args.gui360_category,
            max_samples=args.max_samples // 10 if args.max_samples else 100,  # Smaller eval set
        )
        
        # GUI-360 specific data collator that handles different teacher/student images
        data_collator = GUI360KDDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            mode=args.mode,
            num_image_token=model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token,
        )
    elif args.dataset_type == 'chartinsights':
        # ChartInsights Chart QA Dataset for KD
        # Teacher: chart image + CSV table, Student: chart image only
        dataset_base = args.dataset_path if args.dataset_path else CHARTINSIGHTS_DATASET_PATH
        
        if not is_distributed or local_rank == 0:
            print(f"Using ChartInsights dataset from: {dataset_base}")
            print(f"  Subset: {args.chartinsights_subset}")
            print(f"  Question type: {args.chartinsights_question_type}")
            print(f"  Task type filter: {args.chartinsights_task_type}")
            print(f"  Image type filter: {args.chartinsights_image_type}")
            print(f"  Question level: {args.chartinsights_question_level}")
        
        train_dataset = create_chartinsights_dataset(
            data_path=dataset_base,
            subset=args.chartinsights_subset,
            question_type=args.chartinsights_question_type,
            task_type_filter=args.chartinsights_task_type,
            image_type_filter=args.chartinsights_image_type,
            question_level_filter=args.chartinsights_question_level,
            split='train',
            max_samples=args.max_samples,
        )
        
        eval_dataset = create_chartinsights_dataset(
            data_path=dataset_base,
            subset=args.chartinsights_subset,
            question_type=args.chartinsights_question_type,
            task_type_filter=args.chartinsights_task_type,
            image_type_filter=args.chartinsights_image_type,
            question_level_filter=args.chartinsights_question_level,
            split='validation',
            max_samples=args.max_samples // 10 if args.max_samples else 100,
        )
        
        # Use the same KDDataCollator as GraphMCQDataset since ChartInsights now outputs
        # the same format (no space before answer, proper newlines)
        data_collator = KDDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            mode=args.mode,
            num_image_token=model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token,
        )
    else:
        # Default: Graph MCQ Dataset
        dataset_base = args.dataset_path if args.dataset_path else DATASET_PATH
        dataset_path = Path(dataset_base)
        
        if not is_distributed or local_rank == 0:
            print(f"Using Graph VQA dataset from: {dataset_base}")

        # Support both layouts:
        #  1) Split folders: <base>/train/data.jsonl and <base>/validation/data.jsonl
        #  2) Single file/dir: <base>/data.jsonl with optional `split` field per sample
        if dataset_path.is_dir() and (dataset_path / "train").exists() and (dataset_path / "validation").exists():
            train_dataset = GraphMCQDataset(
                str(dataset_path / "train"),
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_samples=args.max_samples,
            )
            eval_dataset = GraphMCQDataset(
                str(dataset_path / "validation"),
                tokenizer=tokenizer,
                max_length=args.max_length,
                max_samples=args.max_samples // 10 if args.max_samples else None,
            )
        else:
            jsonl_path = dataset_path if dataset_path.is_file() else (dataset_path / "data.jsonl")
            train_dataset = GraphMCQDataset(
                str(jsonl_path),
                tokenizer=tokenizer,
                max_length=args.max_length,
                split="train",
                max_samples=args.max_samples,
            )
            eval_dataset = GraphMCQDataset(
                str(jsonl_path),
                tokenizer=tokenizer,
                max_length=args.max_length,
                split="validation",
                max_samples=args.max_samples // 10 if args.max_samples else None,
            )
            if len(eval_dataset) == 0 and len(train_dataset) > 1:
                # If the dataset doesn't provide a validation split, create a deterministic hold-out set.
                val_size = max(1, int(0.05 * len(train_dataset)))
                train_size = len(train_dataset) - val_size
                generator = torch.Generator().manual_seed(DATA_SEED)
                train_dataset, eval_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, val_size], generator=generator
                )
            if len(eval_dataset) == 0:
                eval_dataset = train_dataset
        
        data_collator = KDDataCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            mode=args.mode,
            # Get from base model (PEFT wraps the model, access via base_model)
            num_image_token=model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token,
        )
    # Get the num_image_token for logging
    num_image_token = model.base_model.model.num_image_token if hasattr(model, 'base_model') else model.num_image_token
    if not is_distributed or local_rank == 0:
        print(f"Using num_image_token = {num_image_token}")
    
    # ========================================================================
    # Print a random training example for verification
    # ========================================================================
    if not is_distributed or local_rank == 0:
        print("\n" + "="*80)
        print("RANDOM TRAINING EXAMPLE FOR VERIFICATION")
        print("="*80)
        
        # Get a random sample
        random_idx = random.randint(0, len(train_dataset) - 1)
        sample = train_dataset[random_idx]
        
        print(f"\nSample index: {random_idx}")
        print(f"Answer: {sample['answer']}")
        print(f"\n--- STUDENT PROMPT (image only) ---")
        print(sample['student_prompt'])
        print(f"\n--- STUDENT FULL TEXT (prompt + answer) ---")
        print(sample['student_text'])
        print(f"\n--- TEACHER PROMPT (image + text description) ---")
        print(sample['teacher_prompt'])
        print(f"\n--- TEACHER FULL TEXT (prompt + answer) ---")
        print(sample['teacher_text'])
        print(f"\n--- IMAGE INFO ---")
        print(f"Image size: {sample['image'].size}")
        print(f"Image mode: {sample['image'].mode}")
        
        # Also show a collated batch to verify tokenization
        print(f"\n--- COLLATED BATCH (single sample) ---")
        batch = data_collator([sample])
        print(f"Student input_ids shape: {batch['input_ids'].shape}")
        print(f"Student attention_mask shape: {batch['attention_mask'].shape}")
        print(f"Student labels shape: {batch['labels'].shape}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        
        if args.mode == "kd":
            print(f"Teacher input_ids shape: {batch['teacher_input_ids'].shape}")
            print(f"Teacher attention_mask shape: {batch['teacher_attention_mask'].shape}")
            print(f"Teacher pixel_values shape: {batch['teacher_pixel_values'].shape}")
        
        # Decode tokens to verify
        print(f"\n--- DECODED STUDENT INPUT ---")
        decoded_student = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
        # Truncate for display (image tokens are very long)
        if len(decoded_student) > 500:
            print(decoded_student[:250] + "\n... [truncated] ...\n" + decoded_student[-250:])
        else:
            print(decoded_student)
        
        if args.mode == "kd":
            print(f"\n--- DECODED TEACHER INPUT ---")
            decoded_teacher = tokenizer.decode(batch['teacher_input_ids'][0], skip_special_tokens=False)
            if len(decoded_teacher) > 500:
                print(decoded_teacher[:250] + "\n... [truncated] ...\n" + decoded_teacher[-250:])
            else:
                print(decoded_teacher)
        
        print("\n" + "="*80)
        print("END OF TRAINING EXAMPLE VERIFICATION")
        print("="*80 + "\n")
    
    # ========================================================================
    # Training
    # ========================================================================
    # Adjust gradient accumulation based on number of GPUs
    # Total batch = per_device_batch * num_gpus * grad_accum_steps
    # We want ~16 effective batch size, so adjust grad_accum based on world_size
    grad_accum_steps = max(1, 16 // (args.batch_size * world_size))
    
    # Create descriptive run name for wandb
    model_short = args.model.split("/")[-1]
    run_name = f"{model_short}_{args.mode}_lr{args.lr}_lora{args.lora_r}"
    if args.train_first_n_llm_layers > 0 and args.train_last_n_llm_layers > 0:
        run_name += f"_llmF{args.train_first_n_llm_layers}L{args.train_last_n_llm_layers}"
    elif args.train_first_n_llm_layers > 0:
        run_name += f"_llmF{args.train_first_n_llm_layers}"
    elif args.train_last_n_llm_layers > 0:
        run_name += f"_llmL{args.train_last_n_llm_layers}"
    if args.mode == "kd":
        run_name += f"_alpha{args.kd_alpha}_beta{args.kd_beta}_temp{args.kd_temperature}"
        if args.kd_gamma > 0 and args.rep_loss_type != "none":
            run_name += f"_rep-{args.rep_loss_type}_gamma{args.kd_gamma}"
        if args.kd_gamma_hidden > 0:
            run_name += f"_hidden_gammah{args.kd_gamma_hidden}"
            run_name += f"_k{args.num_hidden_layers_align}"
        if args.teacher_correct_only:
            run_name += "_tco"  # tco = teacher_correct_only
        if args.student_incorrect_only:
            run_name += "_sio"  # sio = student_incorrect_only
        if args.use_teacher_ema:
            run_name += f"_ema{args.teacher_ema_decay}"
        if args.uot_weight > 0:
            # Include key UOT hyperparameters in run name
            run_name += f"_uot{args.uot_weight}"
            run_name += f"_eps{args.uot_eps}_i{args.uot_iters}"
            run_name += f"_ta{args.uot_tau_a}_tb{args.uot_tau_b}"
            # Only add lam_c if using cosine distance (non-zero)
            if args.uot_lam_c > 0:
                run_name += f"_ld{args.uot_lam_d}_lc{args.uot_lam_c}"
            # Only add weights if they differ from default (cost=0, reg=1)
            if args.uot_cost_weight != 0.0 or args.uot_reg_weight != 1.0:
                run_name += f"_cw{args.uot_cost_weight}_rw{args.uot_reg_weight}"
        if args.target_hidden_layer != 0:
            run_name += f"_thl{args.target_hidden_layer}"
    
    # Initialize wandb before trainer to set custom config
    if args.use_wandb and (not is_distributed or local_rank == 0):
        import wandb
        wandb.init(
            name=run_name,
            config={
                "mode": args.mode,
                "learning_rate": args.lr,
                "lora_r": args.lora_r,
                "train_first_n_llm_layers": args.train_first_n_llm_layers,
                "train_last_n_llm_layers": args.train_last_n_llm_layers,
                "kd_alpha": args.kd_alpha,
                "kd_beta": args.kd_beta,
                "kd_temperature": args.kd_temperature,
                "kd_gamma": args.kd_gamma,
                "kd_gamma_hidden": args.kd_gamma_hidden,
                "rep_loss_type": args.rep_loss_type,
                "num_hidden_layers_align": args.num_hidden_layers_align,
                "teacher_correct_only": args.teacher_correct_only,
                "student_incorrect_only": args.student_incorrect_only,
                "use_teacher_ema": args.use_teacher_ema,
                "teacher_ema_decay": args.teacher_ema_decay,
                "target_hidden_layer": args.target_hidden_layer,
                "uot_weight": args.uot_weight,
                "uot_eps": args.uot_eps,
                "uot_iters": args.uot_iters,
                "uot_tau_a": args.uot_tau_a,
                "uot_tau_b": args.uot_tau_b,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "effective_batch_size": args.batch_size * world_size * grad_accum_steps,
                "seed": args.seed,
                "data_seed": DATA_SEED,
            }
        )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=run_name,
        num_train_epochs=args.epochs,
        max_steps=max(save_steps) if save_steps else -1,  # End training at last checkpoint step
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=args.lr,
        weight_decay=0.01, 
        lr_scheduler_type="constant" if args.const_lr else "cosine",
        warmup_ratio=0.03,  
        optim="adamw_torch", 
        bf16=True,
        eval_strategy="steps", 
        eval_steps=200,
        save_strategy="steps",  # Save at regular intervals
        save_steps=save_steps[0] if save_steps else 200,  # Save every first checkpoint step
        save_total_limit=None,  # Keep all checkpoints
        logging_steps=1 if save_steps else 20,  # More frequent logging
        report_to="wandb" if args.use_wandb else "none",
        seed=args.seed,  # initialization seed (varies)
        data_seed=DATA_SEED,  # Fixed seed for data shuffling (consistent samples)
        remove_unused_columns=False,
        gradient_checkpointing=True,  
        # Multi-GPU settings
        ddp_find_unused_parameters=False,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
    )
    
    if not is_distributed or local_rank == 0:
        print(f"Training config: {world_size} GPU(s), batch_size={args.batch_size}, grad_accum={grad_accum_steps}")
        print(f"Effective batch size: {args.batch_size * world_size * grad_accum_steps}")
        if args.uot_weight > 0:
            print(f"UOT loss enabled: weight={args.uot_weight}, eps={args.uot_eps}, tau_a={args.uot_tau_a}, tau_b={args.uot_tau_b}")
    
    # Use KD trainer for kd mode, regular for others
    # KDTrainer uses the same model with LoRA toggling for teacher/student
    if args.mode == "kd":
        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            kd_beta=args.kd_beta,
            kd_gamma=args.kd_gamma,
            kd_gamma_hidden=args.kd_gamma_hidden,
            rep_loss_type=args.rep_loss_type,
            target_hidden_layer=args.target_hidden_layer,
            num_hidden_layers_align=args.num_hidden_layers_align,
            teacher_correct_only=args.teacher_correct_only,
            student_incorrect_only=args.student_incorrect_only,
            use_teacher_ema=args.use_teacher_ema,
            teacher_ema_decay=args.teacher_ema_decay,
            # UOT loss parameters
            uot_weight=args.uot_weight,
            uot_eps=args.uot_eps,
            uot_iters=args.uot_iters,
            uot_tau_a=args.uot_tau_a,
            uot_tau_b=args.uot_tau_b,
            uot_lam_d=args.uot_lam_d,
            uot_lam_c=args.uot_lam_c,
            uot_cost_weight=args.uot_cost_weight,
            uot_reg_weight=args.uot_reg_weight,
            uot_use_teacher_attention=args.uot_use_teacher_attention,
        )
    else:
        # Use KDTrainer with KD disabled for pure CE loss, but keep unified handling
        trainer = KDTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            kd_alpha=1.0,
            kd_temperature=args.kd_temperature,
            kd_beta=args.kd_beta,
            kd_gamma=0.0,
            rep_loss_type="none",
            enable_kd=False,
            num_hidden_layers_align=args.num_hidden_layers_align,
            teacher_correct_only=False,
            student_incorrect_only=False,
            # UOT loss disabled for non-KD modes
            uot_weight=0.0,
        )
    trainer.remove_callback(PrinterCallback)
    
    print("Starting training...", flush=True)
    trainer.train()
    
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save base model info for evaluation to use
    # This helps the evaluation script detect which base model to use for loading
    base_model_info = {
        "base_model_name": args.model,
        "training_mode": args.mode,
        "lora_r": args.lora_r,
        "train_first_n_llm_layers": args.train_first_n_llm_layers,
        "train_last_n_llm_layers": args.train_last_n_llm_layers,
        "kd_alpha": args.kd_alpha,
        "teacher_correct_only": args.teacher_correct_only,
        "student_incorrect_only": args.student_incorrect_only,
        "use_teacher_ema": args.use_teacher_ema,
        "teacher_ema_decay": args.teacher_ema_decay,
        "uot_weight": args.uot_weight,
        "timestamp": os.path.basename(args.output_dir)
    }
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(base_model_info, f, indent=2)
    
    print(f"Model saved to {args.output_dir}", flush=True)
    print(f"Base model info saved for evaluation", flush=True)


if __name__ == "__main__":
    import sys
    # Allow running POT loss test with --test_pot flag
    if "--test_pot" in sys.argv:
        test_partial_ot_loss()
    else:
        main()
