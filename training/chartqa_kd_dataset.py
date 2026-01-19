"""
ChartQA Knowledge Distillation Dataset for training.

This dataset implements the knowledge distillation approach for ChartQA:
- Teacher: Receives chart image + CSV table + annotations (privileged information)
- Student: Receives only the chart image (visual-only)

The goal is to distill knowledge from a model that has access to precise CSV data and chart
annotations to a model that only sees the chart image, improving visual chart understanding.

Key features:
- Uses ONLY the human-annotated split (train_human.json) for training
- Teacher gets both CSV tables AND chart annotations by default
- Follows the same format as evaluation/loaders/chartqa_custom.py for consistency

Usage in train_kd.py:
    from chartqa_kd_dataset import create_chartqa_dataset, ChartQAKDDataCollator

    dataset = create_chartqa_dataset(
        data_path="/cluster/scratch/rbertolissi/datasets/ChartQA_Dataset",
        split="train",
        include_csv=True,
        include_annotations=True,
        max_samples=None
    )
"""

import os
import json
import csv
import random
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image
import torch
from torch.utils.data import Dataset


def create_chartqa_dataset(
    data_path: str,
    split: str = "train",
    include_csv: bool = True,
    include_annotations: bool = True,
    csv_rows_limit: Optional[int] = None,
    csv_to_text_format: str = "markdown",
    max_samples: Optional[int] = None,
    train_val_ratio: float = 0.9,
) -> 'ChartQAKDDataset':
    """
    Factory function to create ChartQA KD dataset.
    
    Args:
        data_path: Path to ChartQA_Dataset root directory
        split: 'train' or 'validation' (internal split from human data)
        include_csv: Whether to include CSV table data for teacher
        include_annotations: Whether to include chart annotations for teacher
        csv_rows_limit: Maximum number of CSV rows to include (None = all)
        csv_to_text_format: Format for CSV data ("markdown" or "csv")
        max_samples: Limit number of samples (for debugging)
        train_val_ratio: Ratio for train/val split (default 0.9)
    
    Returns:
        ChartQAKDDataset instance
    """
    return ChartQAKDDataset(
        data_path=data_path,
        split=split,
        include_csv=include_csv,
        include_annotations=include_annotations,
        csv_rows_limit=csv_rows_limit,
        csv_to_text_format=csv_to_text_format,
        max_samples=max_samples,
        train_val_ratio=train_val_ratio,
    )


class ChartQAKDDataset(Dataset):
    """
    ChartQA dataset for Knowledge Distillation training.
    
    IMPORTANT: This dataset ONLY uses train_human.json (human-annotated questions).
    The augmented questions are excluded as they are machine-generated.
    
    Loads questions from ChartQA benchmark and prepares:
    - Student: Chart image + question (visual-only)
    - Teacher: Chart image + CSV context + annotations + question (privileged)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        include_csv: bool = True,
        include_annotations: bool = True,
        csv_rows_limit: Optional[int] = None,
        csv_to_text_format: str = "markdown",
        max_samples: Optional[int] = None,
        train_val_ratio: float = 0.9,
    ):
        """
        Initialize ChartQA KD dataset.
        
        Args:
            data_path: Path to ChartQA_Dataset root directory
            split: 'train' or 'validation'
            include_csv: Whether to include CSV context for teacher
            include_annotations: Whether to include annotations for teacher
            csv_rows_limit: Maximum rows to include from CSV
            csv_to_text_format: CSV format ("markdown" or "csv")
            max_samples: Limit samples for debugging
            train_val_ratio: Train/validation split ratio
        """
        self.data_path = Path(data_path)
        self.split = split
        self.include_csv = include_csv
        self.include_annotations = include_annotations
        self.csv_rows_limit = csv_rows_limit
        self.csv_to_text_format = csv_to_text_format
        self.max_samples = max_samples
        self.train_val_ratio = train_val_ratio
        
        # Set up paths - use train directory for both train and validation splits
        # We create an internal split from the human data
        self.train_dir = self.data_path / "train"
        self.tables_dir = self.train_dir / "tables"
        self.annotations_dir = self.train_dir / "annotations"
        self.images_dir = self.train_dir / "png"
        
        # Validate paths
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory not found: {self.train_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Check for optional directories
        if self.include_csv and not self.tables_dir.exists():
            print(f"Warning: Tables directory not found: {self.tables_dir}. CSV context will be empty.")
            self.include_csv = False
        if self.include_annotations and not self.annotations_dir.exists():
            print(f"Warning: Annotations directory not found: {self.annotations_dir}. Annotations will be empty.")
            self.include_annotations = False
        
        # Load data (ONLY human annotations)
        self.data = self._load_data()
        
        print(f"ChartQA KD dataset loaded:")
        print(f"  Data path: {data_path}")
        print(f"  Split: {split}")
        print(f"  Include CSV: {include_csv}")
        print(f"  Include annotations: {include_annotations}")
        print(f"  Total samples: {len(self.data)}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load human-annotated QA pairs and apply train/val split."""
        # Load ONLY human annotations (not augmented)
        human_file = self.train_dir / "train_human.json"
        
        if not human_file.exists():
            raise FileNotFoundError(f"Human annotations file not found: {human_file}")
        
        with open(human_file, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
        
        # Convert to internal format
        all_samples = []
        for item in human_data:
            all_samples.append({
                'imgname': item['imgname'],
                'question': item['query'],
                'answer': item['label'],
            })
        
        # Apply deterministic train/validation split
        random.seed(42)
        indices = list(range(len(all_samples)))
        random.shuffle(indices)
        
        split_idx = int(self.train_val_ratio * len(all_samples))
        
        if self.split == 'train':
            selected_indices = indices[:split_idx]
        elif self.split == 'validation':
            selected_indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'validation'.")
        
        filtered_data = [all_samples[i] for i in selected_indices]
        
        # Apply max_samples limit if specified
        if self.max_samples is not None and self.max_samples > 0:
            filtered_data = filtered_data[:self.max_samples]
        
        return filtered_data
    
    def _format_csv(self, csv_path: Path) -> str:
        """
        Format CSV table data for inclusion in prompt.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Formatted CSV string
        """
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # Limit rows if specified
            if self.csv_rows_limit and len(rows) > self.csv_rows_limit + 1:  # +1 for header
                rows = [rows[0]] + rows[1:self.csv_rows_limit + 1]
                rows.append(['...', '(truncated)'])
            
            if len(rows) < 2:
                return ""
            
            if self.csv_to_text_format == "markdown":
                # Create markdown table
                lines = []
                lines.append("| " + " | ".join(rows[0]) + " |")
                lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
                for row in rows[1:]:
                    if len(row) == len(rows[0]):  # Ensure same number of columns
                        lines.append("| " + " | ".join(row) + " |")
                return "\n".join(lines)
            else:  # CSV format
                lines = []
                for row in rows:
                    lines.append(",".join(row))
                return "\n".join(lines)
        except Exception as e:
            print(f"Warning: Could not read CSV {csv_path}: {e}")
            return ""
    
    def _format_annotations(self, ann_path: Path) -> str:
        """
        Format chart annotations for inclusion in prompt.
        
        Args:
            ann_path: Path to annotation JSON file
            
        Returns:
            Formatted annotation string
        """
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
            
            parts = []
            
            # Chart type
            if 'type' in ann:
                parts.append(f"Chart Type: {ann['type']}")
            
            # Title
            if 'general_figure_info' in ann and 'title' in ann['general_figure_info']:
                title_info = ann['general_figure_info']['title']
                if 'text' in title_info:
                    parts.append(f"Title: {title_info['text']}")
            
            # Axes information
            if 'general_figure_info' in ann:
                fig_info = ann['general_figure_info']
                
                # X-axis labels
                if 'x_axis' in fig_info and 'major_labels' in fig_info['x_axis']:
                    x_labels = fig_info['x_axis']['major_labels'].get('values', [])
                    if x_labels:
                        parts.append(f"X-axis: {', '.join(map(str, x_labels))}")
                
                # Y-axis labels
                if 'y_axis' in fig_info and 'major_labels' in fig_info['y_axis']:
                    y_labels = fig_info['y_axis']['major_labels'].get('values', [])
                    if y_labels:
                        parts.append(f"Y-axis: {', '.join(map(str, y_labels))}")
            
            # Data series/bars
            if 'models' in ann and ann['models']:
                for model in ann['models']:
                    if 'name' in model:
                        parts.append(f"Elements: {model['name']}")
                        
                        # Include actual data values if available
                        if 'x' in model and 'y' in model:
                            x_vals = model['x']
                            y_vals = model['y']
                            if len(x_vals) == len(y_vals) and len(x_vals) <= 10:
                                # Only include if reasonable number of points
                                data_str = ", ".join([f"{x}={y}" for x, y in zip(x_vals, y_vals)])
                                parts.append(f"Data: {data_str}")
            
            return "\n".join(parts)
        except Exception as e:
            print(f"Warning: Could not read annotations {ann_path}: {e}")
            return ""
    
    def _load_context(self, imgname: str) -> str:
        """
        Load additional context (CSV and/or annotations) for an image.
        
        Args:
            imgname: Image filename (e.g., '10095.png')
            
        Returns:
            Formatted context string to prepend to the question
        """
        context_parts = []
        
        # Extract base name without extension
        base_name = Path(imgname).stem
        
        # Load CSV table if requested
        if self.include_csv:
            csv_file = self.tables_dir / f"{base_name}.csv"
            if csv_file.exists():
                csv_context = self._format_csv(csv_file)
                if csv_context:
                    csv_intro = f"The data described by the chart can be summarised by the following {'markdown' if self.csv_to_text_format == 'markdown' else 'CSV'} table:"
                    context_parts.append(csv_intro + "\n" + csv_context)
        
        # Load annotations if requested
        if self.include_annotations:
            ann_file = self.annotations_dir / f"{base_name}.json"
            if ann_file.exists():
                ann_context = self._format_annotations(ann_file)
                if ann_context:
                    structure_intro = "The structure of the chart can be summarised as follows:"
                    context_parts.append(structure_intro + "\n" + ann_context)
        
        if context_parts:
            return "Below we give some additional information about the chart:\n\n" + "\n\n".join(context_parts)
        return ""
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.
        
        Returns dict with:
            - idx: Sample index
            - answer: The answer text
            - image: PIL Image
            - question: Question text
            - student_text: Full prompt + answer for student
            - student_prompt: Just the prompt for student
            - teacher_text: Full prompt + answer for teacher
            - teacher_prompt: Just the prompt for teacher
        """
        sample = self.data[idx]
        
        imgname = sample['imgname']
        question = sample['question']
        answer = sample['answer']
        
        # Build student prompt (image-only, following ChartQA format)
        # ChartQA uses short answers, so we use a specific prompt format
        student_question_text = f"Question: {question}\n"
        student_question_text += "Answer the question using a single word or phrase.\nAnswer:\n"
        
        student_prompt = f"<image>\n{student_question_text}"
        student_text = f"{student_prompt}{answer}"
        
        # Build teacher prompt (image + CSV context + annotations)
        context = self._load_context(imgname)
        
        teacher_context = ""
        if context:
            teacher_context = f"{context}\n\n"
        
        teacher_context += f"Given the image and the additional information above, answer the question.\n"
        teacher_context += f"Question: {question}\n"
        teacher_context += "Answer the question using a single word or phrase.\nAnswer:\n"
        
        teacher_prompt = f"<image>\n{teacher_context}"
        teacher_text = f"{teacher_prompt}{answer}"
        
        # Load image
        image_path = self.images_dir / imgname
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            image = Image.new("RGB", (448, 448), (255, 255, 255))
        
        # For KD training, we need a "graph_text" field that represents the privileged information
        # This is used for representation alignment loss
        graph_text = context if context else ""
        
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


class ChartQAKDDataCollator:
    """
    Data collator for ChartQA KD training.
    
    Handles tokenization and batching for both student and teacher inputs.
    Note: This collator reuses KDDataCollator from train_kd.py with minor adjustments.
    """
    
    # InternVL special tokens
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    
    def __init__(self, tokenizer, max_length=4096, mode="kd", num_image_token=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.num_image_token = num_image_token
    
    def __call__(self, features):
        """Process a batch of samples."""
        # Import here to avoid circular dependency
        # Try train.py first, fallback to train_kd.py
        try:
            from train import load_image_for_internvl
        except ImportError:
            from train_kd import load_image_for_internvl
        
        batch = {}
        
        # ====================================================================
        # Student inputs (image mode)
        # ====================================================================
        if self.mode in ["image", "kd"]:
            # Process images
            images = [f["image"] for f in features]
            pixel_values_list = []
            num_patches_list = []
            
            for img in images:
                pv = load_image_for_internvl(img, input_size=448, max_num=1)
                pixel_values_list.append(pv)
                num_patches_list.append(pv.shape[0])
            
            # Tokenize student prompts
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
                
                # Tokenize prompt and answer separately
                prompt_encoding = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length-50
                )
                # Add space before answer for proper tokenization
                answer_encoding = self.tokenizer(
                    " " + answer, add_special_tokens=False, truncation=True, max_length=50
                )
                
                # Concatenate
                input_ids = prompt_encoding["input_ids"] + answer_encoding["input_ids"]
                attention_mask = prompt_encoding["attention_mask"] + answer_encoding["attention_mask"]
                labels = [-100] * len(prompt_encoding["input_ids"]) + answer_encoding["input_ids"]
                
                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_labels.append(labels)
            
            # Pad to max length in batch
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
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            batch["image_flags"] = torch.ones(batch["pixel_values"].shape[0], dtype=torch.long)
        
        # ====================================================================
        # Teacher inputs (image + CSV + annotations)
        # ====================================================================
        if self.mode == "kd":
            # Tokenize teacher prompts
            teacher_input_ids = []
            teacher_attention_masks = []
            
            for i, f in enumerate(features):
                num_patches = num_patches_list[i]
                image_tokens = (
                    self.IMG_START_TOKEN + 
                    self.IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + 
                    self.IMG_END_TOKEN
                )
                prompt = f["teacher_prompt"].replace("<image>", image_tokens)
                answer = f["answer"]
                
                # Tokenize with space before answer
                prompt_encoding = self.tokenizer(
                    prompt, add_special_tokens=True, truncation=True, max_length=self.max_length-50
                )
                answer_encoding = self.tokenizer(
                    " " + answer, add_special_tokens=False, truncation=True, max_length=50
                )
                
                input_ids = prompt_encoding["input_ids"] + answer_encoding["input_ids"]
                attention_mask = prompt_encoding["attention_mask"] + answer_encoding["attention_mask"]
                
                teacher_input_ids.append(input_ids)
                teacher_attention_masks.append(attention_mask)
            
            # Pad teacher sequences
            max_teacher_len = max(len(ids) for ids in teacher_input_ids)
            padded_teacher_ids = []
            padded_teacher_masks = []
            
            for input_ids, attention_mask in zip(teacher_input_ids, teacher_attention_masks):
                pad_len = max_teacher_len - len(input_ids)
                padded_teacher_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_teacher_masks.append(attention_mask + [0] * pad_len)
            
            batch["teacher_input_ids"] = torch.tensor(padded_teacher_ids, dtype=torch.long)
            batch["teacher_attention_mask"] = torch.tensor(padded_teacher_masks, dtype=torch.long)
            
            # Build teacher_graph_mask: mask selecting the context tokens (CSV + annotations)
            # This is used for representation alignment loss
            teacher_graph_mask = torch.zeros_like(batch["teacher_input_ids"], dtype=torch.bool)
            
            # For ChartQA, the graph_text (context) comes before the question
            # We need to identify where the context tokens are in the teacher sequence
            for i, f in enumerate(features):
                if f.get("graph_text", ""):
                    # Tokenize just the context to find its length
                    context_encoding = self.tokenizer(
                        f["graph_text"], add_special_tokens=False, truncation=False
                    )
                    context_len = len(context_encoding["input_ids"])
                    
                    # The context starts after the image tokens
                    # Find image token positions
                    img_context_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
                    teacher_ids = batch["teacher_input_ids"][i].tolist()
                    
                    # Find end of image tokens
                    img_end = 0
                    for j, tid in enumerate(teacher_ids):
                        if tid == img_context_id:
                            img_end = j
                    
                    # Context starts after image tokens + newline
                    context_start = img_end + 2  # +2 for </img> and newline
                    context_end = min(context_start + context_len, len(teacher_ids))
                    
                    # Set mask for context tokens
                    if context_start < context_end:
                        teacher_graph_mask[i, context_start:context_end] = True
            
            # Apply attention mask
            teacher_graph_mask = teacher_graph_mask & batch["teacher_attention_mask"].bool()
            batch["teacher_graph_mask"] = teacher_graph_mask
            
            # Teacher uses same images as student
            batch["teacher_pixel_values"] = batch["pixel_values"]
            batch["teacher_image_flags"] = batch["image_flags"]
        
        # ====================================================================
        # Text-only mode (no image for student)
        # ====================================================================
        if self.mode == "text":
            # For text-only, use teacher format (with context) as the main input
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
                prompt_encoding = self.tokenizer(prompt, add_special_tokens=True)
                prompt_len = len(prompt_encoding["input_ids"])
                labels[i, :prompt_len] = -100
            batch["labels"] = labels
        
        return batch
