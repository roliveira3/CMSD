"""
CLEVR dataset loader for visual reasoning evaluation.

Implements the evaluation setup from "Zero-Shot Visual Reasoning by Vision-Language Models:
Benchmarking and Analysis" (https://arxiv.org/abs/2409.00106v1).

Supports two evaluation modes:
1. Image-only (VLM mode): Only the image is provided to the model
2. Image+Text (VLM+Metadata mode): Both image and scene description are provided

The scene description follows the exact format from the paper, including:
- Object attributes (color, size, shape, material, coordinates)
- Spatial relationships between objects
- Direction vectors

Key features:
- Stores functional program length for each question (for analysis by reasoning steps)
- Stores question family index for category analysis
- Supports both validation (with answers) and test (without answers) splits
- Uses the exact prompts from the paper

Usage:
    # Image-only evaluation (VLM mode)
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader CLEVR \
        --dataset_path /cluster/scratch/rbertolissi/datasets/CLEVR_v1.0 \
        --clevr_split val \
        --clevr_mode image_only

    # Image+Text evaluation (VLM+Metadata mode)
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader CLEVR \
        --dataset_path /cluster/scratch/rbertolissi/datasets/CLEVR_v1.0 \
        --clevr_split val \
        --clevr_mode image_text

    # Test split evaluation (no ground truth answers)
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader CLEVRTest \
        --dataset_path /cluster/scratch/rbertolissi/datasets/CLEVR_v1.0
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from vlmeval.dataset.image_base import ImageBaseDataset
import warnings


# CLEVR answer vocabulary for standardizing responses
CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
CLEVR_SHAPES = ["cube", "cylinder", "sphere"]
CLEVR_MATERIALS = ["metal", "rubber"]
CLEVR_SIZES = ["large", "small"]
CLEVR_YES_NO = ["yes", "no"]

# Question family names (from CLEVR paper)
CLEVR_QUESTION_FAMILIES = {
    # exist questions (0-9)
    range(0, 10): "exist",
    # count questions (10-19)
    range(10, 20): "count",
    # compare_integer questions (20-29)
    range(20, 30): "compare_integer",
    # query_attribute questions (30-49)
    range(30, 50): "query_attribute",
    # compare_attribute questions (50-89)
    range(50, 90): "compare_attribute",
}


def get_question_family(family_index: int) -> str:
    """Map question family index to family name."""
    for index_range, family_name in CLEVR_QUESTION_FAMILIES.items():
        if family_index in index_range:
            return family_name
    return "unknown"


class CLEVRDataset(ImageBaseDataset):
    """
    CLEVR dataset loader for visual reasoning evaluation.
    
    This loader implements the exact evaluation setup from the paper
    "Zero-Shot Visual Reasoning by Vision-Language Models".
    
    Args:
        dataset: Dataset name
        dataset_path: Path to CLEVR_v1.0 directory
        split: Which split to use ('val' or 'train')
        mode: Evaluation mode - 'image_only' or 'image_text'
        num_samples: Limit number of samples (for debugging)
    """
    
    TYPE = 'VQA'
    
    # Setup prompt from the paper - provides vocabulary constraints
    SETUP_PROMPT = """You may assume that any metal object is shiny, and any rubber object is not shiny ("matte"). All objects are either "metal" or "rubber", and in 2 sizes: "large" or "small". All objects are one of the following colours: "blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow". All objects are one of the following shapes: "cube", "cylinder", "sphere". For numeric answers, give an integer and not in words.

Now answer the following question in one word."""
    
    def __init__(
        self,
        dataset: str = 'CLEVR',
        dataset_path: Optional[str] = None,
        split: str = 'val',
        mode: str = 'image_only',  # 'image_only' or 'image_text'
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize CLEVR dataset loader.
        
        Args:
            dataset: Dataset name
            dataset_path: Path to CLEVR_v1.0 directory
            split: Which split to use ('val' or 'train')
            mode: 'image_only' for VLM evaluation, 'image_text' for VLM+Metadata
            num_samples: Limit number of samples (None = all)
        """
        if dataset_path is None:
            raise ValueError("dataset_path is required for CLEVR dataset")
        
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.mode = mode
        self.num_samples = num_samples
        
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"CLEVR dataset not found: {dataset_path}")
        
        # Set up paths
        self.questions_file = self.dataset_path / "questions" / f"CLEVR_{split}_questions.json"
        self.scenes_file = self.dataset_path / "scenes" / f"CLEVR_{split}_scenes.json"
        self.images_dir = self.dataset_path / "images" / split
        
        if not self.questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_file}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Load scenes if available (needed for image_text mode)
        self.scenes_by_image = {}
        if self.scenes_file.exists():
            print(f"Loading scene descriptions from {self.scenes_file}...")
            with open(self.scenes_file, 'r') as f:
                scenes_data = json.load(f)
            for scene in scenes_data['scenes']:
                self.scenes_by_image[scene['image_filename']] = scene
            print(f"Loaded {len(self.scenes_by_image)} scene descriptions")
        elif mode == 'image_text':
            warnings.warn(
                f"Scene file not found ({self.scenes_file}). "
                "Image+text mode will not include scene descriptions."
            )
        
        # Update dataset name to reflect mode
        if mode == 'image_text':
            dataset = f"{dataset}_image_text"
        else:
            dataset = f"{dataset}_image_only"
        
        # Call parent init (will call load_data)
        super().__init__(dataset=dataset, **kwargs)
    
    def load_data(self, dataset):
        """Load CLEVR questions and metadata."""
        print(f"Loading CLEVR {self.split} questions from {self.questions_file}...")
        
        with open(self.questions_file, 'r') as f:
            questions_data = json.load(f)
        
        data_list = []
        for q in questions_data['questions']:
            item = {
                'index': q['question_index'],
                'question': q['question'],
                'image_filename': q['image_filename'],
                'image': str(self.images_dir / q['image_filename']),
                'image_path': str(self.images_dir / q['image_filename']),
                'image_index': q['image_index'],
                'split': q.get('split', self.split),
            }
            
            # Add answer if available (val/train only)
            if 'answer' in q:
                item['answer'] = str(q['answer']).lower()
            
            # Add program length (for analysis by reasoning steps)
            if 'program' in q:
                item['program_length'] = len(q['program'])
                item['program'] = json.dumps(q['program'])  # Store as JSON string
            else:
                item['program_length'] = 0
                item['program'] = None
            
            # Add question family
            if 'question_family_index' in q:
                item['question_family_index'] = q['question_family_index']
                item['question_family'] = get_question_family(q['question_family_index'])
            
            # Add scene description for image_text mode
            if self.mode == 'image_text' and q['image_filename'] in self.scenes_by_image:
                scene = self.scenes_by_image[q['image_filename']]
                item['scene_description'] = self._format_scene_description(scene)
            else:
                item['scene_description'] = ""
            
            data_list.append(item)
        
        df = pd.DataFrame(data_list)
        
        # Limit samples if requested
        if self.num_samples is not None and self.num_samples > 0:
            df = df.head(self.num_samples)
            print(f"Limited to {len(df)} samples")
        
        print(f"Loaded {len(df)} questions")
        print(f"Mode: {self.mode}")
        print(f"Program length range: {df['program_length'].min()} - {df['program_length'].max()}")
        
        return df
    
    def _format_scene_description(self, scene: Dict[str, Any]) -> str:
        """
        Format scene metadata into text description following the paper's format.
        
        This follows the exact format from the paper's appendix (section A-D2).
        """
        lines = []
        
        # Scene header
        image_index = scene.get('image_index', 0)
        lines.append(f"Scene {image_index}:")
        lines.append("")
        
        # Objects section
        objects = scene.get('objects', [])
        lines.append(f"Objects: {len(objects)}")
        
        for obj in objects:
            obj_lines = []
            obj_lines.append("Object:")
            obj_lines.append(f"Color: {obj.get('color', 'unknown')}")
            obj_lines.append(f"Size: {obj.get('size', 'unknown')}")
            obj_lines.append(f"Rotation: {obj.get('rotation', 0)}")
            obj_lines.append(f"Shape: {obj.get('shape', 'unknown')}")
            obj_lines.append(f"Material: {obj.get('material', 'unknown')}")
            
            if '3d_coords' in obj:
                coords = obj['3d_coords']
                obj_lines.append(f"3D Coords: [{coords[0]}, {coords[1]}, {coords[2]}]")
            
            if 'pixel_coords' in obj:
                px_coords = obj['pixel_coords']
                obj_lines.append(f"Pixel Coords: [{px_coords[0]}, {px_coords[1]}, {px_coords[2]}]")
            
            lines.append(" ".join(obj_lines))
        
        lines.append("")
        
        # Relationships section
        relationships = scene.get('relationships', {})
        if relationships:
            rel_strs = []
            for rel_name, rel_data in relationships.items():
                rel_strs.append(f"'{rel_name}': {rel_data}")
            lines.append(f"Relationships: {', '.join(rel_strs)}")
            lines.append("")
        
        # Directions section
        directions = scene.get('directions', {})
        if directions:
            dir_strs = []
            for dir_name, dir_data in directions.items():
                dir_strs.append(f"'{dir_name}': {dir_data}")
            lines.append(f"Directions: {', '.join(dir_strs)}")
            lines.append("")
        
        # Image filename
        lines.append(f"Image Filename: {scene.get('image_filename', '')}")
        
        return "\n".join(lines)
    
    def build_prompt(self, line):
        """
        Build the prompt for a single question.
        
        Follows the paper's prompt format:
        - For image_text mode: scene description + setup prompt + question
        - For image_only mode: setup prompt + question (with image)
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # Get image path
        image_path = line['image_path']
        
        # Build text prompt
        prompt_parts = []
        
        # Add scene description for image_text mode
        if self.mode == 'image_text' and line.get('scene_description'):
            prompt_parts.append("Given the following scene:")
            prompt_parts.append("")
            prompt_parts.append(line['scene_description'])
            prompt_parts.append("")
        
        # Add setup prompt (vocabulary constraints)
        prompt_parts.append(self.SETUP_PROMPT)
        prompt_parts.append("")
        
        # Add question
        prompt_parts.append(f"Question: {line['question']}")
        prompt_parts.append("Answer:")
        
        prompt_text = "\n".join(prompt_parts)
        
        # Build message list
        msgs = [
            dict(type='image', value=image_path),
            dict(type='text', value=prompt_text)
        ]
        
        return msgs
    
    def evaluate(self, eval_file: str, **judge_kwargs):
        """
        Evaluate predictions against ground truth.
        
        For CLEVR, we do exact string matching (case-insensitive).
        Also computes accuracy by program length and question family.
        """
        from vlmeval.smp import load, dump
        
        # Load predictions
        preds = load(eval_file)
        if isinstance(preds, str):
            preds = pd.read_json(preds, lines=True)
        
        # Check if ground truth answers are available
        required_cols = ['index', 'answer', 'program_length', 'question_family']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        
        if missing_cols:
            print(f"\nWarning: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(self.data.columns)}")
            return pd.DataFrame({
                'split': [self.split],
                'note': [f'Missing columns: {missing_cols}'],
                'num_predictions': [len(preds)]
            })
        
        # Merge with ground truth
        # Note: If preds already has 'answer' column, we need to handle column name conflicts
        gt_data = self.data[['index', 'answer', 'program_length', 'question_family']].copy()
        merged = preds.merge(gt_data, on='index', how='left', suffixes=('_pred', '_gt'))
        
        # Determine which answer column to use (prefer _gt, fallback to _pred or just 'answer')
        if 'answer_gt' in merged.columns:
            answer_col = 'answer_gt'
        elif 'answer' in merged.columns:
            answer_col = 'answer'
        elif 'answer_pred' in merged.columns:
            answer_col = 'answer_pred'
        else:
            print("\nWarning: No answer column found after merge")
            print(f"Available columns: {list(merged.columns)}")
            return pd.DataFrame({
                'split': [self.split],
                'note': ['No answer column found after merge'],
                'num_predictions': [len(preds)]
            })
        
        # Check if answers are available
        if merged[answer_col].isna().all():
            print(f"\nWarning: All answers are NaN in column '{answer_col}'")
            return pd.DataFrame({
                'split': [self.split],
                'note': ['All ground truth answers are missing'],
                'num_predictions': [len(preds)]
            })
        
        # Extract predicted answer (take first word, lowercase)
        def extract_answer(pred):
            if pd.isna(pred) or pred == '':
                return ''
            # Take the first word and lowercase
            pred_str = str(pred).strip().lower()
            # Handle common model output formats
            if ':' in pred_str:
                pred_str = pred_str.split(':')[-1].strip()
            # Take first word
            pred_str = pred_str.split()[0] if pred_str else ''
            # Remove punctuation
            pred_str = ''.join(c for c in pred_str if c.isalnum())
            return pred_str
        
        merged['pred_extracted'] = merged['prediction'].apply(extract_answer)
        merged['gt_normalized'] = merged[answer_col].apply(lambda x: str(x).lower().strip())
        
        # Determine which program_length and question_family columns to use
        if 'program_length_gt' in merged.columns:
            program_length_col = 'program_length_gt'
        elif 'program_length' in merged.columns:
            program_length_col = 'program_length'
        else:
            program_length_col = 'program_length_pred'
            
        if 'question_family_gt' in merged.columns:
            question_family_col = 'question_family_gt'
        elif 'question_family' in merged.columns:
            question_family_col = 'question_family'
        else:
            question_family_col = 'question_family_pred'
        
        # Compute hit (correct prediction)
        merged['hit'] = merged['pred_extracted'] == merged['gt_normalized']
        
        # Overall accuracy
        overall_acc = merged['hit'].mean() * 100
        
        # Accuracy by program length
        acc_by_length = merged.groupby(program_length_col)['hit'].agg(['mean', 'count'])
        acc_by_length.columns = ['accuracy', 'count']
        acc_by_length['accuracy'] = acc_by_length['accuracy'] * 100
        
        # Accuracy by question family
        acc_by_family = merged.groupby(question_family_col)['hit'].agg(['mean', 'count'])
        acc_by_family.columns = ['accuracy', 'count']
        acc_by_family['accuracy'] = acc_by_family['accuracy'] * 100
        
        # Build results DataFrame
        results = pd.DataFrame({
            'split': [self.split],
            'Overall': [overall_acc]
        })
        
        # Add family-specific accuracies
        for family in acc_by_family.index:
            results[f'Family_{family}'] = [acc_by_family.loc[family, 'accuracy']]
        
        # Save detailed results
        eval_dir = Path(eval_file).parent
        
        # Save accuracy by program length
        length_results_file = eval_dir / 'accuracy_by_program_length.csv'
        acc_by_length.to_csv(length_results_file)
        print(f"Saved accuracy by program length to {length_results_file}")
        
        # Save accuracy by question family
        family_results_file = eval_dir / 'accuracy_by_question_family.csv'
        acc_by_family.to_csv(family_results_file)
        print(f"Saved accuracy by question family to {family_results_file}")
        
        # Save detailed predictions with hits
        detailed_file = eval_file.replace('.xlsx', '_detailed.xlsx').replace('.csv', '_detailed.csv')
        # Select columns that actually exist in the merged dataframe
        cols_to_save = ['index', 'prediction', 'pred_extracted', answer_col, 'gt_normalized', 'hit', 
                        program_length_col, question_family_col]
        # Rename back to original names for clarity
        merged_to_save = merged[cols_to_save].copy()
        merged_to_save.columns = ['index', 'prediction', 'pred_extracted', 'answer', 'answer_normalized', 'hit',
                                   'program_length', 'question_family']
        dump(merged_to_save, detailed_file)
        print(f"Saved detailed predictions to {detailed_file}")
        
        print(f"\n{'='*50}")
        print(f"CLEVR Evaluation Results ({self.mode} mode)")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"\nAccuracy by Question Family:")
        for family, row in acc_by_family.iterrows():
            print(f"  {family}: {row['accuracy']:.2f}% (n={int(row['count'])})")
        print(f"\nAccuracy by Program Length (reasoning steps):")
        for length in sorted(acc_by_length.index)[:10]:  # Show first 10
            row = acc_by_length.loc[length]
            print(f"  Length {length}: {row['accuracy']:.2f}% (n={int(row['count'])})")
        if len(acc_by_length) > 10:
            print(f"  ... (see {length_results_file} for full results)")
        print(f"{'='*50}\n")
        
        return results


class CLEVRTestDataset(CLEVRDataset):
    """
    CLEVR test split dataset loader (without ground truth answers).
    
    The test split does not have answers or scene descriptions,
    so only image-only mode is supported.
    """
    
    def __init__(
        self,
        dataset: str = 'CLEVR_test',
        dataset_path: Optional[str] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """Initialize CLEVR test dataset loader."""
        # Force image_only mode for test split (no scene descriptions available)
        super().__init__(
            dataset=dataset,
            dataset_path=dataset_path,
            split='test',
            mode='image_only',
            num_samples=num_samples,
            **kwargs
        )
    
    def evaluate(self, eval_file: str, **judge_kwargs):
        """
        For test split, we can't compute accuracy (no ground truth).
        Just save the predictions.
        """
        print("\nCLEVR Test Split - No ground truth available for evaluation")
        print(f"Predictions saved to: {eval_file}")
        
        return pd.DataFrame({
            'split': ['test'],
            'note': ['No ground truth available'],
            'num_predictions': [len(self.data)]
        })


class CLEVRImageOnlyDataset(CLEVRDataset):
    """Convenience class for image-only mode evaluation."""
    
    def __init__(self, dataset_path: Optional[str] = None, split: str = 'val', **kwargs):
        super().__init__(
            dataset='CLEVR',
            dataset_path=dataset_path,
            split=split,
            mode='image_only',
            **kwargs
        )


class CLEVRImageTextDataset(CLEVRDataset):
    """Convenience class for image+text mode evaluation."""
    
    def __init__(self, dataset_path: Optional[str] = None, split: str = 'val', **kwargs):
        super().__init__(
            dataset='CLEVR',
            dataset_path=dataset_path,
            split=split,
            mode='image_text',
            **kwargs
        )
