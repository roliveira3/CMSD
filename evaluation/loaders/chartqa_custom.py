"""
Custom ChartQA dataset loader with support for CSV tables and/or annotations.

This loader extends the base ChartQA functionality to test whether including:
1. CSV tables (raw data)
2. Annotations (chart metadata and structure)
3. Both CSV and annotations

improves model accuracy on ChartQA questions.

Usage:
    # Test with CSV only
    python evaluation/eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader ChartQACustom \
        --dataset_path /cluster/scratch/rbertolissi/datasets/ChartQA_Dataset/test \
        --no_probing

    # Configuration via environment variables:
    export CHARTQA_INCLUDE_CSV=true
    export CHARTQA_INCLUDE_ANNOTATIONS=true
    
    Or pass as dataset arguments in the loader instantiation.
"""

import pandas as pd
import json
import csv
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
from vlmeval.dataset.image_base import ImageBaseDataset
from functools import partial
import multiprocessing as mp
import numpy as np


class ChartQACustomDataset(ImageBaseDataset):
    """
    Custom ChartQA dataset loader that can include CSV tables and/or annotations
    in the prompt to test their impact on model accuracy.
    
    Supports three test modes:
    1. include_csv=True, include_annotations=False: Add raw CSV data
    2. include_csv=False, include_annotations=True: Add chart structure metadata
    3. include_csv=True, include_annotations=True: Add both
    
    The dataset path should point to the test directory containing:
    - test_human.json and test_augmented.json (questions)
    - png/ directory (chart images)
    - tables/ directory (CSV files with raw data)
    - annotations/ directory (JSON files with chart structure)
    """
    
    TYPE = 'VQA'
    
    def __init__(
        self,
        dataset: str = 'ChartQA_Custom',
        dataset_path: Optional[str] = None,
        include_csv: bool = False,
        include_annotations: bool = False,
        csv_rows_limit: int = None,  # Limit CSV rows to avoid token overflow
        csv_to_text_format: str = "markdwon",  # "csv" or "markdown"
        **kwargs
    ):
        """
        Initialize custom ChartQA loader.
        
        Args:
            dataset: Dataset name
            dataset_path: Path to ChartQA test directory
            include_csv: Whether to include CSV table data in prompts
            include_annotations: Whether to include chart annotations in prompts
            csv_rows_limit: Maximum number of CSV rows to include (None = all)
        """
        self.include_csv = include_csv or os.getenv('CHARTQA_INCLUDE_CSV', 'false').lower() == 'true'
        self.include_annotations = include_annotations or os.getenv('CHARTQA_INCLUDE_ANNOTATIONS', 'false').lower() == 'true'
        self.csv_rows_limit = csv_rows_limit
        self.csv_to_text_format = csv_to_text_format
        
        if dataset_path is None:
            raise ValueError("dataset_path is required for ChartQACustom")
        
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Set up paths to data directories
        self.tables_dir = self.dataset_path / 'tables'
        self.annotations_dir = self.dataset_path / 'annotations'
        self.images_dir = self.dataset_path / 'png'
        
        # Validate directories exist if needed
        if self.include_csv and not self.tables_dir.exists():
            raise FileNotFoundError(f"Tables directory not found: {self.tables_dir}")
        if self.include_annotations and not self.annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory not found: {self.annotations_dir}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Call parent init which will call load_data
        super().__init__(dataset=dataset, **kwargs)
    
    def load_data(self, dataset):
        """Load ChartQA data from JSON files."""
        # Load both human and augmented questions
        human_file = self.dataset_path / 'test_human.json'
        augmented_file = self.dataset_path / 'test_augmented.json'
        
        data_list = []
        
        if human_file.exists():
            with open(human_file, 'r') as f:
                human_data = json.load(f)
                for item in human_data:
                    item['split'] = 'test_human'
                    data_list.append(item)
        
        if augmented_file.exists():
            with open(augmented_file, 'r') as f:
                augmented_data = json.load(f)
                for item in augmented_data:
                    item['split'] = 'test_augmented'
                    data_list.append(item)
        
        if not data_list:
            raise ValueError(f"No data found in {self.dataset_path}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'imgname': 'image_filename',  # Keep original for reference
            'query': 'question',
            'label': 'answer'
        })
        
        # Add full image paths
        # The parent class can work with image_path instead of base64 encoded images
        df['image_path'] = df['image_filename'].apply(lambda x: str(self.images_dir / x))
        
        # For VLMEvalKit compatibility, we need to set image column to image_path
        # The parent class will handle loading images from paths during inference
        df['image'] = df['image_path']
        
        # Add index
        df['index'] = range(len(df))
        
        # Load and attach CSV/annotation data if requested
        if self.include_csv or self.include_annotations:
            df['context'] = df['image_filename'].apply(self._load_context) + "Given the image, as well as the additional information, now answer the question below"
        
        return df
    
    def _load_context(self, imgname: str) -> str:
        """
        Load additional context (CSV and/or annotations) for an image.
        
        Args:
            imgname: Image filename (e.g., '01499440003158.png')
            
        Returns:
            Formatted context string to prepend to the question
        """
        context_parts = ["Below we give some additional infos on the provided chart given in the image:"]
        
        # Extract base name without extension
        base_name = Path(imgname).stem
        
        
        # Load CSV table if requested
        if self.include_csv:
            csv_file = self.tables_dir / f"{base_name}.csv"
            if csv_file.exists():
                csv_context = self._format_csv(csv_file)
                if csv_context:
                    csv_context_intro = f"The data described by the chart can be summarised by the following {'markdown' if self.csv_to_text_format == 'markdown' else 'CSV'} table, the first row describes the header, the later ones the actual data:"
                    context_parts.append(csv_context_intro + "\n" + csv_context)
        
        # Load annotations if requested
        if self.include_annotations:
            ann_file = self.annotations_dir / f"{base_name}.json"
            if ann_file.exists():
                ann_context = self._format_annotations(ann_file)
                if ann_context:
                    structure_text_intro = "The structure of the chart can be roughly summarised as follows. We describe the chart type, the title, the axes labels and the data series/bars represented in the chart:"
                    context_parts.append(structure_text_intro + "\n" + ann_context)
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _format_csv(self, csv_file: Path) -> str:
        """
        Format CSV table data for inclusion in prompt.
        
        Args:
            csv_file: Path to CSV file
            format: Output format - "csv" (default) or "markdown"
            
        Returns:
            Formatted CSV string
        """
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                # Limit rows if specified
                if self.csv_rows_limit and len(rows) > self.csv_rows_limit + 1:  # +1 for header
                    rows = [rows[0]] + rows[1:self.csv_rows_limit + 1]
                    rows.append(['...', '(truncated)'])
                
                if len(rows) < 2:
                    print("WARNING: We do not have csv data")
                    return ""
                
                if self.csv_to_text_format == "markdown":
                    # Create markdown table
                    lines = []
                    lines.append("| " + " | ".join(rows[0]) + " |")
                    lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
                    for row in rows[1:]:
                        if len(row) == len(rows[0]):  # Ensure same number of columns
                            lines.append("| " + " | ".join(row) + " |")
                        else:
                            print(f"Warning: Skipping malformed row in {csv_file}: {row}")
                    return "\n".join(lines)
                else:  # Default to "csv"
                    # Return as CSV format
                    lines = []
                    for row in rows:
                        lines.append(",".join(row))
                    return "\n".join(lines)
        except Exception as e:
            print(f"Warning: Could not read CSV {csv_file}: {e}")
            return ""
    
    def _format_annotations(self, ann_file: Path) -> str:
        """
        Format chart annotations for inclusion in prompt.
        
        Args:
            ann_file: Path to annotation JSON file
            
        Returns:
            Formatted annotation string
        """
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
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
            print(f"Warning: Could not read annotations {ann_file}: {e}")
            return ""
    
    def build_prompt(self, line):
        """Build prompt with optional context from CSV/annotations."""
        # Get base prompt from parent class
        msgs = super().build_prompt(line)
        
        # Find the text message (last one)
        assert msgs[-1]['type'] == 'text'
        
        # Prepend context if available
        if 'context' in line and line['context']:
            # Insert context before the question
            original_text = msgs[-1]['value']
            msgs[-1]['value'] = line['context'] + "\n\n" + original_text
        
        # Add instruction for short answers (matching ChartQA evaluation)
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        
        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate using ChartQA's relaxed accuracy metric.
        
        This matches VLMEvalKit's standard ChartQA evaluation.
        """
        from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line
        from vlmeval.smp import load, dump
        from vlmeval.smp.file import get_intermediate_file_path
        
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        
        # Ensure strings
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        
        # Evaluate using relaxed accuracy (ChartQA's metric)
        lt = len(data)
        pool = mp.Pool(16)
        lines = [data.iloc[i] for i in range(lt)]
        res = pool.map(partial(process_line, method='relaxed_accuracy'), lines)
        
        # Store evaluation results
        data['eval_gt'] = [r['gt'] for r in res]
        data['eval_pred'] = [r['pred'] for r in res]
        data['eval_match'] = [r['match'] for r in res]
        data['eval_score'] = [np.mean(r['match']) for r in res]
        
        # Save detailed results
        detailed_result_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detailed_result_file)
        
        # Calculate accuracy by split (test_human vs test_augmented)
        # hit_calculate returns a list of per-sample scores, we need the mean as percentage
        hit = hit_calculate(res, self.dataset_name)
        ret = dict()
        
        if 'split' in data:
            splits = set(data['split'])
            for sp in splits:
                sub_data = data[data['split'] == sp]
                sub_res = [res[i] for i in range(len(data)) if data.iloc[i]['split'] == sp]
                hit_sp = hit_calculate(sub_res, self.dataset_name)
                # Compute mean accuracy as percentage (wrap in list to match expected format)
                ret[sp] = [np.mean(hit_sp) * 100]
        
        # Compute overall accuracy as percentage (wrap in list to match expected format)
        ret['Overall'] = [np.mean(hit) * 100]
        return ret


# Convenience classes for specific configurations
class ChartQAWithCSV(ChartQACustomDataset):
    """ChartQA with CSV tables only."""
    def __init__(self, dataset: str = 'ChartQA_CSV', dataset_path: Optional[str] = None, **kwargs):
        # Always override dataset name to make it unique - append suffix to whatever is passed
        # This ensures uniqueness even when eval script passes "ChartQA_Dataset"
        if not dataset.endswith('_CSV'):
            dataset = f'{dataset}_CSV'
        super().__init__(dataset=dataset, dataset_path=dataset_path, 
                        include_csv=True, include_annotations=False, **kwargs)


class ChartQAWithAnnotations(ChartQACustomDataset):
    """ChartQA with chart annotations only."""
    def __init__(self, dataset: str = 'ChartQA_Annotations', dataset_path: Optional[str] = None, **kwargs):
        # Always override dataset name to make it unique - append suffix to whatever is passed
        # This ensures uniqueness even when eval script passes "ChartQA_Dataset"
        if not dataset.endswith('_Annotations'):
            dataset = f'{dataset}_Annotations'
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        include_csv=False, include_annotations=True, **kwargs)


class ChartQAWithBoth(ChartQACustomDataset):
    """ChartQA with both CSV tables and annotations."""
    def __init__(self, dataset: str = 'ChartQA_Both', dataset_path: Optional[str] = None, **kwargs):
        # Always override dataset name to make it unique - append suffix to whatever is passed
        # This ensures uniqueness even when eval script passes "ChartQA_Dataset"
        if not dataset.endswith('_Both'):
            dataset = f'{dataset}_Both'
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        include_csv=True, include_annotations=True, **kwargs)
