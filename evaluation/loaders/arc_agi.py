"""
ARC-AGI v1 dataset loader for abstract reasoning evaluation.

This loader implements evaluation for the Abstraction and Reasoning Corpus (ARC-AGI).
ARC-AGI is a benchmark designed to measure a human-like form of general fluid intelligence.

The benchmark consists of tasks where each task has:
- Training examples: input-output grid pairs demonstrating the transformation
- Test examples: input grids for which the model must produce the output

Key features:
- Pure text-based reasoning (no images required, though grids can be visualized)
- Supports multiple text representations (simple ASCII, spreadsheet-like)
- Supports reasoning mode toggle for InternVL3.5 models
- Exact match evaluation (all cells must match)

Reference: "On the Measure of Intelligence" (https://arxiv.org/abs/1911.01547)

Usage:
    # Standard evaluation

    # With reasoning mode enabled
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader ARCAGI \
        --dataset_path /cluster/scratch/rbertolissi/datasets/ARC-AGI \
        --arc_reasoning_mode

    # Evaluation split (default)
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader ARCAGI \
        --dataset_path /cluster/scratch/rbertolissi/datasets/ARC-AGI \
        --arc_split evaluation
"""

import pandas as pd
import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple
from vlmeval.dataset.image_base import ImageBaseDataset
import warnings


# ARC-AGI color names for visualization (0-9)
ARC_COLOR_NAMES = {
    0: "black",
    1: "blue", 
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey",
    6: "pink",
    7: "orange",
    8: "purple",
    9: "brown"
}

# Spreadsheet column labels for ASCII representation
SPREADSHEET_COLS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["AA", "AB", "AC", "AD"]


def grid_to_ascii(grid: List[List[int]], use_spreadsheet: bool = False, separator: str = "|") -> str:
    """
    Convert a grid to ASCII text representation.
    
    Args:
        grid: 2D list of integers (0-9)
        use_spreadsheet: If True, use spreadsheet-like notation (A1, B2, etc.)
        separator: Column separator character
        
    Returns:
        String representation of the grid
    """
    if not grid or not grid[0]:
        return "[Empty grid]"
    
    rows = len(grid)
    cols = len(grid[0])
    
    if use_spreadsheet:
        # Spreadsheet format with column headers
        header = separator.join([" "] + SPREADSHEET_COLS[:cols])
        lines = [header]
        for i, row in enumerate(grid):
            row_str = separator.join([str(i + 1)] + [str(x) for x in row])
            lines.append(row_str)
        return "\n".join(lines)
    else:
        # Simple format
        return "\n".join(separator.join(str(x) for x in row) for row in grid)


def grid_to_json_str(grid: List[List[int]]) -> str:
    """Convert grid to JSON string representation."""
    return json.dumps(grid)


def parse_grid_from_response(response: str) -> Optional[List[List[int]]]:
    """
    Parse a grid from model response.
    
    Attempts multiple parsing strategies:
    1. JSON array format
    2. Simple row-by-row format
    3. Spreadsheet format
    
    Args:
        response: Model response string
        
    Returns:
        Parsed grid or None if parsing fails
    """
    if not response:
        return None
    
    # Clean up response
    response = response.strip()
    
    # Strategy 1: Try JSON parsing
    # Look for JSON array patterns
    json_patterns = [
        r'\[\s*\[[\d\s,\[\]]+\]\s*\]',  # Standard JSON array
        r'```json\s*(\[\s*\[[\d\s,\[\]]+\]\s*\])\s*```',  # JSON in code block
        r'```\s*(\[\s*\[[\d\s,\[\]]+\]\s*\])\s*```',  # Generic code block
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                grid = json.loads(json_str)
                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    # Validate that all elements are integers 0-9
                    valid = True
                    for row in grid:
                        for val in row:
                            if not isinstance(val, int) or val < 0 or val > 9:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        return grid
            except (json.JSONDecodeError, TypeError):
                pass
    
    # Strategy 2: Try parsing row-by-row format
    # Look for lines of numbers
    lines = response.split('\n')
    grid = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove common prefixes/separators
        line = re.sub(r'^[\d]+[\s\|:]+', '', line)  # Remove row numbers
        line = re.sub(r'[\[\](),]', ' ', line)  # Remove brackets
        
        # Extract numbers
        numbers = re.findall(r'\b[0-9]\b', line)
        if numbers:
            row = [int(n) for n in numbers]
            if row:
                grid.append(row)
    
    # Validate grid dimensions
    if grid and len(grid) > 0:
        # Check if all rows have the same length
        row_lengths = [len(row) for row in grid]
        if len(set(row_lengths)) == 1 and row_lengths[0] > 0:
            return grid
    
    return None


class ARCAGIDataset(ImageBaseDataset):
    """
    ARC-AGI dataset loader for abstract reasoning evaluation.
    
    This loader handles the Abstraction and Reasoning Corpus, presenting
    the tasks as text-based reasoning problems.
    
    Args:
        dataset: Dataset name
        dataset_path: Path to ARC-AGI directory (should contain data/evaluation/ and data/training/)
        split: Which split to use ('evaluation' or 'training')
        representation: Grid representation format ('simple', 'spreadsheet', 'json')
        reasoning_mode: Enable extended reasoning prompts
        num_samples: Limit number of samples (for debugging)
    """
    
    TYPE = 'VQA'  # Treat as VQA-style task
    MODALITY = 'TEXT'  # Text-only dataset, no images
    
    # System prompt providing context about the task
    SYSTEM_PROMPT = """You are solving abstract reasoning puzzles from the ARC-AGI benchmark.

Each puzzle shows you some training examples where an input grid is transformed into an output grid.
Your task is to understand the transformation pattern and apply it to a new test input.

Grids are 2D arrays of integers from 0-9, where each number represents a color:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=pink, 7=orange, 8=purple, 9=brown

Important rules:
- The output grid dimensions may differ from the input
- Study all training examples carefully to identify the pattern
- Apply the exact same transformation rule to the test input
- Return ONLY the output grid as a JSON array of arrays (e.g., [[1,2],[3,4]])"""

    REASONING_SYSTEM_PROMPT = """You are solving abstract reasoning puzzles from the ARC-AGI benchmark.

Each puzzle shows you some training examples where an input grid is transformed into an output grid.
Your task is to understand the transformation pattern and apply it to a new test input.

Grids are 2D arrays of integers from 0-9, where each number represents a color:
0=black, 1=blue, 2=red, 3=green, 4=yellow, 5=grey, 6=pink, 7=orange, 8=purple, 9=brown

Follow this reasoning process:
1. <reasoning> Carefully analyze each training example:
   - What objects/patterns exist in the input?
   - How do they change in the output?
   - What is preserved vs modified?
   - What is the transformation rule?
2. Verify your hypothesis against ALL training examples
3. Apply the transformation to the test input </reasoning>
4. Return ONLY the output grid as a JSON array of arrays

Important rules:
- The output grid dimensions may differ from the input
- Study all training examples carefully to identify the pattern  
- Apply the exact same transformation rule to the test input
- Think step by step before giving your final answer"""

    def __init__(
        self,
        dataset: str = 'ARC-AGI',
        dataset_path: Optional[str] = None,
        split: str = 'evaluation',
        representation: str = 'simple',  # 'simple', 'spreadsheet', 'json'
        reasoning_mode: bool = False,
        num_samples: Optional[int] = None,
        max_train_examples: Optional[int] = None,  # Limit training examples in prompt
        verbose: bool = True,  # Print prompts and responses
        **kwargs
    ):
        """
        Initialize ARC-AGI dataset loader.
        
        Args:
            dataset: Dataset name
            dataset_path: Path to ARC-AGI directory
            split: 'evaluation' or 'training'
            representation: Grid representation format
            reasoning_mode: Enable extended reasoning prompts
            num_samples: Limit number of samples (None = all)
            max_train_examples: Limit training examples shown in prompt (None = all)
        """
        if dataset_path is None:
            raise ValueError("dataset_path is required for ARC-AGI dataset")
        
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.representation = representation
        self.max_train_examples = max_train_examples
        self.verbose = verbose
        self._sample_counter = 0  # Track which sample we're on
        self.reasoning_mode = reasoning_mode
        self.num_samples = num_samples
        
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"ARC-AGI dataset not found: {dataset_path}")
        
        # Set up paths
        self.data_dir = self.dataset_path / "data" / split
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.data_dir}")
        
        # Update dataset name to reflect mode
        suffix = "_reasoning" if reasoning_mode else ""
        dataset = f"{dataset}_{split}{suffix}"
        
        # Call parent init (will call load_data)
        # skip_noimg=True because ARC-AGI is text-only, no images
        super().__init__(dataset=dataset, skip_noimg=True, **kwargs)
    
    def load_data(self, dataset):
        """Load ARC-AGI tasks."""
        print(f"Loading ARC-AGI {self.split} tasks from {self.data_dir}...")
        
        # Find all JSON task files
        task_files = sorted(self.data_dir.glob("*.json"))
        
        if not task_files:
            raise FileNotFoundError(f"No task files found in {self.data_dir}")
        
        data_list = []
        for task_idx, task_file in enumerate(task_files):
            task_id = task_file.stem
            
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            train_examples = task_data.get('train', [])
            test_examples = task_data.get('test', [])
            
            # Create one entry per test example
            for test_idx, test_example in enumerate(test_examples):
                item = {
                    'index': f"{task_id}_{test_idx}",
                    'task_id': task_id,
                    'test_index': test_idx,
                    'num_train_examples': len(train_examples),
                    'num_test_examples': len(test_examples),
                    # Store grids as JSON strings
                    'train_examples': json.dumps(train_examples),
                    'test_input': json.dumps(test_example['input']),
                    'input_shape': f"{len(test_example['input'])}x{len(test_example['input'][0]) if test_example['input'] else 0}",
                    # Ground truth (if available)
                    'answer': json.dumps(test_example.get('output', None)),
                    'has_answer': 'output' in test_example,
                    # Build the question text
                    'question': self._build_question(train_examples, test_example['input']),
                    # No image field - ARC-AGI is text-only
                }
                
                # Add output shape if available
                if 'output' in test_example:
                    item['output_shape'] = f"{len(test_example['output'])}x{len(test_example['output'][0]) if test_example['output'] else 0}"
                
                data_list.append(item)
        
        df = pd.DataFrame(data_list)
        
        # Limit samples if requested
        if self.num_samples is not None and self.num_samples > 0:
            df = df.head(self.num_samples)
            print(f"Limited to {len(df)} samples")
        
        print(f"Loaded {len(df)} test cases from {len(task_files)} tasks")
        print(f"Representation mode: {self.representation}")
        print(f"Reasoning mode: {self.reasoning_mode}")
        
        return df
    
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid according to the chosen representation."""
        if self.representation == 'json':
            return grid_to_json_str(grid)
        elif self.representation == 'spreadsheet':
            return grid_to_ascii(grid, use_spreadsheet=True)
        else:  # simple
            return grid_to_ascii(grid, use_spreadsheet=False)
    
    def _build_question(self, train_examples: List[Dict], test_input: List[List[int]]) -> str:
        """
        Build the question text for a task.
        
        Args:
            train_examples: List of training input/output pairs
            test_input: The test input grid
            
        Returns:
            Formatted question string
        """
        parts = []
        
        # Limit training examples if requested
        if self.max_train_examples is not None and self.max_train_examples > 0:
            train_examples = train_examples[:self.max_train_examples]
        
        # Add training examples
        parts.append("## Training Examples\n")
        for i, example in enumerate(train_examples, 1):
            parts.append(f"### Example {i}")
            parts.append(f"Input ({len(example['input'])}x{len(example['input'][0])} grid):")
            parts.append(self._format_grid(example['input']))
            parts.append(f"\nOutput ({len(example['output'])}x{len(example['output'][0])} grid):")
            parts.append(self._format_grid(example['output']))
            parts.append("")
        
        # Add test input
        parts.append("## Test Input")
        parts.append(f"Input ({len(test_input)}x{len(test_input[0])} grid):")
        parts.append(self._format_grid(test_input))
        parts.append("")
        
        # Add instruction
        parts.append("What is the output grid for this test input?")
        parts.append("Return your answer as a JSON array of arrays (e.g., [[1,2,3],[4,5,6]]).")
        
        return "\n".join(parts)
    
    def build_prompt(self, line):
        """
        Build the prompt for a single task.
        
        Since ARC-AGI is text-only, we don't include images.
        The prompt includes the system context and the task question.
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # Increment counter and print progress
        self._sample_counter += 1
        
        # Choose system prompt based on reasoning mode
        system_prompt = self.REASONING_SYSTEM_PROMPT if self.reasoning_mode else self.SYSTEM_PROMPT
        
        # Build the full prompt
        question = line['question']
        full_prompt = f"{system_prompt}\n\n{question}"
        
        # Print verbose information
        if self.verbose:
            print("\n" + "="*80)
            print(f"ARC-AGI SAMPLE {self._sample_counter}/{len(self.data)}")
            print("="*80)
            print(f"Task ID: {line['task_id']}")
            print(f"Test Index: {line['test_index']}")
            print(f"Input Shape: {line['input_shape']}")
            if 'output_shape' in line:
                print(f"Expected Output Shape: {line['output_shape']}")
            print(f"Num Training Examples Shown: {line['num_train_examples'] if self.max_train_examples is None else min(line['num_train_examples'], self.max_train_examples)}")
            print(f"Total Num Training Examples: {line['num_train_examples']}")
            
            # Print expected solution if available
            if line.get('has_answer', False) and line.get('answer'):
                try:
                    expected_output = json.loads(line['answer']) if isinstance(line['answer'], str) else line['answer']
                    print(f"\n--- EXPECTED SOLUTION (Ground Truth) ---")
                    # Format as compact JSON with each row on one line
                    formatted_rows = [json.dumps(row) for row in expected_output]
                    formatted_output = "[\n  " + ",\n  ".join(formatted_rows) + "\n]"
                    print(formatted_output)
                except (json.JSONDecodeError, TypeError):
                    print(f"\n--- EXPECTED SOLUTION (Ground Truth) ---")
                    print(line['answer'])
            else:
                print(f"\n--- EXPECTED SOLUTION: Not available ---")
            
            print(f"\n--- FULL PROMPT ---")
            print(full_prompt)
            print(f"\n--- End of prompt (length: {len(full_prompt)} characters) ---")
            print("\nGenerating model response...")
            import sys
            sys.stdout.flush()
        
        # Create message structure for VLMEvalKit
        # Note: No image since ARC-AGI is text-only
        msgs = [
            dict(type='text', value=full_prompt)
        ]
        
        return msgs
    
    def evaluate(self, eval_file: str, **judge_kwargs):
        """
        Evaluate predictions against ground truth.
        
        For ARC-AGI, we require exact match of all grid cells.
        """
        from vlmeval.smp import load, dump
        
        # Load predictions
        preds = load(eval_file)
        if isinstance(preds, str):
            preds = pd.read_json(preds, lines=True)
        
        # Check if ground truth is available
        if not self.data['has_answer'].any():
            print("\nARC-AGI - No ground truth available for evaluation")
            print(f"Predictions saved to: {eval_file}")
            return pd.DataFrame({
                'split': [self.split],
                'note': ['No ground truth available'],
                'num_predictions': [len(preds)]
            })
        
        # Merge with ground truth
        gt_data = self.data[['index', 'answer', 'task_id', 'num_train_examples']].copy()
        merged = preds.merge(gt_data, on='index', how='left', suffixes=('_pred', '_gt'))
        
        # Determine answer column
        if 'answer_gt' in merged.columns:
            answer_col = 'answer_gt'
        elif 'answer' in merged.columns:
            answer_col = 'answer'
        else:
            print("\nWarning: No answer column found")
            return pd.DataFrame({
                'split': [self.split],
                'note': ['No answer column found'],
                'num_predictions': [len(preds)]
            })
        
        # Parse predictions and compute accuracy
        results = []
        for idx, row in merged.iterrows():
            pred_text = row.get('prediction', '')
            gt_json = row[answer_col]
            
            # Parse ground truth
            try:
                gt_grid = json.loads(gt_json) if isinstance(gt_json, str) else gt_json
            except (json.JSONDecodeError, TypeError):
                gt_grid = None
            
            # Parse prediction
            pred_grid = parse_grid_from_response(str(pred_text) if pred_text else '')
            
            # Check exact match
            hit = False
            if gt_grid is not None and pred_grid is not None:
                hit = gt_grid == pred_grid
            
            results.append({
                'index': row['index'],
                'task_id': row.get('task_id_gt', row.get('task_id', '')),
                'hit': hit,
                'num_train_examples': row.get('num_train_examples_gt', row.get('num_train_examples', 0)),
                'pred_parsed': pred_grid is not None,
                'prediction': pred_text[:200] if pred_text else '',  # Truncate for logging
            })
        
        results_df = pd.DataFrame(results)
        
        # Overall accuracy
        overall_acc = results_df['hit'].mean() * 100
        
        # Accuracy by number of training examples
        acc_by_train = results_df.groupby('num_train_examples')['hit'].agg(['mean', 'count'])
        acc_by_train.columns = ['accuracy', 'count']
        acc_by_train['accuracy'] = acc_by_train['accuracy'] * 100
        
        # Parse success rate
        parse_rate = results_df['pred_parsed'].mean() * 100
        
        # Build summary results
        summary = pd.DataFrame({
            'split': [self.split],
            'Overall_Accuracy': [overall_acc],
            'Parse_Success_Rate': [parse_rate],
            'Total_Tasks': [len(results_df)],
            'Correct_Tasks': [results_df['hit'].sum()],
        })
        
        # Save detailed results
        eval_dir = Path(eval_file).parent
        
        # Save per-task results
        detailed_file = eval_dir / 'arc_agi_detailed_results.csv'
        results_df.to_csv(detailed_file, index=False)
        print(f"Saved detailed results to {detailed_file}")
        
        # Save accuracy by training examples
        train_results_file = eval_dir / 'accuracy_by_train_examples.csv'
        acc_by_train.to_csv(train_results_file)
        print(f"Saved accuracy by training examples to {train_results_file}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"ARC-AGI Evaluation Results ({self.split} split)")
        print(f"{'='*50}")
        print(f"Overall Accuracy: {overall_acc:.2f}%")
        print(f"Parse Success Rate: {parse_rate:.2f}%")
        print(f"Correct Tasks: {int(results_df['hit'].sum())}/{len(results_df)}")
        print(f"\nAccuracy by Number of Training Examples:")
        for n_train, row in acc_by_train.iterrows():
            print(f"  {n_train} examples: {row['accuracy']:.2f}% (n={int(row['count'])})")
        print(f"{'='*50}\n")
        
        return summary


class ARCAGIReasoningDataset(ARCAGIDataset):
    """Convenience class for ARC-AGI with reasoning mode enabled."""
    
    def __init__(self, dataset_path: Optional[str] = None, split: str = 'evaluation', **kwargs):
        super().__init__(
            dataset='ARC-AGI',
            dataset_path=dataset_path,
            split=split,
            reasoning_mode=True,
            **kwargs
        )


class ARCAGITrainingDataset(ARCAGIDataset):
    """Convenience class for ARC-AGI training split evaluation."""
    
    def __init__(self, dataset_path: Optional[str] = None, **kwargs):
        super().__init__(
            dataset='ARC-AGI',
            dataset_path=dataset_path,
            split='training',
            **kwargs
        )
