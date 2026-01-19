"""
DiagramMCQ dataset loader for Multiple Choice questions.

Supports: CSV, TSV, JSON, JSONL, Parquet

Usage:
    cd evaluation
    python eval_with_probing.py \
        --model llava-v1.5-7b \
        --dataset_loader DiagramMCQ \
        --dataset_path datasets/VLQA_testmini_test/data.jsonl

Handles datasets with 'options' field and automatically converts answer text to letter keys.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union

from vlmeval.dataset.image_mcq import ImageMCQDataset
import pandas as pd


class DiagramMCQDataset(ImageMCQDataset):
    """
    DiagramMCQ dataset loader for Multiple Choice datasets.
    
    Supports: CSV, TSV, JSON, JSONL, Parquet
    
    Usage:
        --dataset DiagramMCQ --dataset_path /path/to/data.jsonl
        --dataset DiagramMCQ --dataset_path /path/to/data.json
        --dataset DiagramMCQ --dataset_path /path/to/directory  (looks for data.jsonl/json)
    
    Required columns: image, question, options (list or dict), answer
    Optional columns: index, category, etc.
    
    Special handling:
    - Converts 'options' list to A, B, C, D columns
    - Converts answer text to letter key (e.g., "blue" -> "B" if it's the 2nd option)
    """
    
    TYPE = 'MCQ'
    
    def __init__(
        self,
        dataset: str = 'DiagramMCQ',
        dataset_path: Optional[str] = None,
        question_col: str = 'question',
        answer_col: str = 'answer',
        image_col: str = 'image',
        option_cols: list = None,
        **kwargs
    ):
        """
        Load any local MCQ dataset file.
        
        Args:
            dataset: Dataset name
            dataset_path: Path to file or directory
            question_col: Column name for questions
            answer_col: Column name for answers
            image_col: Column name for images
            option_cols: Option column names (default: ['A', 'B', 'C', 'D'])
        """
        if dataset_path is None:
            raise ValueError("dataset_path is required")
        
        self.dataset_path = Path(dataset_path)
        self.dataset_name = dataset
        self.question_col = question_col
        self.answer_col = answer_col
        self.image_col = image_col
        self.option_cols = option_cols or ['A', 'B', 'C', 'D']
        
        # Initialize parent (will call load_data)
        super(ImageMCQDataset, self).__init__(dataset=dataset, **kwargs)
    
    def load_data(self, dataset):
        """Override parent's load_data to use local files instead of downloading."""
        data = self._load_local_file()
        print(f"Loaded {len(data)} samples from {self.dataset_path}")
        
        # Check if this is a text-only dataset (all images are empty)
        has_images = False
        if 'image' in data.columns:
            # Check if any image paths are non-empty
            has_images = data['image'].apply(lambda x: pd.notna(x) and str(x).strip() != '').any()
        
        if not has_images:
            print("Detected text-only dataset (no images)")
            # For text-only datasets, drop the image column to avoid parent class validation
            if 'image' in data.columns:
                data = data.drop(columns=['image'])
            if 'image_path' in data.columns:
                data = data.drop(columns=['image_path'])
            required_cols = ['index', 'question', 'answer'] + self.option_cols
        else:
            # Ensure all required columns exist and are properly formatted
            required_cols = ['index', 'image', 'question', 'answer'] + self.option_cols
            # Add image_path column (required by parent's build_prompt when meta_only=True)
            # This allows build_prompt to work without base64 encoding
            if 'image_path' not in data.columns:
                data['image_path'] = data['image']
        
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Don't set index as DataFrame index to avoid ambiguity
        # Keep it as a regular column
        # VLMEvalKit's inference and evaluation can handle this
        
        return data
    
    def _load_local_file(self) -> pd.DataFrame:
        """Load data from local file."""
        # If directory, look for common files
        if self.dataset_path.is_dir():
            for filename in ['data.csv', 'data.json', 'data.jsonl', 'data.parquet', 'data.tsv']:
                file_path = self.dataset_path / filename
                if file_path.exists():
                    print(f"Found {filename} in directory")
                    self.dataset_path = file_path
                    break
            else:
                raise ValueError(
                    f"No data file found in {self.dataset_path}. "
                    f"Expected: data.csv, data.json, data.jsonl, data.parquet, or data.tsv"
                )
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        # Load based on extension
        ext = self.dataset_path.suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.dataset_path)
        elif ext == '.tsv':
            df = pd.read_csv(self.dataset_path, sep='\t')
        elif ext == '.json':
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        elif ext == '.jsonl':
            df = pd.read_json(self.dataset_path, lines=True)
        elif ext == '.parquet':
            df = pd.read_parquet(self.dataset_path)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}\n"
                f"Supported: .csv, .tsv, .json, .jsonl, .parquet"
            )
        
        # Normalize column names if needed
        df = self._normalize_columns(df)
        
        # Add index if missing
        if 'index' not in df.columns:
            df['index'] = range(len(df))
        
        # Convert relative image paths to absolute
        # Skip empty strings (text-only datasets)
        if self.image_col in df.columns:
            df[self.image_col] = df[self.image_col].apply(
                lambda p: str(self._resolve_image_path(p)) if (pd.notna(p) and str(p).strip() != '') else ''
            )
        
        return df
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map custom column names to standard names and validate MCQ columns."""
        # Create column mapping
        col_mapping = {}
        
        # Map question column
        if self.question_col != 'question' and self.question_col in df.columns:
            col_mapping[self.question_col] = 'question'
        
        # Map answer column
        if self.answer_col != 'answer' and self.answer_col in df.columns:
            col_mapping[self.answer_col] = 'answer'
        
        # Map image column
        if self.image_col != 'image' and self.image_col in df.columns:
            col_mapping[self.image_col] = 'image'
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
            print(f"Mapped columns: {col_mapping}")
        
        # Validate required columns
        if 'question' not in df.columns:
            raise ValueError(
                f"Dataset missing 'question' column.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Use --question_col to specify the question column name"
            )
        
        if 'image' not in df.columns:
            raise ValueError(
                f"Dataset missing 'image' column.\n"
                f"Available columns: {list(df.columns)}\n"
                f"Use --image_col to specify the image column name"
            )
        
        # Handle datasets with 'options' field instead of A, B, C, D columns
        if 'options' in df.columns and any(col not in df.columns for col in self.option_cols):
            print("Found 'options' field - converting to A, B, C, D columns")
            df = self._expand_options(df)
        
        # Validate MCQ options exist
        missing_options = [col for col in self.option_cols if col not in df.columns]
        if missing_options:
            raise ValueError(
                f"MCQ dataset missing option columns: {missing_options}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Required columns: {self.option_cols}\n"
                f"If your dataset has an 'options' field (list/dict), make sure it's properly formatted."
            )
        
        return df
    
    def _expand_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert 'options' field (list or dict) to separate A, B, C, D columns."""
        import ast
        
        for idx, row in df.iterrows():
            options = row['options']
            
            # If it's a string representation, parse it
            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except:
                    raise ValueError(f"Row {idx}: Could not parse options field: {options}")
            
            # Handle list format: ["option1", "option2", "option3", "option4"]
            if isinstance(options, list):
                for i, opt in enumerate(options):
                    if i < len(self.option_cols):
                        df.at[idx, self.option_cols[i]] = opt
                
                # Convert answer from text to letter key (A, B, C, D)
                if 'answer' in df.columns and not pd.isna(row['answer']):
                    answer_text = str(row['answer']).strip()
                    # Find which option matches the answer
                    for i, opt in enumerate(options):
                        if str(opt).strip() == answer_text:
                            df.at[idx, 'answer'] = self.option_cols[i]
                            break
                    else:
                        # Answer not found in options - check if it's already a letter
                        if answer_text not in self.option_cols:
                            print(f"Warning: Row {idx} answer '{answer_text}' not found in options: {options}")
            
            # Handle dict format: {"A": "option1", "B": "option2", ...}
            elif isinstance(options, dict):
                for key, value in options.items():
                    if key.upper() in self.option_cols:
                        df.at[idx, key.upper()] = value
                
                # Convert answer from text to letter key if needed
                if 'answer' in df.columns and not pd.isna(row['answer']):
                    answer_text = str(row['answer']).strip()
                    # Check if answer is already a key (A, B, C, D)
                    if answer_text.upper() not in self.option_cols:
                        # Find which option matches the answer text
                        for key, value in options.items():
                            if str(value).strip() == answer_text:
                                df.at[idx, 'answer'] = key.upper()
                                break
                        else:
                            print(f"Warning: Row {idx} answer '{answer_text}' not found in options: {options}")
            
            else:
                raise ValueError(
                    f"Row {idx}: 'options' must be a list or dict, got {type(options)}"
                )
        
        return df
    
    def _resolve_image_path(self, image_path: Union[str, Path]) -> Path:
        """Convert relative paths to absolute based on dataset location."""
        # Normalize Windows-style separators to avoid nonexistent paths on Linux
        image_path = Path(str(image_path).replace('\\', '/'))
        
        if image_path.is_absolute():
            return image_path
        
        # Resolve relative to dataset file location
        if self.dataset_path.is_file():
            base_dir = self.dataset_path.parent
        else:
            base_dir = self.dataset_path
        
        # Return absolute path
        resolved_path = (base_dir / image_path).resolve()
        return resolved_path
    
    def build_prompt(self, line):
        """
        Override parent's build_prompt to handle empty image paths.
        If image path is empty/None, only use text (no image).
        """
        import string
        
        if isinstance(line, int):
            line = self.data.iloc[line]

        # Check if image path is empty or None
        has_image = True
        if 'image' not in line:
            has_image = False
        elif pd.isna(line['image']) or str(line['image']).strip() == '':
            has_image = False
        
        # Only process image if it exists
        if has_image:
            # Prefer explicit image_path if present
            image_path_val = None
            if 'image_path' in line and not pd.isna(line['image_path']) and str(line['image_path']).strip() != '':
                image_path_val = line['image_path']

            if self.meta_only:
                from vlmeval.smp import toliststr
                if image_path_val is None:
                    has_image = False
                else:
                    tgt_path = toliststr(image_path_val)
            else:
                # If we already have a path, resolve it; otherwise fall back to dump_image (base64)
                if image_path_val is not None:
                    resolved = self._resolve_image_path(image_path_val)
                    tgt_path = str(resolved)
                else:
                    tgt_path = self.dump_image(line)
        
        # Build the text prompt
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            # prompt += 'Please select the correct answer from the options above. \n'
            prompt += (
               "Do not explain. Answer with a single uppercase letter (A, B, C, or D) and nothing else.\nAnswer: "
            )

        # Build messages - only add image if it exists
        msgs = []
        if has_image:
            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

