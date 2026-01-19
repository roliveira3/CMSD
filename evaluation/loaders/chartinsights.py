"""
ChartInsights dataset loader for the ChartInsights benchmark.

ChartInsights is a benchmark for evaluating MLLMs on low-level chart question answering,
covering 10 task types across 4 question formats:
- fill_the_blank
- Multiple_choice (MCQ)
- Judgement_question (Yes/No)
- Corrective_question

Paper: https://arxiv.org/abs/2405.07001
GitHub: https://github.com/HKUSTDial/ChartInsights

This loader follows the EXACT format from the official ChartInsights evaluation code.
"""

import pandas as pd
import json
import os
import re
import string
from pathlib import Path
from typing import Optional, List, Dict, Any
from vlmeval.dataset.image_mcq import ImageMCQDataset


class ChartInsightsDataset(ImageMCQDataset):
    """
    ChartInsights dataset loader supporting all 5 sub-datasets.
    
    Sub-datasets:
    1. Overall Evaluation (4,388 questions) - Main benchmark
    2. Textual Prompt (255 questions) - CSV assistance test
    3. Vary Chart Element (4,912 questions) - Visual variation robustness
    4. Vary Chart Quality (2,415 questions) - Quality degradation robustness
    5. Visual Prompt (255 questions) - Visual QA test
    
    Key differences from generic loaders:
    1. The answer is a VALUE (e.g., 'Thur', 7), NOT a letter (A, B, C)
    2. Options are embedded in the question text
    3. CSV assistance follows specific Q&A format (only for Textual Prompt)
    4. Different subsets use different image path structures
    """
    
    TYPE = 'MCQ'
    
    # Task types in ChartInsights (10 types)
    TASK_TYPES = [
        'data retrieval', 'extreme', 'determine range', 'order', 'filter',
        'cluster', 'correlation', 'anomaly', 'distribution', 'reasoning'
    ]
    
    # Images in Overall Evaluation that have corresponding CSV files (380 out of 400)
    # These are the images from Textual Prompt that are also in Overall Evaluation
    IMAGES_WITH_CSV = None  # Will be populated on first use
    
    # Subset configurations
    SUBSET_CONFIGS = {
        'Overall Evaluation': {
            'qa_file': 'overall_test_qa_pairs.json',
            'charts_dir': 'Charts',
            'image_field': 'image_index',
            'has_csv': True,  # 95% of images have CSV files (380/400)
            'image_pattern': '{image_index}.png',
            'csv_pattern': '../Textual Prompt/Tables/tables/{image_index}.csv'
        },
        'Textual Prompt': {
            'qa_file': 'textual_prompt_test_qa_pairs.json',
            'charts_dir': 'Charts',
            'image_field': 'image_index',
            'has_csv': True,
            'image_pattern': '{image_index}.png',
            'csv_pattern': 'Tables/tables/{image_index}.csv'
        },
        'Vary Chart Element': {
            'qa_file': 'vary_element_qa_pairs.json',
            'charts_dir': 'Charts',
            'image_field': 'changed_image',
            'has_csv': False,
            'image_pattern': None  # Uses changed_image directly with subdirs
        },
        'Vary Chart Quality': {
            'qa_file': 'attack_qa_pairs.json',
            'charts_dir': 'Charts/new_image_attack',
            'image_field': 'changed_image',
            'has_csv': False,
            'image_pattern': '{changed_image}'
        },
        'Visual Prompt': {
            'qa_file': 'visual_qa_pairs_test_qa_pairs.json',
            'charts_dir': 'Charts',
            'image_field': 'image_index',
            'has_csv': False,
            'image_pattern': '{image_index}.png'
        }
    }
    
    def __init__(
        self,
        dataset: str = 'ChartInsights',
        dataset_path: Optional[str] = None,
        subset: str = 'Overall Evaluation',
        question_type: str = 'Multiple_choice',
        include_csv: bool = False,
        filter_by_csv: bool = False,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize ChartInsights loader.
        
        Args:
            dataset: Dataset name
            dataset_path: Path to ChartInsights root directory (not subset!)
            subset: Which sub-dataset to use ('Overall Evaluation', 'Textual Prompt', etc.)
            question_type: Type of questions (Multiple_choice, Judgement_question, etc.)
            include_csv: Whether to include CSV table data (only for Textual Prompt)
            num_samples: Limit number of samples (for debugging)
        """
        self.question_type = question_type
        self.include_csv = include_csv
        self.filter_by_csv = filter_by_csv
        self.num_samples = num_samples
        self.subset = subset
        
        # Validate subset
        if subset not in self.SUBSET_CONFIGS:
            raise ValueError(f"Invalid subset '{subset}'. Must be one of: {list(self.SUBSET_CONFIGS.keys())}")
        
        # Validate filter_by_csv
        if filter_by_csv and subset != 'Overall Evaluation':
            raise ValueError(f"filter_by_csv only valid for 'Overall Evaluation' subset")
        
        # Get subset configuration
        self.config = self.SUBSET_CONFIGS[subset]
        
        # Validate CSV support
        if include_csv and not self.config['has_csv']:
            raise ValueError(f"CSV support not available for subset '{subset}'. Only 'Textual Prompt' has CSV data.")
        
        if dataset_path is None:
            raise ValueError("dataset_path is required for ChartInsights")
        
        # dataset_path should be the ROOT directory containing all subsets
        self.dataset_root = Path(dataset_path)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root path not found: {dataset_path}")
        
        # Set up paths for this specific subset
        self.dataset_path = self.dataset_root / subset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Subset directory not found: {self.dataset_path}")
        
        self.charts_dir = self.dataset_path / self.config['charts_dir']
        self.qa_file = self.dataset_path / self.config['qa_file']
        
        # Set up CSV directory if applicable
        if self.config['has_csv']:
            if 'csv_pattern' in self.config:
                # Extract base directory from csv_pattern
                # Example: '../Textual Prompt/Tables/tables/{image_index}.csv' -> '../Textual Prompt/Tables/tables/'
                csv_pattern = self.config['csv_pattern']
                csv_dir_part = csv_pattern.split('{')[0]  # Get everything before the placeholder
                self.tables_dir = (self.dataset_path / csv_dir_part).resolve()
            else:
                self.tables_dir = self.dataset_path / 'Tables' / 'tables'
            
            # Only validate if include_csv is requested
            if include_csv and not self.tables_dir.exists():
                raise FileNotFoundError(f"Tables directory not found: {self.tables_dir}")
        else:
            self.tables_dir = None
        
        # Validate directories/files exist
        if not self.charts_dir.exists():
            raise FileNotFoundError(f"Charts directory not found: {self.charts_dir}")
        if not self.qa_file.exists():
            raise FileNotFoundError(f"QA pairs file not found: {self.qa_file}")
        
        # Call parent init
        super(ImageMCQDataset, self).__init__(dataset=dataset, **kwargs)
    
    def _get_images_with_csv(self) -> set:
        """Get set of image indices that have corresponding CSV files."""
        if ChartInsightsDataset.IMAGES_WITH_CSV is not None:
            return ChartInsightsDataset.IMAGES_WITH_CSV
        
        # Populate the set by checking which CSV files exist
        images_with_csv = set()
        if self.tables_dir and self.tables_dir.exists():
            for csv_file in self.tables_dir.glob('*.csv'):
                try:
                    image_id = int(csv_file.stem)
                    images_with_csv.add(image_id)
                except ValueError:
                    continue
        
        # Cache it
        ChartInsightsDataset.IMAGES_WITH_CSV = images_with_csv
        return images_with_csv
    
    # ==================== CSV Processing Methods ====================
    # These follow the EXACT implementation from ChartInsights reference code
    
    def _extract_csv_data(self, csv_path: str) -> str:
        """Extract x-axis labels from CSV (for non-scatter charts)."""
        df = pd.read_csv(csv_path)
        x_axis = df.iloc[:, 0].values.flatten()
        x_axis_str = ', '.join(str(x) for x in x_axis)
        return x_axis_str
    
    def _scatter_extract_csv_data(self, csv_path: str):
        """Extract categories and min/max values for scatter charts."""
        df = pd.read_csv(csv_path)
        categories = set(df.iloc[:, -1].values.flatten())
        max_y_value = df['y_data'].max()
        min_y_value = df['y_data'].min()
        return categories, min_y_value, max_y_value
    
    def _process_dataframe_generic(self, csv_path: str) -> str:
        """Process dataframe into structured string format (EXACT ChartInsights format)."""
        df = pd.read_csv(csv_path, index_col=0)
        rows = df.index.tolist()
        columns = df.columns.tolist()
        
        result_string = ""
        if len(columns) >= 2:
            for row in rows:
                result_string += f"- {row}: "
                for col in columns:
                    value = df.loc[row, col]
                    result_string += f"{col}: {value}, "
                result_string = result_string[:-2]  # Remove trailing comma
                result_string += "\t"
        else:
            for row in rows:
                result_string += f"- {row}: "
                for col in columns:
                    value = df.loc[row, col]
                    result_string += f"{value}, "
                result_string = result_string[:-2]
                result_string += "\t"
        return result_string.strip()
    
    def _build_csv_context(self, image_index: int, image_type: str) -> str:
        """
        Build CSV context prompt following EXACT ChartInsights format.
        
        This matches ChartQ_and_A_prompts from the reference code:
        - Non-scatter: "1.Q:What type is this chart?A:{type}. 2.Q:What are the labels of x-axis?A:{labels}. 
                        3.Q:What are the data labels of each element? A:{data}. 4.Q:"
        - Scatter: "1.Q:What type is this chart?A:scatter chart. 2.Q:What are the names of different categories?A:{cats}. 
                   3.Q:What is the maximum and minimum value of this scatter plot? A:{max}, {min}. 4.Q:"
        """
        csv_path = self.tables_dir / f"{image_index}.csv"
        
        if not csv_path.exists():
            return ""
        
        try:
            if image_type == 'scatter':
                categories, min_value, max_value = self._scatter_extract_csv_data(str(csv_path))
                # EXACT format from reference: ChartQ_and_A_prompts for scatter
                context = (f"1.Q:What type is this chart?A:scatter chart. "
                          f"2.Q:What are the names of different categories?A:{categories}. "
                          f"3.Q:What is the maximum and minimum value of this scatter plot? A:{max_value, min_value}. "
                          f"4.Q:")
            else:
                # Extract chart type (last word of image_type)
                chart_type = image_type.split(' ')[-1] if ' ' in image_type else image_type
                x_axis_names = self._extract_csv_data(str(csv_path))
                chart_data = self._process_dataframe_generic(str(csv_path))
                # EXACT format from reference: ChartQ_and_A_prompts for non-scatter
                context = (f"1.Q:What type is this chart?A:{chart_type}. "
                          f"2.Q:What are the labels of x-axis?A:{x_axis_names}. "
                          f"3.Q:What are the data labels of each element? A:{chart_data}. "
                          f"4.Q:")
            return context
        except Exception as e:
            print(f"Warning: Could not process CSV {csv_path}: {e}")
            return ""
    
    # ==================== Data Loading ====================
    
    def load_data(self, dataset):
        """Load ChartInsights data from JSON file."""
        # Load QA pairs with proper encoding
        with open(self.qa_file, 'r', encoding='latin-1') as f:
            qa_data = json.load(f)
        
        # Get images with CSV if filtering is requested
        images_with_csv = None
        if self.filter_by_csv and self.subset == 'Overall Evaluation':
            images_with_csv = self._get_images_with_csv()
            print(f"Filtering to {len(images_with_csv)} images with CSV files")
        
        data_list = []
        skipped_no_csv = 0
        
        for item in qa_data:
            image_index = item['image_index']
            
            # Skip if filtering by CSV and this image doesn't have one
            if images_with_csv is not None and image_index not in images_with_csv:
                skipped_no_csv += 1
                continue
            
            task_type = item['type']
            question_level = item['question_level']
            image_type = item.get('image_type', '')  # May not exist in all subsets
            pair_index = item['pair_index']
            
            # Get the image reference based on subset config
            image_ref = item.get(self.config['image_field'])
            if image_ref is None:
                print(f"Warning: Missing '{self.config['image_field']}' in item {pair_index}")
                continue
            
            # Extract additional subset-specific fields
            extra_fields = {}
            if self.subset == 'Vary Chart Element':
                extra_fields['vary_element'] = item.get('vary_element', '')
                extra_fields['vary_type'] = item.get('vary_type', '')
            elif self.subset == 'Vary Chart Quality':
                extra_fields['attack_type'] = item.get('attack_type', '')
            
            # Find the specific question type in QA_pairs
            qa_pairs = item['QA_pairs']
            for qa in qa_pairs:
                if self.question_type in qa:
                    question_data = qa[self.question_type]
                    question = question_data[0]  # Full question text with options
                    answer = question_data[1]    # Answer VALUE (not letter!)
                    
                    record = self._create_record(
                        question=question,
                        answer=answer,
                        image_index=image_index,
                        image_ref=image_ref,
                        task_type=task_type,
                        question_level=question_level,
                        image_type=image_type,
                        pair_index=pair_index,
                        **extra_fields
                    )
                    
                    if record is not None:
                        data_list.append(record)
                    break
        
        if not data_list:
            raise ValueError(f"No data found for question_type={self.question_type}")
        
        if skipped_no_csv > 0:
            print(f"Skipped {skipped_no_csv} items without CSV files")
        
        # Limit samples if requested
        if self.num_samples is not None and self.num_samples > 0:
            data_list = data_list[:self.num_samples]
        
        # Convert to DataFrame
        df = pd.DataFrame(data_list)
        df['index'] = range(len(df))
        
        # Print dataset statistics
        print(f"\n{'='*70}")
        print(f"ChartInsights Dataset Loaded")
        print(f"{'='*70}")
        print(f"Subset: {self.subset}")
        print(f"Question Type: {self.question_type}")
        print(f"CSV Assistance: {self.include_csv}")
        print(f"CSV Filtered: {self.filter_by_csv}")
        print(f"Total Samples: {len(df)}")
        print(f"\nTask Type Distribution:")
        for task, count in df['task_type'].value_counts().items():
            print(f"  {task}: {count}")
        print(f"\nQuestion Level Distribution:")
        for level, count in df['question_level'].value_counts().items():
            print(f"  {level}: {count}")
        
        # Print subset-specific statistics
        if 'vary_element' in df.columns:
            print(f"\nVary Element Distribution:")
            for elem, count in df['vary_element'].value_counts().items():
                print(f"  {elem}: {count}")
        if 'attack_type' in df.columns:
            print(f"\nAttack Type Distribution:")
            for attack, count in df['attack_type'].value_counts().items():
                print(f"  {attack}: {count}")
        
        # Print sample records with EXACT prompts
        self._print_sample_records(df)
        
        return df
    
    def _resolve_image_path(self, image_index: int, image_ref: Any) -> Optional[str]:
        """Resolve the actual image file path based on subset configuration."""
        
        if self.subset == 'Vary Chart Element':
            # For Vary Chart Element, the changed_image contains the full path
            # Example: "1098_bar_original.png" or "1098_bar_color_different.png"
            # These are in subdirectories like images/original/, images/color/different/, etc.
            image_name = str(image_ref)
            
            # Try to find the image in various subdirectories
            search_dirs = [
                self.charts_dir / 'images' / 'original',
                self.charts_dir / 'images' / 'color' / 'different',
                self.charts_dir / 'images' / 'color' / 'similar',
                self.charts_dir / 'images' / 'legend' / 'with_legend',
                self.charts_dir / 'images' / 'legend' / 'without_legend',
                self.charts_dir / 'images' / 'orientation' / 'horizontal',
                self.charts_dir / 'images' / 'orientation' / 'vertical',
            ]
            
            for search_dir in search_dirs:
                candidate = search_dir / image_name
                if candidate.exists():
                    return str(candidate)
            
            # If not found in subdirs, try direct path
            candidate = self.charts_dir / image_name
            if candidate.exists():
                return str(candidate)
            
            return None
        
        elif self.subset == 'Vary Chart Quality':
            # For Vary Chart Quality, changed_image is the filename in new_image_attack/
            image_name = str(image_ref)
            image_path = self.charts_dir / image_name
            return str(image_path)
        
        else:
            # For Overall Evaluation, Textual Prompt, Visual Prompt
            # Use image_index to build path
            if self.config['image_pattern']:
                image_name = self.config['image_pattern'].format(image_index=image_index)
                image_path = self.charts_dir / image_name
                return str(image_path)
            else:
                return str(self.charts_dir / f"{image_index}.png")
    
    def _create_record(
        self,
        question: str,
        answer: Any,
        image_index: int,
        image_ref: Any,
        task_type: str,
        question_level: str,
        image_type: str,
        pair_index: int,
        **extra_fields
    ) -> Optional[Dict[str, Any]]:
        """Create a single data record."""
        
        # Build image path based on subset configuration
        image_path = self._resolve_image_path(image_index, image_ref)
        
        if image_path is None or not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path} (index={image_index}, ref={image_ref})")
            return None
        
        # Extract the actual question and options from the question string
        # Format for MCQ: "Question text?NOTE:Choose your answer from the following options [opt1, opt2, opt3].Begin..."
        # Format for Y/N: "Question text?NOTE:You only need to answer 'Yes' or 'No'."
        # Format for fill_the_blank: "Question text?NOTE:Begin your answer with 'My answer is [your answer]'."
        parts = question.split('NOTE:')
        actual_question = parts[0].strip()
        
        # Check question type
        is_yn_question = self.question_type == 'Judgement_question'
        is_fill_blank = self.question_type == 'fill_the_blank'
        
        # Extract options list from the question
        options_match = re.search(r'\[([^\]]+)\]', question)
        options_list = []
        if options_match:
            options_str = options_match.group(1)
            try:
                import ast
                options_list = ast.literal_eval(f"[{options_str}]")
            except:
                # Fallback: split by comma
                options_list = [o.strip().strip("'\"") for o in options_str.split(',')]
        elif is_yn_question:
            # Y/N questions don't have options in brackets, use Yes/No
            options_list = ['Yes', 'No']
        elif is_fill_blank:
            # fill_the_blank has no options - answer is free-form
            options_list = []
        
        # Build options dictionary with letters A, B, C, ...
        options_dict = {}
        for i, opt in enumerate(options_list):
            letter = string.ascii_uppercase[i]
            options_dict[letter] = str(opt)
        
        # Find which letter corresponds to the answer VALUE
        # Need to handle list/tuple format differences: answer might be [1, 2] but option is (1, 2)
        def normalize_for_comparison(val):
            """Normalize value for comparison - handles list/tuple differences."""
            if isinstance(val, (list, tuple)):
                # Convert to tuple for comparison
                return tuple(val)
            if isinstance(val, str):
                # Try to parse string representation
                try:
                    import ast
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, (list, tuple)):
                        return tuple(parsed)
                    return parsed
                except:
                    # Also try replacing brackets
                    normalized = val.replace('[', '(').replace(']', ')')
                    try:
                        parsed = ast.literal_eval(normalized)
                        if isinstance(parsed, (list, tuple)):
                            return tuple(parsed)
                        return parsed
                    except:
                        return val
            return val
        
        answer_normalized = normalize_for_comparison(answer)
        answer_str = str(answer)
        answer_letter = None
        
        # For fill_the_blank, there are no options - answer is the value directly
        if is_fill_blank:
            answer_letter = 'A'  # Placeholder for compatibility
        # For Y/N questions, answer is already 'Yes' or 'No', map directly
        elif is_yn_question:
            answer_upper = answer_str.upper()
            if 'YES' in answer_upper:
                answer_letter = 'A'  # Yes is option A
            elif 'NO' in answer_upper:
                answer_letter = 'B'  # No is option B
            else:
                # Try exact match
                for letter, opt_value in options_dict.items():
                    if str(opt_value).upper() == answer_upper:
                        answer_letter = letter
                        break
        else:
            # For MCQ questions, match answer value to options
            for letter, opt_value in options_dict.items():
                opt_normalized = normalize_for_comparison(opt_value)
                
                # Direct comparison after normalization
                if opt_normalized == answer_normalized:
                    answer_letter = letter
                    break
                
                # String comparison
                if str(opt_value) == str(answer):
                    answer_letter = letter
                    break
                
                # Also try numeric comparison for simple values
                try:
                    if float(opt_value) == float(answer):
                        answer_letter = letter
                        break
                except:
                    pass
            
            if answer_letter is None:
                # Fallback: fuzzy string match
                for letter, opt_value in options_dict.items():
                    if answer_str.lower() in opt_value.lower() or opt_value.lower() in answer_str.lower():
                        answer_letter = letter
                        break
        
        if answer_letter is None and options_list:
            print(f"Warning: Could not match answer '{answer}' to options {options_dict}")
            answer_letter = 'A'  # Default fallback
        
        # Build CSV context if needed
        csv_context = ""
        if self.include_csv:
            csv_context = self._build_csv_context(image_index, image_type)
        
        record = {
            'image': image_path,
            'image_path': image_path,
            'image_index': image_index,
            'image_ref': str(image_ref),  # Store the original reference
            'task_type': task_type,
            'question_level': question_level,
            'image_type': image_type,
            'pair_index': pair_index,
            'question': actual_question,
            'full_question': question,  # Original question with NOTE
            'answer': answer_letter,     # Letter (A, B, C, ...)
            'answer_value': answer_str,  # Actual value
            'csv_context': csv_context,
            'num_options': len(options_list),
        }
        
        # Add subset-specific fields
        record.update(extra_fields)
        
        # Add option columns dynamically
        for letter in string.ascii_uppercase[:len(options_list)]:
            record[letter] = options_dict.get(letter, '')
        # Fill remaining standard columns with empty for compatibility
        for letter in string.ascii_uppercase[len(options_list):8]:  # Up to H
            record[letter] = ''
        
        return record
    
    def _print_sample_records(self, df: pd.DataFrame, num_samples: int = 3):
        """Print sample records showing exact prompts."""
        print(f"\n{'='*70}")
        print("SAMPLE RECORDS (showing exact prompts)")
        print(f"{'='*70}")
        
        for idx in range(min(num_samples, len(df))):
            row = df.iloc[idx]
            print(f"\n{'='*70}")
            print(f"SAMPLE {idx + 1}")
            print(f"{'='*70}")
            print(f"Image: {row['image_index']}.png")
            print(f"Task Type: {row['task_type']}")
            print(f"Question Level: {row['question_level']}")
            print(f"Image Type: {row['image_type']}")
            
            # Get available options
            options = {}
            for letter in string.ascii_uppercase:
                if letter in row and row[letter] and str(row[letter]).strip():
                    options[letter] = row[letter]
            
            print(f"\n--- EXACT PROMPT SENT TO MODEL ---")
            prompt = self._build_full_prompt(row)
            print(prompt)
            
            print(f"\n--- EXPECTED OUTPUT ---")
            print(f"Expected Answer Letter: {row['answer']}")
            print(f"Expected Answer Value: {row['answer_value']}")
            if row['answer'] in options:
                print(f"Option {row['answer']}: {options[row['answer']]}")
            
            print(f"{'='*70}")
        
        print(f"\n{'='*70}\n")
    
    def _build_full_prompt(self, row) -> str:
        """Build the exact prompt that will be sent to the model."""
        parts = []
        
        # Add CSV context if present
        if self.include_csv and row.get('csv_context'):
            parts.append(row['csv_context'])
        
        # Add the question
        parts.append(row['question'])
        
        # Check question type
        is_yn_question = self.question_type == 'Judgement_question'
        is_fill_blank = self.question_type == 'fill_the_blank'
        
        if is_fill_blank:
            # For fill_the_blank, ask for answer in specific format
            parts.append("\nBegin your answer with 'My answer is ' followed by your answer.")
        elif is_yn_question:
            # For Y/N questions, just ask for Yes/No answer
            parts.append("\nAnswer with 'Yes' or 'No' only.")
        else:
            # Add options for MCQ
            options = []
            for letter in string.ascii_uppercase:
                if letter in row and row[letter] and str(row[letter]).strip():
                    options.append(f"{letter}. {row[letter]}")
            
            if options:
                parts.append("\nOptions:")
                parts.extend(options)
            
            # Add answer instruction
            if options:
                option_letters = ', '.join([opt.split('.')[0] for opt in options])
                parts.append(f"\nAnswer with a single letter ({option_letters}) only.")
        
        # Add A: suffix (following ChartInsights format)
        parts.append("Answer:")
        
        return '\n'.join(parts)
    
    # ==================== Prompt Building ====================
    
    def build_prompt(self, line):
        """Build prompt for model evaluation."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        
        # Get image path
        if self.meta_only:
            from vlmeval.smp import toliststr
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        
        # Build the text prompt
        prompt = self._build_full_prompt(line)
        
        # Build messages
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        
        return msgs
    
    # ==================== Evaluation ====================
    
    def evaluate(self, eval_file, **judge_kwargs):
        """
        Evaluate ChartInsights predictions.
        
        Reports accuracy by:
        - Task type (10 types)
        - Question level (common, complex)
        - Overall
        """
        from vlmeval.smp import load, dump
        
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        
        # Clean predictions and answers
        data['prediction'] = [str(x).strip().upper() if pd.notna(x) else '' for x in data['prediction']]
        data['answer'] = [str(x).strip().upper() for x in data['answer']]
        
        # Determine question type from the data
        # Y/N answers are 'A' or 'B' where A='Yes', B='No'
        # fill_the_blank has free-form answers (no letter options)
        is_yn_eval = False
        is_fill_blank_eval = False
        if 'answer_value' in data.columns:
            # Sample a few answer_value entries to check if they're Yes/No
            sample_answers = data['answer_value'].head(10).str.upper()
            is_yn_eval = sample_answers.isin(['YES', 'NO']).any()
        
        # Check if num_options is 0 (fill_the_blank)
        if 'num_options' in data.columns:
            is_fill_blank_eval = (data['num_options'] == 0).any()
        
        # Calculate match for each sample
        def check_match(pred: str, ans: str, ans_value: str = None, is_fill_blank: bool = False) -> int:
            """Check if prediction matches answer.
            
            For fill_the_blank: Extract answer from 'My answer is X' format and compare value
            For Y/N questions: Match against 'Yes'/'No' or letter 'A'/'B'
            For MCQ questions: Match against letter (A, B, C, ...)
            """
            if not pred or not ans:
                return 0
            
            pred_upper = pred.upper().strip()
            
            # Handle fill_the_blank questions
            if is_fill_blank and ans_value:
                # Extract answer from 'My answer is X' format
                match = re.search(r'MY ANSWER IS\s+(.+)', pred_upper)
                if match:
                    extracted_answer = match.group(1).strip()
                else:
                    # Fallback: use entire prediction
                    extracted_answer = pred_upper
                
                # Compare with expected answer value
                ans_value_upper = str(ans_value).upper().strip()
                
                # Exact match
                if extracted_answer == ans_value_upper:
                    return 1
                
                # Partial match (answer contains expected value or vice versa)
                if ans_value_upper in extracted_answer or extracted_answer in ans_value_upper:
                    return 1
                
                # Try numeric comparison
                try:
                    if float(extracted_answer) == float(ans_value_upper):
                        return 1
                except:
                    pass
                
                return 0
            
            # For Y/N questions, check both direct text match and letter match
            if is_yn_eval and ans_value:
                ans_value_upper = ans_value.upper()
                
                # Direct yes/no match
                if ans_value_upper in pred_upper:
                    return 1
                
                # Letter match (A=Yes, B=No)
                if ans == 'A' and 'YES' in pred_upper:
                    return 1
                if ans == 'B' and 'NO' in pred_upper:
                    return 1
            
            # Standard letter matching (for both MCQ and Y/N with letters)
            # Direct match
            if pred_upper == ans:
                return 1
            
            # Extract first letter from prediction
            pred_clean = pred.replace('.', '').replace(':', '').replace(',', '').strip()
            
            # Check first character
            if pred_clean and pred_clean[0].upper() == ans:
                return 1
            
            # Check for standalone letter
            if re.search(rf'\b{ans}\b', pred_upper):
                return 1
            
            # Check for patterns like "A)", "(A)", "Answer: A"
            if re.search(rf'(?:^|[^A-Z]){ans}(?:[^A-Z]|$)', pred_upper):
                return 1
            
            return 0
        
        # Calculate hits with answer_value if available
        if 'answer_value' in data.columns:
            data['hit'] = [check_match(p, a, av, is_fill_blank_eval) for p, a, av in zip(data['prediction'], data['answer'], data['answer_value'])]
        else:
            data['hit'] = [check_match(p, a, None, is_fill_blank_eval) for p, a in zip(data['prediction'], data['answer'])]
        
        # Calculate metrics
        results = {}
        
        # Overall accuracy
        overall_acc = data['hit'].mean() * 100
        results['overall_accuracy'] = round(overall_acc, 2)
        results['total_samples'] = len(data)
        results['correct_samples'] = int(data['hit'].sum())
        
        # Accuracy by task type
        if 'task_type' in data.columns:
            task_results = {}
            for task_type in data['task_type'].unique():
                mask = data['task_type'] == task_type
                task_acc = data.loc[mask, 'hit'].mean() * 100
                task_results[task_type] = {
                    'accuracy': round(task_acc, 2),
                    'count': int(mask.sum()),
                    'correct': int(data.loc[mask, 'hit'].sum())
                }
            results['by_task_type'] = task_results
        
        # Accuracy by question level
        if 'question_level' in data.columns:
            level_results = {}
            for level in data['question_level'].unique():
                mask = data['question_level'] == level
                level_acc = data.loc[mask, 'hit'].mean() * 100
                level_results[level] = {
                    'accuracy': round(level_acc, 2),
                    'count': int(mask.sum()),
                    'correct': int(data.loc[mask, 'hit'].sum())
                }
            results['by_question_level'] = level_results
        
        # Print results
        print(f"\n{'='*70}")
        print("ChartInsights Evaluation Results")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {overall_acc:.2f}% ({results['correct_samples']}/{results['total_samples']})")
        
        if 'by_task_type' in results:
            print(f"\nAccuracy by Task Type:")
            for task, info in sorted(results['by_task_type'].items()):
                print(f"  {task}: {info['accuracy']:.2f}% ({info['correct']}/{info['count']})")
        
        if 'by_question_level' in results:
            print(f"\nAccuracy by Question Level:")
            for level, info in sorted(results['by_question_level'].items()):
                print(f"  {level}: {info['accuracy']:.2f}% ({info['correct']}/{info['count']})")
        
        print(f"{'='*70}\n")
        
        # Save detailed results
        result_file = eval_file.replace('.xlsx', '_results.json').replace('.csv', '_results.json')
        dump(results, result_file)
        print(f"Detailed results saved to: {result_file}")
        
        # Return in format expected by the evaluation framework
        return {'Overall': overall_acc}


# ==================== Convenience Classes ====================
# These classes provide easy access to different subsets and question types

# --- Overall Evaluation (Main Benchmark) ---
class ChartInsightsOverallMCQ(ChartInsightsDataset):
    """Overall Evaluation - Multiple Choice without CSV (4,388 questions)"""
    def __init__(self, dataset='ChartInsightsOverallMCQ', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Multiple_choice',
                        include_csv=False, **kwargs)

class ChartInsightsOverallMCQWithCSV(ChartInsightsDataset):
    """Overall Evaluation - Multiple Choice with CSV (4,388 questions, 95% have CSV)"""
    def __init__(self, dataset='ChartInsightsOverallMCQWithCSV', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Multiple_choice',
                        include_csv=True, **kwargs)

class ChartInsightsOverallJudgement(ChartInsightsDataset):
    """Overall Evaluation - Yes/No without CSV (4,388 questions)"""
    def __init__(self, dataset='ChartInsightsOverallJudgement', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Judgement_question',
                        include_csv=False, **kwargs)

class ChartInsightsOverallJudgementWithCSV(ChartInsightsDataset):
    """Overall Evaluation - Yes/No with CSV (4,388 questions, 95% have CSV)"""
    def __init__(self, dataset='ChartInsightsOverallJudgementWithCSV', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Judgement_question',
                        include_csv=True, **kwargs)

class ChartInsightsOverallFillBlank(ChartInsightsDataset):
    """Overall Evaluation - Fill the Blank without CSV (4,388 questions)"""
    def __init__(self, dataset='ChartInsightsOverallFillBlank', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='fill_the_blank',
                        include_csv=False, **kwargs)

class ChartInsightsOverallFillBlankWithCSV(ChartInsightsDataset):
    """Overall Evaluation - Fill the Blank with CSV (4,388 questions, 95% have CSV)"""
    def __init__(self, dataset='ChartInsightsOverallFillBlankWithCSV', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='fill_the_blank',
                        include_csv=True, **kwargs)

# Classes that filter to only the 380 images with CSV files
class ChartInsightsOverallMCQWithCSVFiltered(ChartInsightsDataset):
    """Overall Evaluation - MCQ with CSV, filtered to 380 images with tables"""
    def __init__(self, dataset='ChartInsightsOverallMCQWithCSVFiltered', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Multiple_choice',
                        include_csv=True, filter_by_csv=True, **kwargs)

class ChartInsightsOverallJudgementWithCSVFiltered(ChartInsightsDataset):
    """Overall Evaluation - Yes/No with CSV, filtered to 380 images with tables"""
    def __init__(self, dataset='ChartInsightsOverallJudgementWithCSVFiltered', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='Judgement_question',
                        include_csv=True, filter_by_csv=True, **kwargs)

class ChartInsightsOverallFillBlankWithCSVFiltered(ChartInsightsDataset):
    """Overall Evaluation - Fill the Blank with CSV, filtered to 380 images with tables"""
    def __init__(self, dataset='ChartInsightsOverallFillBlankWithCSVFiltered', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Overall Evaluation', question_type='fill_the_blank',
                        include_csv=True, filter_by_csv=True, **kwargs)

# --- Textual Prompt (CSV Test) ---
class ChartInsightsTextualMCQ(ChartInsightsDataset):
    """Textual Prompt - Multiple Choice without CSV (255 questions)"""
    def __init__(self, dataset='ChartInsightsTextualMCQ', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Textual Prompt', question_type='Multiple_choice',
                        include_csv=False, **kwargs)

class ChartInsightsTextualMCQWithCSV(ChartInsightsDataset):
    """Textual Prompt - Multiple Choice with CSV (255 questions)"""
    def __init__(self, dataset='ChartInsightsTextualMCQWithCSV', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Textual Prompt', question_type='Multiple_choice',
                        include_csv=True, **kwargs)

class ChartInsightsTextualJudgement(ChartInsightsDataset):
    """Textual Prompt - Yes/No without CSV (255 questions)"""
    def __init__(self, dataset='ChartInsightsTextualJudgement', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Textual Prompt', question_type='Judgement_question',
                        include_csv=False, **kwargs)

class ChartInsightsTextualJudgementWithCSV(ChartInsightsDataset):
    """Textual Prompt - Yes/No with CSV (255 questions)"""
    def __init__(self, dataset='ChartInsightsTextualJudgementWithCSV', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Textual Prompt', question_type='Judgement_question',
                        include_csv=True, **kwargs)

# --- Vary Chart Element (Robustness) ---
class ChartInsightsVaryElementMCQ(ChartInsightsDataset):
    """Vary Chart Element - Multiple Choice (4,912 questions)"""
    def __init__(self, dataset='ChartInsightsVaryElementMCQ', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Vary Chart Element', question_type='Multiple_choice',
                        include_csv=False, **kwargs)

class ChartInsightsVaryElementJudgement(ChartInsightsDataset):
    """Vary Chart Element - Yes/No (4,912 questions)"""
    def __init__(self, dataset='ChartInsightsVaryElementJudgement', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Vary Chart Element', question_type='Judgement_question',
                        include_csv=False, **kwargs)

# --- Vary Chart Quality (Adversarial) ---
class ChartInsightsVaryQualityMCQ(ChartInsightsDataset):
    """Vary Chart Quality - Multiple Choice (2,415 questions)"""
    def __init__(self, dataset='ChartInsightsVaryQualityMCQ', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Vary Chart Quality', question_type='Multiple_choice',
                        include_csv=False, **kwargs)

class ChartInsightsVaryQualityJudgement(ChartInsightsDataset):
    """Vary Chart Quality - Yes/No (2,415 questions)"""
    def __init__(self, dataset='ChartInsightsVaryQualityJudgement', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Vary Chart Quality', question_type='Judgement_question',
                        include_csv=False, **kwargs)

# --- Visual Prompt ---
class ChartInsightsVisualMCQ(ChartInsightsDataset):
    """Visual Prompt - Multiple Choice (255 questions)"""
    def __init__(self, dataset='ChartInsightsVisualMCQ', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Visual Prompt', question_type='Multiple_choice',
                        include_csv=False, **kwargs)

class ChartInsightsVisualJudgement(ChartInsightsDataset):
    """Visual Prompt - Yes/No (255 questions)"""
    def __init__(self, dataset='ChartInsightsVisualJudgement', dataset_path=None, **kwargs):
        super().__init__(dataset=dataset, dataset_path=dataset_path,
                        subset='Visual Prompt', question_type='Judgement_question',
                        include_csv=False, **kwargs)

# --- Legacy aliases for backward compatibility ---
ChartInsightsMCQ = ChartInsightsTextualMCQ
ChartInsightsMCQWithCSV = ChartInsightsTextualMCQWithCSV
ChartInsightsJudgement = ChartInsightsTextualJudgement
ChartInsightsJudgementWithCSV = ChartInsightsTextualJudgementWithCSV
