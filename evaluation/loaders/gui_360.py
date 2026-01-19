"""
GUI-360 dataset loader for GUI action prediction evaluation.

This implementation exactly matches the evaluation from the GUI-360 GitHub repository:
https://github.com/2020-qqtcg/GUI-360

It uses:
- Exact prompts from gui_360_prompts.py (copied from the repo)
- Exact evaluation logic from gui_360_eval.py (copied from eval_func/tool_func.py)

Supports two evaluation modes:
1. Visual-only: Only the clean screenshot is provided to the model
2. A11y (with accessibility): Both annotated screenshot and accessibility information are provided

The dataset contains trajectories for:
- Excel (in_app, search, online)
- Word (in_app, search, online)
- PowerPoint (in_app, search, online)

Key features:
- Uses exact prompts from the GUI-360 paper/repo
- Uses exact evaluation logic from the repo (rectangle-based coordinate matching)
- Stores function, args, and status predictions for evaluation
- Supports filtering by application domain (excel, word, ppt)
- Supports filtering by category (in_app, search, online)

Usage:
    # Visual-only evaluation on all test data
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader GUI360 \
        --dataset_path /cluster/scratch/rbertolissi/datasets/GUI-360/test \
        --gui360_mode visual

    # A11y evaluation on all test data
    python eval_with_probing.py \
        --model OpenGVLab/InternVL3_5-4B \
        --dataset_loader GUI360 \
        --dataset_path /cluster/scratch/rbertolissi/datasets/GUI-360/test \
        --gui360_mode a11y
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from vlmeval.dataset.image_base import ImageBaseDataset
import warnings

# Import prompts from gui_360_prompts.py (exact copy from repo)
from .gui_360_prompts import (
    ACTION_PREDICTION_USER_PROMPT_QWEN,
    ACTION_PREDICTION_A11Y_USER_PROMPT_QWEN,
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
    SUPPORTED_ACTIONS_WORD_A11Y,
    SUPPORTED_ACTIONS_EXCEL_A11Y,
    SUPPORTED_ACTIONS_PPT_A11Y,
)

# Import evaluation function from gui_360_eval.py (exact copy from repo)
from .gui_360_eval import eval_tool, eval_tool_a11y


def get_supported_actions(domain: str, a11y_mode: bool = False) -> str:
    """Get supported actions based on domain and mode."""
    domain_lower = domain.lower()
    if a11y_mode:
        actions_map = {
            "excel": SUPPORTED_ACTIONS_EXCEL_A11Y,
            "word": SUPPORTED_ACTIONS_WORD_A11Y,
            "ppt": SUPPORTED_ACTIONS_PPT_A11Y,
        }
    else:
        actions_map = {
            "excel": SUPPORTED_ACTIONS_EXCEL,
            "word": SUPPORTED_ACTIONS_WORD,
            "ppt": SUPPORTED_ACTIONS_PPT,
        }
    return actions_map.get(domain_lower, SUPPORTED_ACTIONS_EXCEL)


def format_a11y_info(control_infos: Dict) -> str:
    """
    Format accessibility information exactly as in the GUI-360 repo.
    
    This provides the a11y info in a structured text format that includes:
    - Application window info
    - Merged controls with labels, types, text, and rectangles
    """
    if not control_infos:
        return "No accessibility information available."
    
    result_lines = []
    
    # Application window info
    app_windows = control_infos.get("application_windows_info", {})
    if app_windows:
        app_name = app_windows.get("active_application_name", "Unknown")
        result_lines.append(f"Active Application: {app_name}")
        
        # Window rectangle if available
        if "active_window_rect" in app_windows:
            rect = app_windows["active_window_rect"]
            result_lines.append(f"Window Rectangle: {rect}")
    
    # Merged controls info
    merged_controls = control_infos.get("merged_controls_info", [])
    if merged_controls:
        result_lines.append("\nControl Elements:")
        for control in merged_controls:
            label = control.get("label", "?")
            control_type = control.get("control_type", "Unknown")
            control_text = control.get("control_text", "")
            rect = control.get("control_rect", [])
            
            # Format exactly as repo does
            if control_text:
                line = f"[{label}] {control_type}: \"{control_text}\""
            else:
                line = f"[{label}] {control_type}"
            
            if rect:
                line += f" @ {rect}"
            
            result_lines.append(f"  {line}")
    
    return "\n".join(result_lines) if result_lines else "No accessibility information available."


def format_history(previous_actions: List[str]) -> str:
    """Format history of actions exactly as in the repo."""
    if not previous_actions:
        return "None"
    return "\n".join(previous_actions)


class GUI360Dataset(ImageBaseDataset):
    """
    GUI-360 dataset loader for action prediction evaluation.
    
    This implementation exactly matches the GUI-360 GitHub repository evaluation.
    
    Args:
        dataset: Dataset name
        dataset_path: Path to GUI-360 test/train directory
        mode: Evaluation mode - 'visual' or 'a11y'
        domain: Filter by domain - 'excel', 'word', 'ppt', or 'all'
        category: Filter by category - 'in_app', 'search', 'online', or 'all'
        num_samples: Limit number of samples (for debugging)
    """
    
    TYPE = 'ActionPrediction'
    
    def __init__(
        self,
        dataset: str = 'GUI360',
        dataset_path: Optional[str] = None,
        mode: str = 'visual',  # 'visual' or 'a11y'
        domain: str = 'all',  # 'excel', 'word', 'ppt', or 'all'
        category: str = 'all',  # 'in_app', 'search', 'online', or 'all'
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """Initialize GUI-360 dataset loader."""
        if dataset_path is None:
            raise ValueError("dataset_path is required for GUI-360 dataset")
        
        self.dataset_path = Path(dataset_path)
        self.mode = mode
        self.domain_filter = domain.lower()
        self.category_filter = category.lower()
        self.num_samples = num_samples
        
        # Validate paths
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"GUI-360 dataset not found: {dataset_path}")
        
        # Set up paths
        self.data_dir = self.dataset_path / "data"
        self.image_dir = self.dataset_path / "image"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Build dataset name
        name_parts = ['GUI360', mode]
        if domain != 'all':
            name_parts.append(domain)
        if category != 'all':
            name_parts.append(category)
        self.dataset_name = '_'.join(name_parts)
        
        # Load data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} samples for {self.dataset_name}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from GUI-360 dataset directory."""
        samples = []
        
        # Determine domains to process
        if self.domain_filter == 'all':
            domains = ['excel', 'word', 'ppt']
        else:
            domains = [self.domain_filter]
        
        for domain in domains:
            domain_data_path = self.data_dir / domain
            if not domain_data_path.exists():
                continue
            
            # Determine categories to process
            if self.category_filter == 'all':
                categories = [c.name for c in domain_data_path.iterdir() if c.is_dir()]
            else:
                categories = [self.category_filter]
            
            for category in categories:
                category_path = domain_data_path / category / "success"
                if not category_path.exists():
                    continue
                
                # Process each jsonl file (each is a trajectory)
                jsonl_files = list(category_path.glob("*.jsonl"))
                
                for jsonl_file in jsonl_files:
                    execution_id = jsonl_file.stem
                    
                    # Load all steps from the trajectory
                    try:
                        with open(jsonl_file, 'r', encoding='utf-8') as f:
                            all_steps = []
                            for line_num, line in enumerate(f, 1):
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line.strip())
                                    all_steps.append({"line_num": line_num, "data": data})
                                except json.JSONDecodeError:
                                    continue
                        
                        # Process each step
                        for i, step_info in enumerate(all_steps):
                            data = step_info["data"]
                            line_num = step_info["line_num"]
                            step = data.get("step", {})
                            
                            # Skip if no action
                            if "action" not in step:
                                continue
                            
                            action = step["action"]
                            
                            # Build image paths
                            if self.mode == 'a11y':
                                screenshot_key = "screenshot_annotated"
                            else:
                                screenshot_key = "screenshot_clean"
                            
                            screenshot_path = step.get(screenshot_key, "")
                            if not screenshot_path:
                                continue
                            
                            # Full image path
                            image_path = self.image_dir / domain / category / screenshot_path
                            
                            # Check image exists
                            if not image_path.exists():
                                continue
                            
                            # Construct previous actions history (matching repo format)
                            previous_actions = []
                            for j in range(i):
                                prev_step_data = all_steps[j]["data"]
                                prev_thought = prev_step_data.get("step", {}).get("thought", "")
                                if prev_thought:
                                    previous_actions.append(f"Step {j+1}: {prev_thought}")
                            
                            # Build sample ID
                            sample_id = f"{domain}_{category}_{execution_id}_{line_num}"
                            
                            # Get status - keep original format for ground truth
                            # (eval_tool handles normalization of OVERALL_FINISH)
                            status = step.get("status", "CONTINUE")
                            
                            # Get a11y info for a11y mode
                            control_infos = {}
                            if self.mode == 'a11y':
                                control_infos = step.get("control_infos", {})
                            
                            # Store ground truth in the format expected by eval_tool
                            # For a11y mode: convert coordinate-based args to control_label when available
                            function_name = action.get("function", "")
                            args = action.get("args", {})
                            
                            if self.mode == 'a11y' and "control_label" in action:
                                # Convert ground truth to use control_label instead of coordinates
                                args = dict(args)  # Make a copy
                                control_label = action.get("control_label", "")
                                
                                if function_name == "click":
                                    # Replace coordinate with control_label
                                    args = {
                                        "control_label": control_label,
                                        "button": args.get("button", "left"),
                                        "double": args.get("double", False),
                                    }
                                    if args.get("pressed"):
                                        args["pressed"] = args["pressed"]
                                        
                                elif function_name == "drag":
                                    # For drag, if it has control_label, use it
                                    args = {
                                        "control_label": control_label,
                                        "button": args.get("button", "left"),
                                    }
                                    if args.get("duration"):
                                        args["duration"] = args["duration"]
                                    if args.get("key_hold"):
                                        args["key_hold"] = args["key_hold"]
                                        
                                elif function_name == "type":
                                    # Type can have control_label for where to type
                                    if "x" in args or "y" in args:
                                        args = {
                                            "control_label": control_label,
                                            "text": args.get("text", ""),
                                            "clear": args.get("clear", False),
                                        }
                            
                            ground_truth = {
                                "function": function_name,
                                "args": args,
                                "status": status
                            }
                            
                            # Get rectangle for coordinate matching (if available)
                            ground_bbox = action.get("rectangle", {})
                            
                            sample = {
                                "index": len(samples),
                                "sample_id": sample_id,
                                "domain": domain,
                                "category": category,
                                "request": data.get("request", ""),
                                "image": str(image_path),
                                "previous_actions": json.dumps(previous_actions),
                                # Store ground truth both as JSON (for evaluation) and separate columns (for detection)
                                "ground_truth": json.dumps(ground_truth),
                                "ground_bbox": json.dumps(ground_bbox),
                                # Separate columns for ground truth detection in eval_with_probing.py
                                "gt_function": ground_truth["function"],
                                "gt_args": json.dumps(ground_truth["args"]),
                                "gt_status": ground_truth["status"],
                                # Store control_infos for a11y prompt building
                                "control_infos": json.dumps(control_infos) if control_infos else "",
                                "step_index": i + 1,
                                "execution_id": execution_id,
                            }
                            
                            samples.append(sample)
                            
                            # Check sample limit
                            if self.num_samples and len(samples) >= self.num_samples:
                                return pd.DataFrame(samples)
                    
                    except Exception as e:
                        warnings.warn(f"Error processing {jsonl_file}: {e}")
                        continue
        
        return pd.DataFrame(samples)
    
    def build_prompt(self, line: Dict) -> str:
        """
        Build the prompt for a single sample.
        
        Uses exact prompts from the GUI-360 repository (imported from gui_360_prompts.py).
        No truncation is applied - the full prompt is passed to the model.
        """
        domain = line['domain']
        request = line['request']
        
        # Parse previous actions
        previous_actions_str = line.get('previous_actions', '[]')
        if isinstance(previous_actions_str, str):
            previous_actions = json.loads(previous_actions_str) if previous_actions_str else []
        else:
            previous_actions = previous_actions_str if previous_actions_str else []
        
        # Format history
        history = format_history(previous_actions)
        
        # Get supported actions based on domain and mode
        a11y_mode = self.mode == 'a11y'
        actions = get_supported_actions(domain, a11y_mode)
        
        if a11y_mode:
            # Get a11y info
            control_infos_str = line.get('control_infos', '')
            if control_infos_str:
                control_infos = json.loads(control_infos_str)
                a11y_text = format_a11y_info(control_infos)
            else:
                a11y_text = "No accessibility information available."
            
            # Use exact prompt from repo
            prompt = ACTION_PREDICTION_A11Y_USER_PROMPT_QWEN.format(
                instruction=request,
                a11y=a11y_text,
                history=history,
                actions=actions
            )
        else:
            # Use exact prompt from repo
            prompt = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
                instruction=request,
                history=history,
                actions=actions
            )
        
        return prompt
    
    def evaluate(self, result_file: str, **kwargs) -> Dict[str, Any]:
        """
        Evaluate predictions using GUI-360 metrics.
        
        Uses the exact evaluation logic from the GUI-360 repository
        (imported from gui_360_eval.py).
        
        Metrics:
        - Function Accuracy: % of correct function predictions
        - Args Accuracy: % of correct argument predictions
        - Status Accuracy: % of correct status predictions
        - Overall Success Rate: % where all three match
        
        Returns nested dict with overall and per-domain/category breakdowns.
        """
        # Load predictions based on file format
        if result_file.endswith('.jsonl'):
            pred_data = pd.read_json(result_file, lines=True)
        elif result_file.endswith('.xlsx'):
            pred_data = pd.read_excel(result_file)
        elif result_file.endswith('.csv'):
            pred_data = pd.read_csv(result_file)
        elif result_file.endswith('.tsv'):
            pred_data = pd.read_csv(result_file, sep='\t')
        else:
            # Try to auto-detect
            try:
                pred_data = pd.read_excel(result_file)
            except:
                pred_data = pd.read_csv(result_file)
        
        # Merge with ground truth
        merged = self.data.merge(pred_data[['index', 'prediction']], on='index', how='left')
        
        # Collect results per sample
        results = []
        
        for _, row in merged.iterrows():
            # Get prediction string
            pred_response = str(row.get('prediction', ''))
            
            # Get ground truth in the format expected by eval_tool
            ground_truth_str = row.get('ground_truth', '{}')
            ground_truth = json.loads(ground_truth_str) if isinstance(ground_truth_str, str) else ground_truth_str
            
            # Get ground bbox for coordinate matching
            ground_bbox_str = row.get('ground_bbox', '{}')
            ground_bbox = json.loads(ground_bbox_str) if isinstance(ground_bbox_str, str) else ground_bbox_str
            
            # Use the appropriate eval function based on mode
            # For a11y mode, use eval_tool_a11y which handles both control_label and coordinates
            # For visual mode, use eval_tool which handles only coordinates
            if self.mode == 'a11y':
                function_match, args_match, status_match = eval_tool_a11y(
                    predict=pred_response,
                    ground_truth=ground_truth,
                    ground_bbox=ground_bbox
                )
            else:
                function_match, args_match, status_match = eval_tool(
                    predict=pred_response,
                    ground_truth=ground_truth,
                    ground_bbox=ground_bbox
                )
            
            results.append({
                'domain': row['domain'],
                'category': row['category'],
                'function_match': function_match,
                'args_match': args_match,
                'status_match': status_match,
                'success': function_match and args_match and status_match,
            })
        
        # Compute overall metrics
        n_samples = len(results)
        if n_samples == 0:
            return {'error': 'No samples to evaluate'}
        
        metrics = {
            'overall': {
                'total_samples': n_samples,
                'function_accuracy': 100 * sum(r['function_match'] for r in results) / n_samples,
                'args_accuracy': 100 * sum(r['args_match'] for r in results) / n_samples,
                'status_accuracy': 100 * sum(r['status_match'] for r in results) / n_samples,
                'success_rate': 100 * sum(r['success'] for r in results) / n_samples,
            }
        }
        
        # Compute per-domain metrics
        metrics['by_domain'] = {}
        for domain in ['excel', 'word', 'ppt']:
            domain_results = [r for r in results if r['domain'] == domain]
            if domain_results:
                n = len(domain_results)
                metrics['by_domain'][domain] = {
                    'total_samples': n,
                    'function_accuracy': 100 * sum(r['function_match'] for r in domain_results) / n,
                    'args_accuracy': 100 * sum(r['args_match'] for r in domain_results) / n,
                    'status_accuracy': 100 * sum(r['status_match'] for r in domain_results) / n,
                    'success_rate': 100 * sum(r['success'] for r in domain_results) / n,
                }
        
        # Compute per-category metrics
        metrics['by_category'] = {}
        for category in ['in_app', 'search', 'online']:
            cat_results = [r for r in results if r['category'] == category]
            if cat_results:
                n = len(cat_results)
                metrics['by_category'][category] = {
                    'total_samples': n,
                    'function_accuracy': 100 * sum(r['function_match'] for r in cat_results) / n,
                    'args_accuracy': 100 * sum(r['args_match'] for r in cat_results) / n,
                    'status_accuracy': 100 * sum(r['status_match'] for r in cat_results) / n,
                    'success_rate': 100 * sum(r['success'] for r in cat_results) / n,
                }
        
        # Compute per domain+category metrics
        metrics['by_domain_category'] = {}
        for domain in ['excel', 'word', 'ppt']:
            metrics['by_domain_category'][domain] = {}
            for category in ['in_app', 'search', 'online']:
                dc_results = [r for r in results if r['domain'] == domain and r['category'] == category]
                if dc_results:
                    n = len(dc_results)
                    metrics['by_domain_category'][domain][category] = {
                        'total_samples': n,
                        'function_accuracy': 100 * sum(r['function_match'] for r in dc_results) / n,
                        'args_accuracy': 100 * sum(r['args_match'] for r in dc_results) / n,
                        'status_accuracy': 100 * sum(r['status_match'] for r in dc_results) / n,
                        'success_rate': 100 * sum(r['success'] for r in dc_results) / n,
                    }
        
        return metrics


# =============================================================================
# Convenience classes for specific configurations
# =============================================================================

class GUI360VisualDataset(GUI360Dataset):
    """GUI-360 Visual-only evaluation dataset."""
    def __init__(self, dataset_path: str, domain: str = 'all', category: str = 'all', 
                 num_samples: Optional[int] = None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            mode='visual',
            domain=domain,
            category=category,
            num_samples=num_samples,
            **kwargs
        )


class GUI360A11yDataset(GUI360Dataset):
    """GUI-360 A11y (with accessibility info) evaluation dataset."""
    def __init__(self, dataset_path: str, domain: str = 'all', category: str = 'all',
                 num_samples: Optional[int] = None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            mode='a11y',
            domain=domain,
            category=category,
            num_samples=num_samples,
            **kwargs
        )


class GUI360ExcelDataset(GUI360Dataset):
    """GUI-360 Excel-only evaluation dataset."""
    def __init__(self, dataset_path: str, mode: str = 'visual', category: str = 'all',
                 num_samples: Optional[int] = None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            mode=mode,
            domain='excel',
            category=category,
            num_samples=num_samples,
            **kwargs
        )


class GUI360WordDataset(GUI360Dataset):
    """GUI-360 Word-only evaluation dataset."""
    def __init__(self, dataset_path: str, mode: str = 'visual', category: str = 'all',
                 num_samples: Optional[int] = None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            mode=mode,
            domain='word',
            category=category,
            num_samples=num_samples,
            **kwargs
        )


class GUI360PPTDataset(GUI360Dataset):
    """GUI-360 PowerPoint-only evaluation dataset."""
    def __init__(self, dataset_path: str, mode: str = 'visual', category: str = 'all',
                 num_samples: Optional[int] = None, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            mode=mode,
            domain='ppt',
            category=category,
            num_samples=num_samples,
            **kwargs
        )
