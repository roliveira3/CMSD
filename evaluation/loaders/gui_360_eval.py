"""
Evaluation functions for GUI-360 action prediction.
Exactly matches GUI-360 repository: https://github.com/2020-qqtcg/GUI-360/blob/main/eval_func/tool_func.py
"""

import re
import json
from typing import Optional, Dict, List, Union, Tuple
from .gui_360_tool_definitions import normalize_tool_args, normalize_tool_args_a11y


def extract_json_from_tool_call(text: str) -> Optional[Union[Dict, List]]:
    """
    Extract JSON from text, handling <tool_call> tags and raw JSON.
    """
    if not text:
        return None
    
    # Try to extract from <tool_call> tags first
    tool_call_pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(tool_call_pattern, text, re.DOTALL)
    
    if matches:
        json_str = matches[-1].strip()
    else:
        # Try to find JSON with "function" key
        # Look for the most complete JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, text, re.DOTALL)
        
        # Find the one containing "function"
        json_str = None
        for match in json_matches:
            if '"function"' in match or "'function'" in match:
                json_str = match
                break
        
        if json_str is None:
            return None
    
    # Try to parse the JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try with more lenient parsing
        try:
            # Replace single quotes with double quotes
            json_str_fixed = json_str.replace("'", '"')
            return json.loads(json_str_fixed)
        except json.JSONDecodeError:
            pass
        
        # Try with json5 if available
        try:
            import json5
            return json5.loads(json_str)
        except (ImportError, Exception):
            pass
    
    return None


def eval_tool(predict: Union[Dict, str], ground_truth: Union[Dict, str], 
              ground_bbox: Union[List, Dict, None] = None) -> Tuple[int, int, int]:
    """
    Evaluate the tool prediction against the ground truth.
    Exactly matches GUI-360 repository logic.

    Args:
        predict: The predicted tool call (dict or string to parse)
        ground_truth: The ground truth tool call (dict or string to parse)
        ground_bbox: The ground truth bounding box (rectangle or list of rectangles)

    Returns:
        A tuple of (function_match, args_match, status_match). For example: (1, 1, 1)
    """
    # Parse strings to dicts
    if isinstance(predict, str):
        predict = extract_json_from_tool_call(predict)
    if isinstance(ground_truth, str):
        ground_truth = extract_json_from_tool_call(ground_truth)
    
    def _to_rect(rect_like):
        """Convert various rectangle formats to standard dict format."""
        if rect_like is None:
            return None
        if isinstance(rect_like, dict):
            keys = {k.lower() for k in rect_like.keys()}
            if {"left", "top", "right", "bottom"}.issubset(keys):
                return {
                    "left": float(rect_like.get("left") or rect_like.get("Left")),
                    "top": float(rect_like.get("top") or rect_like.get("Top")),
                    "right": float(rect_like.get("right") or rect_like.get("Right")),
                    "bottom": float(rect_like.get("bottom") or rect_like.get("Bottom")),
                }
        if isinstance(rect_like, (list, tuple)) and len(rect_like) == 4:
            l, t, r, b = rect_like
            try:
                return {
                    "left": float(l),
                    "top": float(t),
                    "right": float(r),
                    "bottom": float(b),
                }
            except (TypeError, ValueError):
                return None
        return None

    def _split_rects(bbox: Union[List, Dict, None]) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Split bbox into start and end rectangles (for drag operations)."""
        if bbox is None:
            return None, None
        if isinstance(bbox, dict):
            return _to_rect(bbox), None
        if isinstance(bbox, list):
            # Support [rect] or [start_rect, end_rect] or [l,t,r,b]
            if len(bbox) == 1:
                return _to_rect(bbox[0]), None
            if len(bbox) >= 2:
                return _to_rect(bbox[0]), _to_rect(bbox[1])
            if len(bbox) == 4:
                return _to_rect(bbox), None
        return None, None

    def _extract_action(obj: Union[Dict, List, None]) -> Tuple[Optional[str], Dict, Optional[str]]:
        """Extract function, args, and status from action object."""
        if not obj:
            return None, {}, None
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if not isinstance(obj, dict):
            return None, {}, None

        # Get function name from various possible keys
        func = obj.get("function") or obj.get("name") or obj.get("tool")
        if not func and "action" in obj and isinstance(obj["action"], dict):
            func = obj["action"].get("function")
        if not func and "tool_call" in obj and isinstance(obj["tool_call"], dict):
            func = obj["tool_call"].get("function")

        # Get args from various possible keys
        args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
        if not args and "action" in obj and isinstance(obj["action"], dict):
            args = obj["action"].get("args", {})
        if not args and "tool_call" in obj and isinstance(obj["tool_call"], dict):
            args = obj["tool_call"].get("args", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        # Get status
        status = obj.get("status")
        if status is None and "action" in obj and isinstance(obj["action"], dict):
            status = obj["action"].get("status")
        if status is None and "tool_call" in obj and isinstance(obj["tool_call"], dict):
            status = obj["tool_call"].get("status")

        # Normalize coordinates in args
        if isinstance(args, dict):
            if func == "drag":
                # Handle start_x/start_y/end_x/end_y format
                if {"start_x", "start_y", "end_x", "end_y"}.issubset(args.keys()):
                    try:
                        args["start_coordinate"] = [float(args["start_x"]), float(args["start_y"])]
                        args["end_coordinate"] = [float(args["end_x"]), float(args["end_y"])]
                        for k in ["start_x", "start_y", "end_x", "end_y"]:
                            args.pop(k, None)
                    except (TypeError, ValueError):
                        pass
            else:
                # Handle x/y format
                if {"x", "y"}.issubset(args.keys()):
                    try:
                        args["coordinate"] = [float(args["x"]), float(args["y"])]
                        args.pop("x", None)
                        args.pop("y", None)
                    except (TypeError, ValueError):
                        pass
                # Handle coordinate_x/coordinate_y format
                if {"coordinate_x", "coordinate_y"}.issubset(obj.keys()):
                    try:
                        args["coordinate"] = [float(obj["coordinate_x"]), float(obj["coordinate_y"])]
                    except (TypeError, ValueError):
                        pass

        return (str(func).lower() if func else None), (args if isinstance(args, dict) else {}), (str(status) if status is not None else None)

    # Handle None predictions
    if predict is None or ground_truth is None:
        return 0, 0, 0

    # Extract actions
    pred_func, pred_args, pred_status = _extract_action(predict)
    gt_func, gt_args, gt_status = _extract_action(ground_truth)

    # Function match (case-insensitive)
    function_match = (pred_func == gt_func) if (pred_func and gt_func) else False

    # Status match (normalize common aliases)
    def _norm_status(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        s_norm = str(s).strip().upper()
        if s_norm == "OVERALL_FINISH":
            return "FINISH"
        return s_norm

    status_match = (_norm_status(pred_status) == _norm_status(gt_status)) if (pred_status is not None and gt_status is not None) else False

    # Args match with rectangle/tolerance logic
    def _compare_drag_args(p_args: Dict, g_args: Dict, rect_start: Optional[Dict], rect_end: Optional[Dict]) -> bool:
        """Compare drag operation arguments."""
        try:
            p_norm = normalize_tool_args("drag", p_args)
            g_norm = normalize_tool_args("drag", g_args)

            if "start_coordinate" not in p_norm or "end_coordinate" not in p_norm:
                return False
            if "start_coordinate" not in g_norm or "end_coordinate" not in g_norm:
                return False

            ps = p_norm["start_coordinate"]
            pe = p_norm["end_coordinate"]
            gs = g_norm["start_coordinate"]
            ge = g_norm["end_coordinate"]

            # Validate coordinate format
            if not (isinstance(ps, (list, tuple)) and len(ps) == 2):
                return False
            if not (isinstance(pe, (list, tuple)) and len(pe) == 2):
                return False
            if not (isinstance(gs, (list, tuple)) and len(gs) == 2):
                return False
            if not (isinstance(ge, (list, tuple)) and len(ge) == 2):
                return False

            tol = 25.0  # Default tolerance

            def _in_rect(coord, rect):
                if not rect:
                    return None
                x, y = float(coord[0]), float(coord[1])
                return rect["left"] <= x <= rect["right"] and rect["top"] <= y <= rect["bottom"]

            # Check start coordinate
            start_match = _in_rect(ps, rect_start)
            if start_match is None:
                start_match = abs(float(ps[0]) - float(gs[0])) <= tol and abs(float(ps[1]) - float(gs[1])) <= tol

            # Check end coordinate
            end_match = _in_rect(pe, rect_end)
            if end_match is None:
                end_match = abs(float(pe[0]) - float(ge[0])) <= tol and abs(float(pe[1]) - float(ge[1])) <= tol

            # Check other arguments
            other_ok = True
            for key in ["button", "duration", "key_hold"]:
                pv = p_norm.get(key)
                gv = g_norm.get(key)
                pv_str = str(pv).lower() if pv is not None else "none"
                gv_str = str(gv).lower() if gv is not None else "none"
                if pv_str != gv_str:
                    other_ok = False
                    break

            return bool(start_match and end_match and other_ok)
        except Exception:
            return False

    def _compare_regular_args(p_args: Dict, g_args: Dict, rect: Optional[Dict], p_func: str, g_func: str) -> bool:
        """Compare regular (non-drag) operation arguments."""
        try:
            p_norm = normalize_tool_args(p_func, p_args)
            g_norm = normalize_tool_args(g_func, g_args)

            # Handle coordinate-based operations
            if "coordinate" in p_norm and "coordinate" in g_norm:
                pc = p_norm["coordinate"]
                gc = g_norm["coordinate"]
                if isinstance(pc, (list, tuple)) and len(pc) == 2 and isinstance(gc, (list, tuple)) and len(gc) == 2:
                    tol = 25.0
                    # Priority 1: Rectangle-based matching
                    if rect:
                        x, y = float(pc[0]), float(pc[1])
                        coord_ok = rect["left"] <= x <= rect["right"] and rect["top"] <= y <= rect["bottom"]
                    else:
                        # Priority 2: Tolerance-based matching
                        coord_ok = abs(float(pc[0]) - float(gc[0])) <= tol and abs(float(pc[1]) - float(gc[1])) <= tol

                    # Check other arguments
                    other_ok = True
                    for key in p_norm:
                        if key == "coordinate":
                            continue
                        pv = p_norm.get(key)
                        gv = g_norm.get(key)
                        pv_str = str(pv).lower() if pv is not None else "none"
                        gv_str = str(gv).lower() if gv is not None else "none"
                        if pv_str != gv_str:
                            other_ok = False
                            break

                    return bool(coord_ok and other_ok)

            # Fallback: compare normalized dicts with string-normalization
            def _to_cmp(d: Dict) -> Dict:
                out = {}
                for k, v in d.items():
                    if isinstance(v, (str, bool)):
                        out[k] = str(v).lower()
                    elif v is None:
                        out[k] = "none"
                    else:
                        out[k] = v
                return out

            return _to_cmp(p_norm) == _to_cmp(g_norm)
        except Exception:
            return False

    # Get rectangles from ground_bbox
    rect_start, rect_end = _split_rects(ground_bbox)

    # Compute args_match
    if pred_func and gt_func and pred_args is not None and gt_args is not None:
        if pred_func == "drag" and gt_func == "drag":
            args_match = _compare_drag_args(pred_args, gt_args, rect_start, rect_end)
        else:
            args_match = _compare_regular_args(pred_args, gt_args, rect_start, pred_func or "unknown", gt_func or "unknown")
    else:
        args_match = False

    return int(function_match), int(args_match), int(status_match)


def eval_tool_a11y(predict: Union[Dict, str], ground_truth: Union[Dict, str], 
                   ground_bbox: Union[List, Dict, None] = None) -> Tuple[int, int, int]:
    """
    Evaluate the tool prediction against the ground truth with A11Y support.
    Prioritizes control_label matching over coordinate matching.

    Args:
        predict: The predicted tool call (dict or string to parse)
        ground_truth: The ground truth tool call (dict or string to parse)
        ground_bbox: The ground truth bounding box

    Returns:
        A tuple of (function_match, args_match, status_match). For example: (1, 1, 1)
    """
    # Parse strings to dicts
    if isinstance(predict, str):
        predict = extract_json_from_tool_call(predict)
    if isinstance(ground_truth, str):
        ground_truth = extract_json_from_tool_call(ground_truth)
    
    if predict is None or ground_truth is None:
        return 0, 0, 0
    
    # Extract function, args, status
    def _extract_action(obj):
        if not obj or not isinstance(obj, dict):
            return None, {}, None
        
        func = obj.get("function") or obj.get("name")
        if not func and "tool_call" in obj:
            func = obj["tool_call"].get("function")
        
        args = obj.get("args") or obj.get("arguments") or {}
        if not args and "tool_call" in obj:
            args = obj["tool_call"].get("args", {})
        
        status = obj.get("status")
        if not status and "tool_call" in obj:
            status = obj["tool_call"].get("status")
        
        return (str(func).lower() if func else None), args, status
    
    pred_func, pred_args, pred_status = _extract_action(predict)
    gt_func, gt_args, gt_status = _extract_action(ground_truth)
    
    # Function match
    function_match = (pred_func == gt_func) if (pred_func and gt_func) else False
    
    # Status match
    def _norm_status(s):
        if s is None:
            return None
        s_norm = str(s).strip().upper()
        return "FINISH" if s_norm == "OVERALL_FINISH" else s_norm
    
    status_match = (_norm_status(pred_status) == _norm_status(gt_status)) if (pred_status and gt_status) else False
    
    # Args match with A11Y support
    # In a11y mode, accept BOTH control_label and coordinate predictions
    args_match = False
    if function_match and pred_args and gt_args:
        gt_control_label = gt_args.get("control_label")
        pred_control_label = pred_args.get("control_label")
        
        # If ground truth has control_label, check if prediction matches either way
        if gt_control_label is not None:
            # Prediction can use control_label OR coordinates
            if pred_control_label is not None:
                # Control label comparison
                args_match = str(pred_control_label) == str(gt_control_label)
            else:
                # Prediction uses coordinates, fall back to coordinate comparison
                # Need to convert gt back to coordinate format for comparison
                return eval_tool(predict, ground_truth, ground_bbox)
        else:
            # Ground truth doesn't have control_label, use coordinate comparison
            return eval_tool(predict, ground_truth, ground_bbox)
    
    return int(function_match), int(args_match), int(status_match)
