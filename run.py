#!/usr/bin/env python3
"""
Unified run script for VLM Knowledge Distillation training and evaluation.

This script provides a single entry point to:
1. Train a model on a dataset (graph or chartqa)
2. Evaluate a model on a dataset  
3. Run both training and evaluation sequentially

Usage:
    # Full pipeline (train + eval) on graph dataset with KD
    python run.py --dataset graph
    
    # Full pipeline on ChartQA dataset
    python run.py --dataset chartqa
    
    # Train CE baseline (no KD, beta=0)
    python run.py --dataset graph --baseline
    
    # Train only
    python run.py --dataset graph --train_only
    
    # Evaluate only (requires --model)
    python run.py --dataset graph --eval_only --model ./checkpoints/model
    
    # Custom KD beta
    python run.py --dataset graph --kd_beta 0.5
    
    # Evaluate specific checkpoints
    python run.py --dataset graph --eval_checkpoints 50,100
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SEED as DEFAULT_SEED,
    DEFAULT_CHECKPOINT_STEPS,
    DATASET_CONFIGS,
    get_kd_beta,
    get_output_dir,
)


def run_training(args) -> bool:
    """Run the training script."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "train.py"),
        "--dataset_type", args.dataset,
        "--mode", "kd",  # Always use KD mode
    ]
    
    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])
    
    if args.kd_beta is not None:
        cmd.extend(["--kd_beta", str(args.kd_beta)])
    
    if args.lr:
        cmd.extend(["--lr", str(args.lr)])
    
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    
    if args.checkpoint_steps:
        cmd.extend(["--checkpoint_steps", args.checkpoint_steps])
    
    if args.seed:
        cmd.extend(["--seed", str(args.seed)])
    
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_evaluation(args, model_path: str) -> bool:
    """Run the evaluation script on a single model/checkpoint."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "evaluate.py"),
        "--model", model_path,
        "--dataset_type", args.dataset,
    ]
    
    if args.eval_split:
        cmd.extend(["--split", args.eval_split])
    
    if args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    
    if args.skip_inference:
        cmd.append("--skip_inference")
    
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='VLM Knowledge Distillation - Training and Evaluation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==========================================================================
    # Dataset and mode
    # ==========================================================================
    parser.add_argument(
        '--dataset', type=str, default='graph',
        choices=['graph', 'chartqa'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Train CE baseline (no KD, beta=0.0)'
    )
    
    # ==========================================================================
    # Pipeline control
    # ==========================================================================
    parser.add_argument(
        '--train_only', action='store_true',
        help='Only run training, skip evaluation'
    )
    parser.add_argument(
        '--eval_only', action='store_true',
        help='Only run evaluation (requires --model)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model path for evaluation (required for --eval_only)'
    )
    
    # ==========================================================================
    # Training arguments
    # ==========================================================================
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for checkpoints (default: auto-generated)'
    )
    parser.add_argument(
        '--kd_beta', type=float, default=None,
        help='KD loss weight (default: 0.25 for graph, 0.075 for chartqa)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Learning rate (default: 5e-4)'
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=None,
        help='Batch size per device'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit training/eval samples (for debugging)'
    )
    parser.add_argument(
        '--checkpoint_steps', type=str, default=None,
        help='Comma-separated checkpoint steps (default: 10,20,...,100)'
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help='Random seed for training (data order, dropout). Student init seed is FIXED per dataset.'
    )
    
    # ==========================================================================
    # Evaluation arguments
    # ==========================================================================
    parser.add_argument(
        '--eval_split', type=str, default='test',
        help='Dataset split for evaluation'
    )
    parser.add_argument(
        '--eval_checkpoints', type=str, default=None,
        help='Comma-separated checkpoint steps to evaluate (default: all). Use "last" for final only.'
    )
    parser.add_argument(
        '--skip_inference', action='store_true',
        help='Skip inference, only compute metrics on existing predictions'
    )
    parser.add_argument(
        '--num_samples', type=int, default=None,
        help='Limit evaluation samples'
    )
    
    args = parser.parse_args()
    
    # ==========================================================================
    # Validate arguments
    # ==========================================================================
    if args.train_only and args.eval_only:
        print("Error: Cannot use both --train_only and --eval_only")
        sys.exit(1)
    
    if args.eval_only and not args.model:
        print("Error: --model is required when using --eval_only")
        sys.exit(1)
    
    # ==========================================================================
    # Handle baseline mode
    # ==========================================================================
    if args.baseline:
        args.kd_beta = 0.0  # CE baseline = no KD
    
    kd_beta = args.kd_beta if args.kd_beta is not None else get_kd_beta(args.dataset)
    
    # ==========================================================================
    # Print configuration
    # ==========================================================================
    print("=" * 60)
    print("VLM Knowledge Distillation Pipeline")
    print("=" * 60)
    print(f"Dataset:      {args.dataset} ({DATASET_CONFIGS[args.dataset]['name']})")
    print(f"Baseline:     {args.baseline} (CE only, no KD)" if args.baseline else f"KD Beta:      {kd_beta}")
    print(f"Train only:   {args.train_only}")
    print(f"Eval only:    {args.eval_only}")
    print("=" * 60)
    
    # ==========================================================================
    # Determine output directory
    # ==========================================================================
    if args.output_dir is None:
        if args.baseline:
            args.output_dir = str(get_output_dir(args.dataset, f"CE_baseline_seed{args.seed}"))
        else:
            args.output_dir = str(get_output_dir(args.dataset, f"kd_seed{args.seed}"))
    
    # ==========================================================================
    # Run training
    # ==========================================================================
    if not args.eval_only:
        success = run_training(args)
        if not success:
            print("\n" + "=" * 60)
            print("TRAINING FAILED")
            print("=" * 60)
            sys.exit(1)
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
    
    # ==========================================================================
    # Run evaluation
    # ==========================================================================
    if not args.train_only:
        # Determine checkpoint steps
        checkpoint_steps_str = args.checkpoint_steps or ",".join(map(str, DEFAULT_CHECKPOINT_STEPS))
        all_steps = sorted([int(s) for s in checkpoint_steps_str.split(",")])
        
        # Filter checkpoints if --eval_checkpoints specified
        if args.eval_checkpoints:
            if args.eval_checkpoints.lower() == 'last':
                eval_steps = [max(all_steps)]
            else:
                eval_steps = sorted([int(s) for s in args.eval_checkpoints.split(",")])
        else:
            eval_steps = all_steps
        
        # Determine model paths to evaluate
        if args.eval_only:
            model_paths = [args.model]
        else:
            model_paths = []
            for step in eval_steps:
                checkpoint_path = Path(args.output_dir) / f"checkpoint-{step}"
                if checkpoint_path.exists():
                    model_paths.append(str(checkpoint_path))
                else:
                    print(f"Warning: Checkpoint {checkpoint_path} not found, skipping")
            
            # Fallback to output directory if no checkpoints found
            if not model_paths and Path(args.output_dir).exists():
                model_paths = [args.output_dir]
            
            if not model_paths:
                print("Error: No checkpoints found to evaluate")
                sys.exit(1)
        
        print(f"\nEvaluating {len(model_paths)} checkpoint(s)...")
        
        all_success = True
        for i, model_path in enumerate(model_paths, 1):
            print(f"\n[{i}/{len(model_paths)}] Evaluating: {model_path}")
            success = run_evaluation(args, model_path)
            if not success:
                print(f"Evaluation FAILED for {model_path}")
                all_success = False
        
        if not all_success:
            print("\n" + "=" * 60)
            print("SOME EVALUATIONS FAILED")
            print("=" * 60)
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("ALL EVALUATIONS COMPLETED")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
