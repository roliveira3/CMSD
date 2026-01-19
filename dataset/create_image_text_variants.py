#!/usr/bin/env python3
"""
Create image-only, image+text, and text-only variants from a dataset that has both
'question' and 'text_prompt' fields.

Usage:
    python create_image_text_variants.py --input dataset/collections/graph_dataset_4x4/test
"""

import argparse
import json
import shutil
from pathlib import Path


def create_variants(input_dir: Path, output_base: Path = None):
    """
    Create image-only, image+text, and text-only variants from a source dataset.
    
    Args:
        input_dir: Path to source dataset (should contain data.jsonl and images/)
        output_base: Base directory for output (defaults to input_dir.parent)
    """
    input_dir = Path(input_dir)
    
    if output_base is None:
        output_base = input_dir.parent
    else:
        output_base = Path(output_base)
    
    # Read source data
    source_file = input_dir / "data.jsonl"
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Create output directories
    image_only_dir = output_base / f"{input_dir.name}_image_only"
    image_text_dir = output_base / f"{input_dir.name}_image_text"
    text_only_dir = output_base / f"{input_dir.name}_text_only"
    
    image_only_dir.mkdir(parents=True, exist_ok=True)
    image_text_dir.mkdir(parents=True, exist_ok=True)
    text_only_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating image-only variant: {image_only_dir}")
    print(f"Creating image+text variant: {image_text_dir}")
    print(f"Creating text-only variant: {text_only_dir}")
    
    # Copy images directory if it exists
    if (input_dir / "images").exists():
        print("Copying images...")
        if (image_only_dir / "images").exists():
            shutil.rmtree(image_only_dir / "images")
        if (image_text_dir / "images").exists():
            shutil.rmtree(image_text_dir / "images")
        
        shutil.copytree(input_dir / "images", image_only_dir / "images")
        shutil.copytree(input_dir / "images", image_text_dir / "images")
    
    # Process data
    with open(source_file, 'r') as f_in:
        with open(image_only_dir / "data.jsonl", 'w') as f_image_only:
            with open(image_text_dir / "data.jsonl", 'w') as f_image_text:
                with open(text_only_dir / "data.jsonl", 'w') as f_text_only:
                    for line in f_in:
                        data = json.loads(line)
                        
                        # Image-only variant: keep question as-is
                        image_only_data = data.copy()
                        f_image_only.write(json.dumps(image_only_data) + '\n')
                        
                        # Image+text variant: replace question with text_prompt if available
                        image_text_data = data.copy()
                        if 'text_prompt' in data and data['text_prompt']:
                            image_text_data['question'] = data['text_prompt']
                        f_image_text.write(json.dumps(image_text_data) + '\n')
                        
                        # Text-only variant: use text_prompt and remove image
                        text_only_data = data.copy()
                        if 'text_prompt' in data and data['text_prompt']:
                            text_only_data['question'] = data['text_prompt']
                        text_only_data['image'] = ""  # Empty string to indicate no image
                        f_text_only.write(json.dumps(text_only_data) + '\n')
    
    print(f"✓ Created {image_only_dir / 'data.jsonl'}")
    print(f"✓ Created {image_text_dir / 'data.jsonl'}")
    print(f"✓ Created {text_only_dir / 'data.jsonl'}")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create image-only, image+text, and text-only variants from a dataset"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to source dataset directory (should contain data.jsonl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Base directory for output variants (defaults to parent of input)'
    )
    
    args = parser.parse_args()
    
    create_variants(Path(args.input), Path(args.output) if args.output else None)
