#!/usr/bin/env python3
"""
Split a 4×4 grid graph dataset into train/val/test splits while avoiding 
data leakage from symmetric duplicates using canonical pattern IDs.

The canonical ID is computed by:
1. Normalizing translation (shifting pattern to top-left corner)
2. Generating all 8 D4 symmetries (rotations + flips) 
3. Serializing each variant and returning the lexicographically smallest

This ensures all symmetric versions of the same pattern get the same canonical ID,
and the split is done by groups (canonical IDs) rather than individual samples.

sbatch --ntasks=1 --cpus-per-task=2 --mem-per-cpu=8G  --time=00:59:00 --wrap="python split_dataset_canonical.py --input collections/balanced/data/v4-10_undirected_planar_c112_image_only/data.jsonl --output collections/graph_dataset_4x4 --train-target 30000 --test-target 5000 --seed 42"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict


# Type aliases
GridPosition = Tuple[int, int]
Edge = Tuple[GridPosition, GridPosition]


@dataclass
class Graph4x4:
    """Represents a graph on a 4×4 grid."""
    grid_size: Tuple[int, int]
    occupied_cells: Set[GridPosition]
    edges: Set[Edge]
    
    @classmethod
    def from_dict(cls, graph_dict: Dict) -> "Graph4x4":
        """Create Graph4x4 from the dataset's graph dictionary format."""
        grid_size = tuple(graph_dict["grid_size"])
        
        # Extract vertex positions
        occupied_cells = set()
        id_to_pos = {}
        for v in graph_dict["vertices"]:
            pos = (v["row"], v["col"])
            occupied_cells.add(pos)
            id_to_pos[v["id"]] = pos
        
        # Extract edges (normalized: smaller position first for undirected)
        edges = set()
        for e in graph_dict["edges"]:
            src_pos = id_to_pos[e["source"]]
            tgt_pos = id_to_pos[e["target"]]
            # Normalize edge representation (smaller tuple first)
            edge = tuple(sorted([src_pos, tgt_pos]))
            edges.add(edge)
        
        return cls(grid_size=grid_size, occupied_cells=occupied_cells, edges=edges)
    
    def to_canonical_representation(self) -> Tuple[Tuple[GridPosition, ...], Tuple[Edge, ...]]:
        """Return a canonical tuple representation for hashing/comparison."""
        sorted_cells = tuple(sorted(self.occupied_cells))
        sorted_edges = tuple(sorted(self.edges))
        return (sorted_cells, sorted_edges)


class D4SymmetryHandler:
    """
    Handles D4 symmetry group operations (rotations and reflections) on a 4×4 grid.
    D4 = {identity, r90, r180, r270, flip_h, flip_v, flip_d1, flip_d2}
    """
    
    def __init__(self, grid_rows: int = 4, grid_cols: int = 4):
        self.rows = grid_rows
        self.cols = grid_cols
    
    def _rotate_90_cw(self, pos: GridPosition) -> GridPosition:
        """Rotate position 90° clockwise around grid center."""
        r, c = pos
        # For a 4×4 grid: (r, c) -> (c, 3-r)
        return (c, self.rows - 1 - r)
    
    def _flip_horizontal(self, pos: GridPosition) -> GridPosition:
        """Flip position horizontally (left-right)."""
        r, c = pos
        return (r, self.cols - 1 - c)
    
    def _flip_vertical(self, pos: GridPosition) -> GridPosition:
        """Flip position vertically (top-bottom)."""
        r, c = pos
        return (self.rows - 1 - r, c)
    
    def _apply_transform_to_pos(self, pos: GridPosition, transform_id: int) -> GridPosition:
        """Apply one of 8 D4 transforms to a position.
        
        Transforms:
        0: identity
        1: rotate 90° CW
        2: rotate 180°
        3: rotate 270° CW
        4: flip horizontal
        5: flip vertical  
        6: flip horizontal + rotate 90° CW (diagonal flip)
        7: flip vertical + rotate 90° CW (anti-diagonal flip)
        """
        r, c = pos
        
        if transform_id == 0:  # Identity
            return (r, c)
        elif transform_id == 1:  # Rotate 90° CW
            return self._rotate_90_cw((r, c))
        elif transform_id == 2:  # Rotate 180°
            return self._rotate_90_cw(self._rotate_90_cw((r, c)))
        elif transform_id == 3:  # Rotate 270° CW
            return self._rotate_90_cw(self._rotate_90_cw(self._rotate_90_cw((r, c))))
        elif transform_id == 4:  # Flip horizontal
            return self._flip_horizontal((r, c))
        elif transform_id == 5:  # Flip vertical
            return self._flip_vertical((r, c))
        elif transform_id == 6:  # Flip horizontal then rotate 90° CW
            return self._rotate_90_cw(self._flip_horizontal((r, c)))
        elif transform_id == 7:  # Flip vertical then rotate 90° CW
            return self._rotate_90_cw(self._flip_vertical((r, c)))
        else:
            raise ValueError(f"Unknown transform_id: {transform_id}")
    
    def apply_transform(self, graph: Graph4x4, transform_id: int) -> Graph4x4:
        """Apply a D4 transform to the entire graph."""
        new_cells = {self._apply_transform_to_pos(p, transform_id) for p in graph.occupied_cells}
        new_edges = set()
        for e in graph.edges:
            p1, p2 = e
            new_p1 = self._apply_transform_to_pos(p1, transform_id)
            new_p2 = self._apply_transform_to_pos(p2, transform_id)
            new_edges.add(tuple(sorted([new_p1, new_p2])))
        
        return Graph4x4(grid_size=graph.grid_size, occupied_cells=new_cells, edges=new_edges)
    
    def generate_all_transforms(self, graph: Graph4x4) -> List[Graph4x4]:
        """Generate all 8 D4 symmetric versions of the graph."""
        return [self.apply_transform(graph, i) for i in range(8)]


class TranslationNormalizer:
    """Normalizes translation by shifting the pattern to the top-left corner."""
    
    @staticmethod
    def normalize(graph: Graph4x4) -> Graph4x4:
        """Shift the graph so that the top-left occupied cell is at (0, 0)."""
        if not graph.occupied_cells:
            return graph
        
        # Find the minimum row and column
        min_row = min(p[0] for p in graph.occupied_cells)
        min_col = min(p[1] for p in graph.occupied_cells)
        
        # Shift all positions
        new_cells = {(r - min_row, c - min_col) for r, c in graph.occupied_cells}
        new_edges = set()
        for e in graph.edges:
            (r1, c1), (r2, c2) = e
            new_p1 = (r1 - min_row, c1 - min_col)
            new_p2 = (r2 - min_row, c2 - min_col)
            new_edges.add(tuple(sorted([new_p1, new_p2])))
        
        return Graph4x4(grid_size=graph.grid_size, occupied_cells=new_cells, edges=new_edges)


class CanonicalIDGenerator:
    """
    Generates canonical IDs for graphs that are invariant under D4 symmetry
    and translation.
    """
    
    def __init__(self, use_translation_normalization: bool = True):
        self.symmetry_handler = D4SymmetryHandler()
        self.translation_normalizer = TranslationNormalizer()
        self.use_translation = use_translation_normalization
    
    def _serialize_graph(self, graph: Graph4x4) -> str:
        """Serialize a graph to a string representation."""
        cells_str = ",".join(f"({r},{c})" for r, c in sorted(graph.occupied_cells))
        edges_str = ",".join(
            f"({e[0][0]},{e[0][1]})-({e[1][0]},{e[1][1]})" 
            for e in sorted(graph.edges)
        )
        return f"cells:[{cells_str}]|edges:[{edges_str}]"
    
    def compute_canonical_id(self, graph: Graph4x4) -> str:
        """
        Compute the canonical ID for a graph.
        
        Algorithm:
        1. Normalize translation (shift to top-left)
        2. Generate all 8 D4 symmetric versions
        3. Serialize each version
        4. Return the lexicographically smallest as the canonical ID
        """
        # Step 1: Normalize translation
        if self.use_translation:
            normalized = self.translation_normalizer.normalize(graph)
        else:
            normalized = graph
        
        # Step 2: Generate all D4 symmetric versions
        all_variants = self.symmetry_handler.generate_all_transforms(normalized)
        
        # Step 3: For each variant, also normalize translation and serialize
        serialized = []
        for variant in all_variants:
            if self.use_translation:
                variant = self.translation_normalizer.normalize(variant)
            serialized.append(self._serialize_graph(variant))
        
        # Step 4: Return lexicographically smallest
        canonical = min(serialized)
        
        # Create a shorter hash for practical use
        hash_id = hashlib.md5(canonical.encode()).hexdigest()[:16]
        
        return hash_id
    
    def compute_canonical_id_from_dict(self, graph_dict: Dict) -> str:
        """Convenience method to compute canonical ID directly from graph dict."""
        graph = Graph4x4.from_dict(graph_dict)
        return self.compute_canonical_id(graph)


class DatasetSplitter:
    """
    Splits a dataset by canonical IDs to avoid data leakage from symmetric patterns.
    """
    
    def __init__(
        self,
        train_target: int = 30000,
        test_target: int = 5000,
        seed: int = 42,
        use_translation_normalization: bool = True
    ):
        self.train_target = train_target
        self.test_target = test_target
        self.seed = seed
        self.canonical_id_gen = CanonicalIDGenerator(use_translation_normalization)
    
    def split(
        self, 
        samples: List[Dict]
    ) -> Tuple[List[int], List[int], List[int], Dict[str, str]]:
        """
        Split samples into train/val/test sets by canonical ID groups.
        
        Args:
            samples: List of sample dictionaries, each containing a 'graph' key
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices, index_to_canonical_id)
        """
        print("Computing canonical IDs for all samples...")
        
        # Step 1: Compute canonical ID for each sample
        index_to_canonical_id: Dict[int, str] = {}
        canonical_id_to_indices: Dict[str, List[int]] = defaultdict(list)
        
        for idx, sample in enumerate(samples):
            if idx % 5000 == 0:
                print(f"  Processing sample {idx}/{len(samples)}...")
            
            canonical_id = self.canonical_id_gen.compute_canonical_id_from_dict(sample["graph"])
            index_to_canonical_id[idx] = canonical_id
            canonical_id_to_indices[canonical_id].append(idx)
        
        print(f"Found {len(canonical_id_to_indices)} unique canonical IDs from {len(samples)} samples")
        
        # Step 2: Get list of unique canonical IDs and shuffle
        unique_ids = list(canonical_id_to_indices.keys())
        rng = random.Random(self.seed)
        rng.shuffle(unique_ids)
        
        # Step 3: Assign groups to splits
        train_indices: List[int] = []
        test_indices: List[int] = []
        val_indices: List[int] = []
        
        train_ids: Set[str] = set()
        test_ids: Set[str] = set()
        val_ids: Set[str] = set()
        
        for canonical_id in unique_ids:
            group_indices = canonical_id_to_indices[canonical_id]
            
            if len(train_indices) < self.train_target:
                train_indices.extend(group_indices)
                train_ids.add(canonical_id)
            elif len(test_indices) < self.test_target:
                test_indices.extend(group_indices)
                test_ids.add(canonical_id)
            else:
                val_indices.extend(group_indices)
                val_ids.add(canonical_id)
        
        # Step 4: Sanity checks
        self._validate_splits(
            train_indices, val_indices, test_indices,
            train_ids, val_ids, test_ids,
            index_to_canonical_id
        )
        
        # Convert index_to_canonical_id to string keys for JSON serialization
        index_to_canonical_id_str = {str(k): v for k, v in index_to_canonical_id.items()}
        
        return train_indices, val_indices, test_indices, index_to_canonical_id_str
    
    def _validate_splits(
        self,
        train_indices: List[int],
        val_indices: List[int], 
        test_indices: List[int],
        train_ids: Set[str],
        val_ids: Set[str],
        test_ids: Set[str],
        index_to_canonical_id: Dict[int, str]
    ) -> None:
        """Perform sanity checks on the splits."""
        print("\nValidating splits...")
        
        # Check 1: No overlapping indices
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0, "Train and val indices overlap!"
        assert len(train_set & test_set) == 0, "Train and test indices overlap!"
        assert len(val_set & test_set) == 0, "Val and test indices overlap!"
        print("  ✓ No overlapping indices between splits")
        
        # Check 2: No overlapping canonical IDs
        assert len(train_ids & val_ids) == 0, "Train and val canonical IDs overlap!"
        assert len(train_ids & test_ids) == 0, "Train and test canonical IDs overlap!"
        assert len(val_ids & test_ids) == 0, "Val and test canonical IDs overlap!"
        print("  ✓ No overlapping canonical IDs between splits")
        
        # Check 3: Each canonical ID appears in exactly one split
        all_ids = train_ids | val_ids | test_ids
        assert len(all_ids) == len(train_ids) + len(val_ids) + len(test_ids), \
            "Canonical IDs are not mutually exclusive!"
        print("  ✓ Each canonical ID appears in exactly one split")
        
        # Check 4: Verify indices use correct canonical IDs for their split
        for idx in train_indices:
            assert index_to_canonical_id[idx] in train_ids, \
                f"Train index {idx} has canonical ID not in train_ids!"
        for idx in val_indices:
            assert index_to_canonical_id[idx] in val_ids, \
                f"Val index {idx} has canonical ID not in val_ids!"
        for idx in test_indices:
            assert index_to_canonical_id[idx] in test_ids, \
                f"Test index {idx} has canonical ID not in test_ids!"
        print("  ✓ All indices correctly assigned to their canonical ID group's split")
        
        # Summary
        print(f"\nSplit summary:")
        print(f"  Train: {len(train_indices)} samples ({len(train_ids)} unique patterns)")
        print(f"  Val:   {len(val_indices)} samples ({len(val_ids)} unique patterns)")
        print(f"  Test:  {len(test_indices)} samples ({len(test_ids)} unique patterns)")
        print(f"  Total: {len(train_indices) + len(val_indices) + len(test_indices)} samples")


def generate_text_prompt(sample: Dict) -> str:
    """
    Generate the text-only prompt for a sample.
    Uses the adjacency list format matching the text_only dataset format.
    """
    graph_dict = sample["graph"]
    vertex_labels = sample.get("vertex_labels", {})
    directed = graph_dict.get("directed", False)
    
    # Build adjacency from edges
    adjacency = defaultdict(list)
    for edge in graph_dict["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        adjacency[src].append(tgt)
        if not directed:
            adjacency[tgt].append(src)
    
    # Get vertex IDs in order
    vertex_ids = [v["id"] for v in graph_dict["vertices"]]
    
    # Build adjacency list text
    intro = "Below we describe a diagram which consists of entities which can be connected to each other.\n"
    intro += "The diagram connections are listed below. each entity is followed by the list of entities it directly connects to.\n"
    intro += f"The connections are: {'Oneway' if directed else 'Bothways'}.\n"
    intro += "Connections:\n\n"
    
    lines = []
    for vid in vertex_ids:
        label = vertex_labels.get(vid, vid)
        neighbors = adjacency.get(vid, [])
        neighbor_labels = [vertex_labels.get(n, n) for n in neighbors]
        lines.append(f"{label}:[{','.join(neighbor_labels)}]")
    
    adjacency_text = intro + "\n".join(lines)
    
    # Get the original question (just the question part, without graph description)
    original_question = sample.get("question", "")
    
    # For image_only, the question is short. For text, we need to combine
    # Find where the actual question starts (after graph description)
    if "Which" in original_question:
        question_part = original_question[original_question.find("Which"):]
    elif "What" in original_question:
        question_part = original_question[original_question.find("What"):]
    elif "How" in original_question:
        question_part = original_question[original_question.find("How"):]
    else:
        # Fallback: use the whole question
        question_part = original_question
    
    # Combine graph description with question
    text_prompt = adjacency_text + "\n\n" + question_part
    
    return text_prompt


def process_dataset(
    input_path: str,
    output_dir: str,
    train_target: int = 30000,
    test_target: int = 5000,
    seed: int = 42,
    use_translation_normalization: bool = True,
    copy_images: bool = True
) -> None:
    """
    Process the dataset: compute canonical IDs, split, add text prompts, and save.
    
    Creates a folder structure:
    output_dir/
        train/
            data.jsonl
            images/
        validation/
            data.jsonl
            images/
        test/
            data.jsonl
            images/
        metadata.json
        canonical_id_mapping.json
    
    Args:
        input_path: Path to input data.jsonl
        output_dir: Directory to save output files
        train_target: Target number of training samples
        test_target: Target number of test samples
        seed: Random seed for reproducibility
        use_translation_normalization: Whether to normalize translations
        copy_images: Whether to copy images to split folders
    """
    # Load data
    print(f"Loading data from {input_path}...")
    samples = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples")
    
    # Create splitter and compute splits
    splitter = DatasetSplitter(
        train_target=train_target,
        test_target=test_target,
        seed=seed,
        use_translation_normalization=use_translation_normalization
    )
    
    train_indices, val_indices, test_indices, index_to_canonical_id = splitter.split(samples)
    
    # Determine the source directory for images
    input_dir = os.path.dirname(input_path)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    splits = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices
    }
    
    print("\nCreating folder structure and processing splits...")
    
    for split_name, indices in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        images_dir = os.path.join(split_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        output_path = os.path.join(split_dir, "data.jsonl")
        print(f"\n  Processing {split_name} split ({len(indices)} samples)...")
        
        with open(output_path, 'w') as f:
            for i, idx in enumerate(indices):
                if i % 5000 == 0 and i > 0:
                    print(f"    Processed {i}/{len(indices)} samples...")
                
                sample = samples[idx].copy()
                
                # Add text-only prompt
                sample["text_prompt"] = generate_text_prompt(sample)
                
                # Add canonical ID
                sample["canonical_id"] = index_to_canonical_id[str(idx)]
                
                # Add split information
                sample["split"] = split_name
                
                # Handle image path
                original_image_path = sample.get("image", "")
                if original_image_path and copy_images:
                    # Resolve the source image path relative to input directory
                    source_image_path = os.path.normpath(
                        os.path.join(input_dir, original_image_path)
                    )
                    
                    # Get just the image filename
                    image_filename = os.path.basename(original_image_path)
                    
                    # Create new image path (relative to split's data.jsonl)
                    new_image_path = os.path.join("images", image_filename)
                    
                    # Copy image if source exists
                    dest_image_path = os.path.join(split_dir, new_image_path)
                    if os.path.exists(source_image_path) and not os.path.exists(dest_image_path):
                        shutil.copy2(source_image_path, dest_image_path)
                    
                    # Update the image path in sample
                    sample["image"] = new_image_path
                
                f.write(json.dumps(sample) + "\n")
        
        print(f"    Saved to {output_path}")
    
    # Save metadata
    metadata = {
        "source_file": input_path,
        "total_samples": len(samples),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "train_target": train_target,
        "test_target": test_target,
        "seed": seed,
        "use_translation_normalization": use_translation_normalization,
        "unique_canonical_ids": len(set(index_to_canonical_id.values())),
        "structure": {
            "train": "train/data.jsonl + train/images/",
            "validation": "validation/data.jsonl + validation/images/",
            "test": "test/data.jsonl + test/images/"
        }
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")
    
    # Save canonical ID mapping
    canonical_id_path = os.path.join(output_dir, "canonical_id_mapping.json")
    with open(canonical_id_path, 'w') as f:
        json.dump(index_to_canonical_id, f)
    print(f"Saved canonical ID mapping to {canonical_id_path}")
    
    print("\n" + "="*60)
    print("Dataset processing complete!")
    print(f"Output directory: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Split 4×4 grid graph dataset with canonical ID deduplication"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input data.jsonl file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for split files"
    )
    parser.add_argument(
        "--train-target",
        type=int,
        default=30000,
        help="Target number of training samples (default: 30000)"
    )
    parser.add_argument(
        "--test-target",
        type=int,
        default=5000,
        help="Target number of test samples (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-translation-norm",
        action="store_true",
        help="Disable translation normalization (default: enabled)"
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy images to split folders (default: copy images)"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_path=args.input,
        output_dir=args.output,
        train_target=args.train_target,
        test_target=args.test_target,
        seed=args.seed,
        use_translation_normalization=not args.no_translation_norm,
        copy_images=not args.no_copy_images
    )


if __name__ == "__main__":
    main()
