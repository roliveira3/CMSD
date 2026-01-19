"""
Dataset generation script using the improved graph generator.

Generates datasets with controlled vertex and edge distributions.
Supports multiple edge sampling strategies and tracks global statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm
import numpy as np
import hashlib

try:
    from .graph_generator import GeneratedGraph, GraphGenerator
    from .graph_to_image import GraphToImageRenderer
    from .graph_to_text import GraphToTextConverter
    from .question_generator import QuestionGenerator, QuestionType
except ImportError:
    from graph_generator import GeneratedGraph, GraphGenerator  # type: ignore
    from graph_to_image import GraphToImageRenderer  # type: ignore
    from graph_to_text import GraphToTextConverter  # type: ignore
    from question_generator import QuestionGenerator, QuestionType  # type: ignore


def sample_edge_density_long_tail(rng: random.Random, scale: float = 0.25, min_density: float = 0.1, max_density: float = 0.7) -> float:
    sampled = rng.expovariate(1.0 / scale)
    sampled = min(sampled - min_density, max_density - min_density)
    density = min_density + sampled
    return min(max(density, min_density), max_density)


def hash_graph(graph: GeneratedGraph) -> str:
    """Create canonical hash of graph structure INCLUDING grid positions."""
    # Build vertex position map: (row, col) for each vertex
    vertex_positions = {v.id: (v.row, v.col) for v in graph.vertices}
    
    # Build edge list with actual grid cell positions
    edges = []
    for edge in graph.edges:
        src_pos = vertex_positions[edge.source]
        tgt_pos = vertex_positions[edge.target]
        
        if graph.directed:
            edges.append((src_pos, tgt_pos))
        else:
            # For undirected, sort positions for canonical form
            edges.append(tuple(sorted([src_pos, tgt_pos])))
    
    edges.sort()
    
    # Include grid positions of all vertices + edges
    vertex_str = ','.join(f'{r},{c}' for r, c in sorted(vertex_positions.values()))
    edge_str = ','.join(f'{s[0]},{s[1]}-{t[0]},{t[1]}' for s, t in edges)
    canonical = f"v:{vertex_str}|e:{edge_str}"
    
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced graph datasets with distribution controls."
    )
    parser.add_argument(
        "--questions",
        type=int,
        required=True,
        help="Number of questions to generate per dataset.",
    )
    parser.add_argument(
        "--min-vertices",
        type=int,
        default=4,
        help="Minimum number of vertices per graph.",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=12,
        help="Maximum number of vertices per graph.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=4,
        help="Grid size (4 = 4x4 grid). Must be large enough to fit max vertices.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=112,
        help="Size of each grid cell in pixels (112 = 448x448 image for 4x4 grid).",
    )
    parser.add_argument(
        "--edge-density-mode",
        type=str,
        choices=["fixed", "long_tail"],
        default="long_tail",
        help="Edge density mode: 'fixed' uses constant density, 'long_tail' samples from distribution favoring sparse graphs.",
    )
    parser.add_argument(
        "--edge-density",
        type=float,
        default=0.25,
        help="Fixed edge density value (0.0-1.0) when mode is 'fixed', or scale parameter for 'long_tail' mode.",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=0.1,
        help="Minimum edge density (0.0-1.0) when mode is 'fixed'. Required for 'fixed' mode.",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=0.7,
        help="Maximum edge density (0.0-1.0) when mode is 'fixed'. Required for 'fixed' mode.",
    )
    parser.add_argument(
        "--planarity-prob",
        type=float,
        default=0.92,
        help="Probability of attempting planarity (0.0-1.0). Higher = more planar graphs.",
    )
    parser.add_argument(
        "--enforce-planarity",
        action="store_true",
        help="Enforce 100%% planarity (reject graphs with edge crossings).",
    )
    parser.add_argument(
        "--no-none-of-above",
        action="store_true",
        help="All 4 options will be valid answers (no 'None of the above' option). Tests correctness, not abstention.",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help="Generate directed graphs (arrows show direction).",
    )
    parser.add_argument(
        "--collections-dir",
        type=Path,
        default=Path("collections/balanced"),
        help="Root directory for output datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    # Validate edge density mode arguments
    if args.edge_density_mode == "fixed":
        if args.min_density < 0.0 or args.min_density > 1.0:
            parser.error("--min-density must be between 0.0 and 1.0")
        if args.max_density < 0.0 or args.max_density > 1.0:
            parser.error("--max-density must be between 0.0 and 1.0")
        if args.min_density > args.max_density:
            parser.error("--min-density must be less than or equal to --max-density")

    # Create dataset name
    direction_str = "directed" if args.directed else "undirected"
    planar_str = "_planar" if args.enforce_planarity else ""
    dataset_name = (
        f"v{args.min_vertices}-{args.max_vertices}_"
        f"{direction_str}{planar_str}_c{args.cell_size}"
    )

    collections_dir = args.collections_dir
    collections_dir.mkdir(parents=True, exist_ok=True)

    images_root = collections_dir / "images" / dataset_name
    data_root = collections_dir / "data"

    # Create modality-specific dataset paths
    image_only_path = data_root / f"{dataset_name}_image_only" / "data.jsonl"
    image_text_path = data_root / f"{dataset_name}_image_text" / "data.jsonl"
    text_only_path = data_root / f"{dataset_name}_text_only" / "data.jsonl"
    stats_path = data_root / f"{dataset_name}_stats.json"

    # Check for existing datasets
    if (
        images_root.exists()
        or image_only_path.parent.exists()
        or image_text_path.parent.exists()
        or text_only_path.parent.exists()
    ):
        raise FileExistsError(
            f"Dataset '{dataset_name}' already exists. Remove it before re-running."
        )

    # Create directories
    images_root.mkdir(parents=True, exist_ok=True)
    image_only_path.parent.mkdir(parents=True, exist_ok=True)
    image_text_path.parent.mkdir(parents=True, exist_ok=True)
    text_only_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    graph_generator = GraphGenerator(seed=args.seed)
    question_generator = QuestionGenerator(
        seed=args.seed * 3 + 1,
        force_all_valid_options=args.no_none_of_above  # When no_none_of_above, all 4 options are valid answers
    )
    text_converter = GraphToTextConverter()
    renderer = GraphToImageRenderer(
        output_dir=images_root,
        cell_pixels=args.cell_size,
        supersample=2,
        shape_fill_solid=True,
        shape_scale=1.0,
        shape_stroke_width=3.0,
        arrow_stroke_width=3.0,
        draw_grid=True,
    )

    # Build vertex labels
    vertex_labels = build_vertex_labels("number", args.grid_size)
    entity_label = "number"

    # Statistics tracking
    vertex_count_dist = Counter()
    edge_length_dist = Counter()
    edge_count_dist = Counter()
    edge_direction_dist = Counter()  # Track edge directions/angles
    question_variants = Counter()
    intersection_counts = []  # Track number of intersections per graph
    edge_densities: List[float] = []  # Track sampled edge densities
    seen_graphs = set()  # Track graph hashes for duplicate detection
    
    # Answer edge statistics (what edges the model needs to identify)
    answer_edge_directions = Counter()  # Direction/angle of answer edges
    answer_edge_lengths = Counter()  # Length of answer edges
    answer_vertex_pairs = Counter()  # Track (source, target) pairs for answer edges
    
    # Target ratios for question variants (roughly, not strict)
    TARGET_NONE_RATIO = 0.25  # ~25% none_of_the_above
    RATIO_TOLERANCE = 0.1  # Allow 15-35% range
    accepted_unique = 0
    accepted_none = 0

    # Generation loop
    graph_counter = 0
    question_counter = 0
    attempts = 0
    max_attempts = args.questions * 1000
    rng = random.Random(args.seed)

    with (
        open(image_only_path, "w", encoding="utf-8") as f_img,
        open(image_text_path, "w", encoding="utf-8") as f_img_text,
        open(text_only_path, "w", encoding="utf-8") as f_text,
    ):
        with tqdm(total=args.questions, desc=f"Generating {dataset_name}", unit="q") as pbar:
            while question_counter < args.questions:
                # Sample vertex count once per successful graph (uniform distribution)
                target_num_vertices = rng.randint(args.min_vertices, args.max_vertices)
                
                # Sample edge density once per graph
                if args.edge_density_mode == "long_tail":
                    edge_density = sample_edge_density_long_tail(
                        rng=rng,
                        scale=args.edge_density,
                        min_density=args.min_density,
                        max_density=args.max_density
                    )
                else:
                    # Fixed mode: sample uniformly between min and max density
                    edge_density = rng.uniform(args.min_density, args.max_density)
                
                # Keep retrying with the SAME vertex count and edge density until success
                graph_attempts = 0
                max_graph_attempts = 10000
                while graph_attempts < max_graph_attempts:
                    attempts += 1
                    graph_attempts += 1
                    
                    if attempts > max_attempts:
                        raise RuntimeError(
                            f"Exceeded {max_attempts} total attempts. "
                            f"Generated {question_counter}/{args.questions} questions."
                        )
                    
                    # Generate graph with fixed vertex count
                    try:
                        graph = graph_generator.generate_graph(
                            min_vertices=args.min_vertices,
                            max_vertices=args.max_vertices,
                            grid_size=args.grid_size,
                            directed=args.directed,
                            edge_strategy="uniform",  # Fixed: uniform with slight short-edge preference
                            vertex_placement="uniform",  # Fixed: random uniform placement
                            edge_density=edge_density,
                            strategy_param=None,
                            balance_globally=True,  # Always balance edge lengths globally
                            planarity_prob=args.planarity_prob,
                            num_vertices=target_num_vertices,  # Pass fixed vertex count
                        )
                    except ValueError:
                        continue
                    
                    # Check for duplicate graph structure
                    graph_hash = hash_graph(graph)
                    if graph_hash in seen_graphs:
                        continue
                    
                    # If enforce_planarity is set, reject graphs with any intersections
                    if args.enforce_planarity:
                        num_intersections = count_edge_intersections(graph)
                        if num_intersections > 0:
                            continue

                    # Generate question
                    payload = question_generator.try_generate(
                        graph,
                        vertex_labels,
                        include_details=True,
                        entity_label=entity_label,
                    )
                    if payload is None:
                        continue

                    _, questions_map = payload
                    question = questions_map.get(QuestionType.Q1)
                    if question is None:
                        continue

                    variant = question.variant
                    if variant not in ("unique_neighbor", "none_of_the_above"):
                        continue
                    
                    # When no_none_of_above is True, questions will only be unique_neighbor
                    # (all 4 options are valid answers, no "None of the above")
                    # Otherwise, control the ratio of none_of_the_above questions (roughly ~25%)
                    if not args.no_none_of_above:
                        current_total = accepted_unique + accepted_none
                        if current_total > 20:  # Only enforce after some questions
                            current_none_ratio = accepted_none / current_total
                            
                            # If we're way over the target ratio, skip none_of_the_above
                            if variant == "none_of_the_above" and current_none_ratio > TARGET_NONE_RATIO + RATIO_TOLERANCE:
                                continue
                            
                            # If we're way under, skip unique_neighbor sometimes
                            if variant == "unique_neighbor" and current_none_ratio < TARGET_NONE_RATIO - RATIO_TOLERANCE:
                                if graph_generator._rng.random() < 0.4:
                                    continue
                    
                    # Mark graph as seen (successful generation)
                    seen_graphs.add(graph_hash)
                    
                    # Success! Break out of retry loop
                    break
                
                # Check if we failed after max attempts - resample vertex count instead of failing
                if graph_attempts >= max_graph_attempts:
                    # This vertex count is too hard, skip to next iteration
                    # (will sample a new vertex count)
                    continue
                
                # Track edge density after successful generation
                if args.edge_density_mode == "long_tail":
                    edge_densities.append(edge_density)

                # Generate text description
                text_result = text_converter.convert(
                    graph,
                    vertex_labels,
                    conversion_type="adjacency_list",
                    use_cells=False,
                )

                # Render image
                image_identifier = f"graph_{graph_counter:05d}"
                image_result = renderer.render(
                    graph,
                    vertex_labels,
                    identifier=image_identifier,
                    label_set_hint=entity_label,
                )

                # Prepare question data
                options = [str(opt) for opt in question.options]
                answer_idx = question.correct_index
                question_id = f"q{question_counter:06d}"

                vertex_labels_subset = {
                    vertex.id: vertex_labels[vertex.id] for vertex in graph.vertices
                }
                label_to_vertex = {
                    label: vid for vid, label in vertex_labels_subset.items()
                }
                answer_label = options[answer_idx]
                answer_vertex = (
                    label_to_vertex.get(answer_label)
                    if question.variant == "unique_neighbor"
                    else None
                )

                base_payload = {
                    "question_id": question_id,
                    "options": options,
                    "answer": options[answer_idx],
                    "answer_idx": answer_idx,
                    "question_type": question.question_type.name,
                    "edge_variant": question.variant,
                    "directed": args.directed,
                    "graph": serialize_graph(graph),
                    "vertex_labels": vertex_labels_subset,
                    "source_vertex": question.source_vertex,
                    "answer_vertex": answer_vertex,
                    "num_vertices": len(graph.vertices),
                    "num_edges": len(graph.edges),
                    "cell_size": args.cell_size,
                }

                # Write to image_only
                image_path_rel = os.path.relpath(image_result.image_path, image_only_path.parent)
                image_only_row = {
                    **base_payload,
                    "question": question.prompt,
                    "image": image_path_rel,
                }
                f_img.write(json.dumps(image_only_row, ensure_ascii=False) + "\n")

                # Write to image_text
                text_desc = (text_result.text or "").strip()
                enriched_prompt = (
                    f"{text_desc}\n\n{question.prompt}" if text_desc else question.prompt
                )
                image_path_rel_text = os.path.relpath(
                    image_result.image_path, image_text_path.parent
                )
                image_text_row = {
                    **base_payload,
                    "question": enriched_prompt,
                    "image": image_path_rel_text,
                }
                f_img_text.write(json.dumps(image_text_row, ensure_ascii=False) + "\n")

                # Write to text_only
                text_only_row = {
                    **base_payload,
                    "question": enriched_prompt,
                    "image": "",
                }
                f_text.write(json.dumps(text_only_row, ensure_ascii=False) + "\n")

                # Update statistics
                vertex_count_dist[len(graph.vertices)] += 1
                edge_count_dist[len(graph.edges)] += 1
                question_variants[variant] += 1

                # Track edge lengths and directions
                for edge in graph.edges:
                    source = next(v for v in graph.vertices if v.id == edge.source)
                    target = next(v for v in graph.vertices if v.id == edge.target)
                    length = ((source.row - target.row) ** 2 + (source.col - target.col) ** 2) ** 0.5
                    edge_length_dist[round(length, 2)] += 1
                    
                    # Track edge direction (angle in degrees, binned to 45-degree sectors)
                    import math
                    dx = target.col - source.col
                    dy = target.row - source.row
                    angle = math.degrees(math.atan2(dy, dx))
                    # Normalize to [0, 360) and bin to 45-degree sectors
                    angle = (angle + 360) % 360
                    sector = int(angle) 
                    edge_direction_dist[sector] += 1
                
                # Track answer edge statistics (for unique_neighbor questions)
                if variant == "unique_neighbor" and answer_vertex is not None:
                    # Find the answer edge in the graph
                    source_v = next(v for v in graph.vertices if v.id == question.source_vertex)
                    target_v = next(v for v in graph.vertices if v.id == answer_vertex)
                    
                    # Calculate answer edge properties
                    import math
                    dx = target_v.col - source_v.col
                    dy = target_v.row - source_v.row
                    length = ((dx ** 2) + (dy ** 2)) ** 0.5
                    angle = math.degrees(math.atan2(dy, dx))
                    angle = (angle + 360) % 360
                    sector = int(angle)
                    
                    # Track statistics
                    answer_edge_lengths[round(length, 2)] += 1
                    answer_edge_directions[sector] += 1
                    answer_vertex_pairs[(question.source_vertex, answer_vertex)] += 1
                
                # Count edge intersections
                num_intersections = count_edge_intersections(graph)
                intersection_counts.append(num_intersections)

                graph_counter += 1
                question_counter += 1
                pbar.update(1)
                
                # Update variant counters
                if variant == "unique_neighbor":
                    accepted_unique += 1
                else:
                    accepted_none += 1

    # Save statistics
    stats = {
        "dataset_name": dataset_name,
        "config": {
            "questions": args.questions,
            "min_vertices": args.min_vertices,
            "max_vertices": args.max_vertices,
            "grid_size": args.grid_size,
            "cell_size": args.cell_size,
            "edge_density_mode": args.edge_density_mode,
            "edge_density": args.edge_density,
            "directed": args.directed,
            "planarity_prob": args.planarity_prob,
            "enforce_planarity": args.enforce_planarity,
            "no_none_of_above": args.no_none_of_above,
            "seed": args.seed,
        },
        "statistics": {
            "total_graphs": graph_counter,
            "total_questions": question_counter,
            "total_attempts": attempts,
            "vertex_count_distribution": dict(vertex_count_dist),
            "edge_count_distribution": dict(edge_count_dist),
            "edge_length_distribution": {str(k): v for k, v in edge_length_dist.items()},
            "edge_direction_distribution": dict(edge_direction_dist),
            "question_variants": dict(question_variants),
            "global_edge_length_bins": graph_generator.get_edge_length_stats(),
            "answer_edge_statistics": {
                "directions": dict(answer_edge_directions),
                "lengths": {str(k): v for k, v in answer_edge_lengths.items()},
                "vertex_pairs": {f"{s}->{t}": v for (s, t), v in answer_vertex_pairs.items()},
            },
            "intersections": {
                "mean": sum(intersection_counts) / len(intersection_counts) if intersection_counts else 0,
                "max": max(intersection_counts) if intersection_counts else 0,
                "planar_graphs": sum(1 for x in intersection_counts if x == 0),
                "planar_percentage": 100 * sum(1 for x in intersection_counts if x == 0) / len(intersection_counts) if intersection_counts else 0,
            },
        },
    }
    
    # Add edge density stats if using long_tail mode
    if args.edge_density_mode == "long_tail" and edge_densities:
        stats["statistics"]["edge_densities"] = {
            "mean": float(np.mean(edge_densities)),
            "median": float(np.median(edge_densities)),
            "min": float(np.min(edge_densities)),
            "max": float(np.max(edge_densities)),
            "std": float(np.std(edge_densities)),
        }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Dataset '{dataset_name}' generated successfully!")
    print(f"  Images: {images_root}")
    print(f"  Data: {data_root}")
    print(f"  Stats: {stats_path}")
    print(f"\nStatistics:")
    print(f"  Graphs: {graph_counter}")
    print(f"  Questions: {question_counter}")
    print(f"  Attempts: {attempts}")
    print(f"  Vertex count distribution: {dict(vertex_count_dist)}")
    print(f"  Edge count distribution: {dict(edge_count_dist)}")
    total_edges = sum(k*v for k,v in edge_count_dist.items())
    print(f"  Mean edges per graph: {total_edges / graph_counter:.2f}")
    if args.edge_density_mode == "long_tail" and edge_densities:
        print(f"  Edge density stats: mean={np.mean(edge_densities):.3f}, median={np.median(edge_densities):.3f}, range=[{np.min(edge_densities):.3f}, {np.max(edge_densities):.3f}]")
    print(f"  Question variants: {dict(question_variants)}")
    if intersection_counts:
        planar = sum(1 for x in intersection_counts if x == 0)
        print(f"  Planar graphs: {planar}/{graph_counter} ({100*planar/graph_counter:.1f}%)")


def build_vertex_labels(label_mode: str, grid_size: int) -> Dict[str, str]:
    """Build vertex labels."""
    generator = GraphGenerator()
    vertex_ids = generator.abstract_vertices(grid_size)
    if label_mode == "number":
        return {vertex_id: str(idx + 1) for idx, vertex_id in enumerate(vertex_ids)}
    raise ValueError(f"Unsupported label_mode '{label_mode}'.")


def serialize_graph(graph: GeneratedGraph) -> Dict[str, object]:
    """Serialize graph to dictionary."""
    return {
        "directed": bool(graph.directed),
        "grid_size": list(graph.grid_size),
        "vertices": [
            {"id": v.id, "row": v.row, "col": v.col} for v in graph.vertices
        ],
        "edges": [
            {"source": e.source, "target": e.target} for e in graph.edges
        ],
    }


def count_edge_intersections(graph: GeneratedGraph) -> int:
    """Count the number of edge intersections in a graph."""
    from graph_generator import GraphGenerator
    
    vertices = {v.id: v for v in graph.vertices}
    count = 0
    edges_list = graph.edges
    
    for i in range(len(edges_list)):
        for j in range(i + 1, len(edges_list)):
            e1 = edges_list[i]
            e2 = edges_list[j]
            
            e1_start = vertices[e1.source]
            e1_end = vertices[e1.target]
            e2_start = vertices[e2.source]
            e2_end = vertices[e2.target]
            
            if GraphGenerator._edges_intersect(e1_start, e1_end, e2_start, e2_end):
                count += 1
    
    return count


if __name__ == "__main__":
    main()
