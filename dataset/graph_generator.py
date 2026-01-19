from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Literal, Optional, Tuple

GridSize = Tuple[int, int]
EdgeLengthStrategy = Literal["uniform", "long_tail", "cutoff"]
VertexPlacementStrategy = Literal["uniform", "clustered", "spread"]


@dataclass(frozen=True)
class Vertex:
    id: str
    row: int
    col: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class GeneratedGraph:
    directed: bool
    grid_size: GridSize
    vertices: List[Vertex]
    edges: List[Edge]
    adjacency: Dict[str, List[str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "directed": self.directed,
            "grid_size": list(self.grid_size),
            "vertices": [vertex.to_dict() for vertex in self.vertices],
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": self.adjacency,
        }


class EdgeLengthTracker:
    def __init__(self) -> None:
        self.bin_counts: Counter[int] = Counter()
        self.total_edges: int = 0

    def record_edges(self, edge_bins: Iterable[int]) -> None:
        for bin_idx in edge_bins:
            self.bin_counts[bin_idx] += 1
            self.total_edges += 1

    def get_correction_weight(self, bin_idx: int) -> float:
        if self.total_edges == 0:
            return 1.0
        current_count = self.bin_counts.get(bin_idx, 0)
        return 1.0 / max(current_count + 1, 1)


class GraphGenerator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._edge_length_tracker = EdgeLengthTracker()
        self._length_bins: Dict[GridSize, List[float]] = {}

    def generate_graph(
        self,
        min_vertices: int,
        max_vertices: int,
        grid_size: int | GridSize,
        directed: bool = False,
        edge_strategy: EdgeLengthStrategy = "uniform",
        vertex_placement: VertexPlacementStrategy = "uniform",
        edge_density: float = 0.5,
        strategy_param: Optional[float] = None,
        balance_globally: bool = True,
        planarity_prob: float = 0.9,
        num_vertices: Optional[int] = None,
    ) -> GeneratedGraph:
        rows, cols = self._normalize_grid_size(grid_size)
        
        if num_vertices is None:
            num_vertices = self._rng.randint(min_vertices, max_vertices)
        elif not (min_vertices <= num_vertices <= max_vertices):
            raise ValueError(f"num_vertices {num_vertices} must be in [{min_vertices}, {max_vertices}]")
        
        if not (1 <= num_vertices <= rows * cols):
            raise ValueError(f"num_vertices must be in [1, {rows * cols}]")
        
        vertices = self._create_vertices(num_vertices, rows, cols, vertex_placement)
        
        grid_key = (rows, cols)
        if grid_key not in self._length_bins:
            self._length_bins[grid_key] = self._compute_length_bins(rows, cols)
        length_bins = self._length_bins[grid_key]
        
        edges, adjacency = self._create_edges(
            vertices, directed, edge_strategy, edge_density, length_bins,
            strategy_param, balance_globally, planarity_prob
        )
        
        # Apply random rotation/reflection to balance directions without changing properties
        if directed or True:  # Apply to both directed and undirected for consistency
            vertices, edges, adjacency = self._apply_random_transform(
                vertices, edges, adjacency, rows, cols, directed
            )
        
        return GeneratedGraph(directed, (rows, cols), vertices, edges, adjacency)

    def reset_global_tracker(self) -> None:
        self._edge_length_tracker = EdgeLengthTracker()

    def get_edge_length_stats(self) -> Dict[int, int]:
        return dict(self._edge_length_tracker.bin_counts)

    def abstract_vertices(self, grid_size: int | GridSize) -> List[str]:
        rows, cols = self._normalize_grid_size(grid_size)
        return [f"v{idx}" for idx in range(rows * cols)]

    def _apply_random_transform(
        self, vertices: List[Vertex], edges: List[Edge], 
        adjacency: Dict[str, List[str]], rows: int, cols: int, directed: bool
    ) -> Tuple[List[Vertex], List[Edge], Dict[str, List[str]]]:
        """Apply random rotation/reflection to balance directions."""
        # 8 possible transformations: 4 rotations × 2 reflections
        rotation = self._rng.choice([0, 90, 180, 270])
        reflect = self._rng.choice([False, True])
        
        # Transform vertices
        new_vertices = []
        for v in vertices:
            r, c = v.row, v.col
            
            if reflect:
                c = cols - 1 - c
            
            if rotation == 90:
                r, c = c, rows - 1 - r
            elif rotation == 180:
                r, c = rows - 1 - r, cols - 1 - c
            elif rotation == 270:
                r, c = cols - 1 - c, r
            
            new_vertices.append(Vertex(id=v.id, row=r, col=c))
        
        return new_vertices, edges, adjacency

    def _normalize_grid_size(self, grid_size: int | GridSize) -> GridSize:
        if isinstance(grid_size, int):
            return grid_size, grid_size
        return grid_size

    def _compute_length_bins(self, rows: int, cols: int) -> List[float]:
        distances = set()
        for r1 in range(rows):
            for c1 in range(cols):
                for r2 in range(rows):
                    for c2 in range(cols):
                        if r1 != r2 or c1 != c2:
                            distances.add(math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))
        return sorted(distances)

    def _create_vertices(
        self, num_vertices: int, rows: int, cols: int, placement: VertexPlacementStrategy
    ) -> List[Vertex]:
        available_cells = [(r, c) for r in range(rows) for c in range(cols)]

        if placement == "uniform":
            selected = self._rng.sample(available_cells, num_vertices)
        elif placement == "clustered":
            selected = self._sample_weighted(available_cells, num_vertices, closer_better=True)
        elif placement == "spread":
            selected = self._sample_weighted(available_cells, num_vertices, closer_better=False)
        else:
            raise ValueError(f"Unknown placement: {placement}")

        selected.sort()
        return [Vertex(id=f"v{idx}", row=r, col=c) for idx, (r, c) in enumerate(selected)]

    def _sample_weighted(
        self, cells: List[Tuple[int, int]], num: int, closer_better: bool
    ) -> List[Tuple[int, int]]:
        if num == 0:
            return []
        
        selected = [self._rng.choice(cells)]
        cells = [c for c in cells if c != selected[0]]
        
        for _ in range(num - 1):
            if not cells:
                break
            weights = []
            for r, c in cells:
                min_dist = min(math.sqrt((r - sr) ** 2 + (c - sc) ** 2) for sr, sc in selected)
                weights.append((1.0 / (min_dist + 0.5)) if closer_better else (min_dist + 0.1))
            
            total = sum(weights)
            r = self._rng.random() * total
            cumsum = 0.0
            for idx, w in enumerate(weights):
                cumsum += w
                if r < cumsum:
                    selected.append(cells[idx])
                    cells.pop(idx)
                    break
        
        return selected

    def _create_edges(
        self, vertices: List[Vertex], directed: bool, edge_strategy: EdgeLengthStrategy,
        edge_density: float, length_bins: List[float], strategy_param: Optional[float],
        balance_globally: bool, planarity_prob: float
    ) -> Tuple[List[Edge], Dict[str, List[str]]]:
        adjacency: Dict[str, List[str]] = {v.id: [] for v in vertices}
        edges: List[Edge] = []
        
        if len(vertices) <= 1:
            return edges, adjacency
        
        vertex_lookup = {v.id: v for v in vertices}
        candidates = []
        
        for i, source in enumerate(vertices):
            start_j = 0 if directed else i + 1
            for target in vertices[start_j:]:
                if source.id != target.id and not self._passes_through_vertex(source, target, vertex_lookup):
                    candidates.append((source.id, target.id))
        
        if not candidates:
            return edges, adjacency
        
        weights = self._compute_edge_weights(
            candidates, vertices, edge_strategy, length_bins,
            strategy_param, balance_globally
        )
        
        num_vertices = len(vertices)
        
        # Simple target: use edge_density fraction of candidates, ensure connectivity
        min_edges = num_vertices - 1  # Minimum for connectivity
        target_edges = int(edge_density * len(candidates))
        target_edges = max(min_edges, min(target_edges, len(candidates)))
        
        sampled_indices = self._weighted_sample(weights, target_edges)
        sampled_bins = []
        enforce_planarity = self._rng.random() < planarity_prob
        
        for idx in sampled_indices:
            source_id, target_id = candidates[idx]
            source = vertex_lookup[source_id]
            target = vertex_lookup[target_id]
            
            if enforce_planarity:
                intersects = any(
                    self._edges_intersect(source, target, vertex_lookup[e.source], vertex_lookup[e.target])
                    for e in edges
                )
                if intersects and len(edges) >= len(vertices) - 1:
                    continue
            
            edges.append(Edge(source=source_id, target=target_id))
            adjacency[source_id].append(target_id)
            if not directed:
                adjacency[target_id].append(source_id)
            
            length = math.sqrt((source.row - target.row) ** 2 + (source.col - target.col) ** 2)
            sampled_bins.append(self._find_bin_index(length, length_bins))
        
        if balance_globally:
            self._edge_length_tracker.record_edges(sampled_bins)
        
        return edges, adjacency

    def _compute_edge_weights(
        self, candidates: List[Tuple[str, str]], vertices: List[Vertex],
        edge_strategy: EdgeLengthStrategy, length_bins: List[float],
        strategy_param: Optional[float], balance_globally: bool
    ) -> List[float]:
        vertex_lookup = {v.id: v for v in vertices}
        max_length = length_bins[-1] if length_bins else 1.0
        
        # First pass: group candidates by edge length bin for uniform strategy
        bin_to_count: Dict[int, int] = {}
        candidate_bins: List[int] = []
        
        for source_id, target_id in candidates:
            source = vertex_lookup[source_id]
            target = vertex_lookup[target_id]
            length = math.sqrt((source.row - target.row) ** 2 + (source.col - target.col) ** 2)
            bin_idx = self._find_bin_index(length, length_bins)
            candidate_bins.append(bin_idx)
            bin_to_count[bin_idx] = bin_to_count.get(bin_idx, 0) + 1
        
        # Second pass: compute weights
        weights = []
        for idx, (source_id, target_id) in enumerate(candidates):
            source = vertex_lookup[source_id]
            target = vertex_lookup[target_id]
            length = math.sqrt((source.row - target.row) ** 2 + (source.col - target.col) ** 2)
            bin_idx = candidate_bins[idx]
            
            if edge_strategy == "uniform":
                # Each length bin gets equal probability, uniform within bin
                # Weight = 1 / (num_bins * num_in_bin)
                base_weight = 1.0 / bin_to_count[bin_idx]
            elif edge_strategy == "long_tail":
                decay = strategy_param if strategy_param is not None else 0.3
                base_weight = math.exp(-decay * (length / max_length) * 10)
            elif edge_strategy == "cutoff":
                cutoff_frac = strategy_param if strategy_param is not None else 0.5
                base_weight = 1.0 if length <= cutoff_frac * max_length else 0.0
            else:
                base_weight = 1.0
            
            if balance_globally:
                base_weight *= self._edge_length_tracker.get_correction_weight(bin_idx)
            
            weights.append(base_weight)
        
        return weights

    def _find_bin_index(self, length: float, length_bins: List[float]) -> int:
        min_diff, min_idx = float('inf'), 0
        for idx, bin_length in enumerate(length_bins):
            if abs(length - bin_length) < 1e-6:
                return idx
            diff = abs(length - bin_length)
            if diff < min_diff:
                min_diff, min_idx = diff, idx
        return min_idx

    def _weighted_sample(self, weights: List[float], k: int) -> List[int]:
        if sum(weights) == 0:
            return self._rng.sample(range(len(weights)), k)
        
        indices_and_keys = [
            (self._rng.random() ** (1.0 / w), idx)
            for idx, w in enumerate(weights) if w > 0
        ]
        indices_and_keys.sort(reverse=True)
        return [idx for _, idx in indices_and_keys[:k]]

    def _passes_through_vertex(
        self, source: Vertex, target: Vertex, vertex_lookup: Dict[str, Vertex], tolerance: float = 0.1
    ) -> bool:
        sx, sy, tx, ty = source.col, source.row, target.col, target.row
        dx, dy = tx - sx, ty - sy
        
        for vertex in vertex_lookup.values():
            if vertex.id in (source.id, target.id):
                continue
            px, py = vertex.col, vertex.row
            
            if dx == dy == 0:
                if math.hypot(px - sx, py - sy) <= tolerance:
                    return True
                continue
            
            t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
            if 0 < t < 1:
                proj_x, proj_y = sx + t * dx, sy + t * dy
                if math.hypot(px - proj_x, py - proj_y) <= tolerance:
                    return True
        return False

    @staticmethod
    def _edges_intersect(e1_start: Vertex, e1_end: Vertex, e2_start: Vertex, e2_end: Vertex) -> bool:
        x1, y1 = e1_start.col, e1_start.row
        x2, y2 = e1_end.col, e1_end.row
        x3, y3 = e2_start.col, e2_start.row
        x4, y4 = e2_end.col, e2_end.row
        
        if any(v1 == v2 for v1, v2 in [
            (e1_start.id, e2_start.id), (e1_start.id, e2_end.id),
            (e1_end.id, e2_start.id), (e1_end.id, e2_end.id)
        ]):
            return False
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        eps = 0.001
        return (eps < t < 1 - eps) and (eps < u < 1 - eps)
