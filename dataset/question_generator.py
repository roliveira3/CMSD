from __future__ import annotations

import random
import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

TOTAL_OPTIONS = 4  # Includes the trailing "None" option.

if TYPE_CHECKING:  # Avoid importing graph generator at runtime.
    try:
        from .graph_generator import GeneratedGraph
    except ImportError:  # pragma: no cover
        from graph_generator import GeneratedGraph  # type: ignore
else:
    GeneratedGraph = Any  # type: ignore


class QuestionType(Enum):
    Q1 = auto()
    Q2 = auto()
    Q3 = auto()


@dataclass
class Question:
    question_type: QuestionType
    prompt: str
    options: List[str]
    correct_index: int

    def answers_vector(self) -> List[bool]:
        if len(self.options) != TOTAL_OPTIONS:
            raise ValueError(
                f"Each question must expose exactly {TOTAL_OPTIONS} answer options."
            )
        return [idx == self.correct_index for idx in range(len(self.options))]


@dataclass
class Question1(Question):
    source_vertex: str
    variant: str  # "unique_neighbor" | "none_of_the_above"
    target_length_steps: int
    target_length_euclidean: float


@dataclass
class Question2(Question):
    source_vertex: str
    variant: str  # "reachable" | "none_of_the_above"


@dataclass
class Question3(Question):
    source_vertex: str
    steps: int
    variant: str  # "distance_match" | "none_of_the_above"


class QuestionGenerator:
    """Generates the full QA payload for a graph, skipping invalid question types."""

    NUM_OPTIONS = TOTAL_OPTIONS
    NONE_OPTION = "None of the above"

    def __init__(self, seed: Optional[int] = None, force_all_valid_options: bool = False) -> None:
        self._rng = random.Random(seed)
        self._force_all_valid = force_all_valid_options

    def generate(
        self,
        graph: GeneratedGraph,
        vertex_labels: Dict[str, str],
        *,
        path_length: Optional[int] = None,
        include_details: bool = False,
        entity_label: str = "vertex",
    ) -> (
        Dict[QuestionType, List[bool]]
        | Tuple[Dict[QuestionType, List[bool]], Dict[QuestionType, Question]]
    ):
        """Generate all currently supported questions for a graph.

        Raises:
            ValueError: If none of the question types can be instantiated for the graph.
        """
        entity_label = self._normalize_entity_label(entity_label)
        vertex_ids = self._collect_vertex_ids(graph)
        positions = self._collect_positions(graph)
        adjacency = self._collect_adjacency(graph, vertex_ids)
        distance_cache = self._build_distance_cache(vertex_ids, adjacency)

        questions: Dict[QuestionType, Question] = {}
        errors: Dict[QuestionType, str] = {}

        for question_type, builder in (
            (
                QuestionType.Q1,
                lambda: self._generate_q1(
                    vertex_ids,
                    adjacency,
                    vertex_labels,
                    positions,
                    entity_label,
                ),
            ),
            (
                QuestionType.Q2,
                lambda: self._generate_q2(
                    vertex_ids, adjacency, distance_cache, vertex_labels, entity_label
                ),
            ),
            (
                QuestionType.Q3,
                lambda: self._generate_q3(
                    vertex_ids,
                    adjacency,
                    distance_cache,
                    vertex_labels,
                    path_length,
                    entity_label,
                ),
            ),
        ):
            try:
                questions[question_type] = builder()
            except ValueError as exc:
                errors[question_type] = str(exc)

        if not questions:
            raise ValueError(
                "Unable to generate any questions for the provided graph. "
                f"Encountered errors: {errors}"
            )

        answers = {
            question_type: question.answers_vector()
            for question_type, question in questions.items()
        }

        if include_details:
            return answers, questions
        return answers

    def try_generate(
        self,
        graph: GeneratedGraph,
        vertex_labels: Dict[str, str],
        *,
        path_length: Optional[int] = None,
        include_details: bool = False,
        entity_label: str = "vertex",
    ) -> Optional[
        Dict[QuestionType, List[bool]]
        | Tuple[Dict[QuestionType, List[bool]], Dict[QuestionType, Question]]
    ]:
        """Best-effort variant that returns None when no question can be built."""
        try:
            return self.generate(
                graph,
                vertex_labels,
                path_length=path_length,
                include_details=include_details,
                entity_label=entity_label,
            )
        except ValueError:
            return None

    def _generate_q1(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
        vertex_labels: Dict[str, str],
        positions: Dict[str, Tuple[int, int]],
        entity_label: str,
    ) -> Question1:
        if len(vertex_ids) < self.NUM_OPTIONS:
            raise ValueError("Graph must contain at least four vertices to build Q1.")

        positive_pool = self._collect_q1_positive(vertex_ids, adjacency)
        none_pool = self._collect_q1_none(vertex_ids, adjacency)

        # If force_all_valid is True, only generate positive questions (all options are valid)
        if self._force_all_valid:
            if positive_pool:
                return self._build_q1_positive(
                    positive_pool,
                    vertex_labels,
                    positions,
                    entity_label,
                )
            raise ValueError("Unable to craft a valid Q1 with all valid options for the supplied graph.")
        
        # Normal behavior: mix of positive and none questions
        if positive_pool and (not none_pool or self._rng.random() > 0.35):
            return self._build_q1_positive(
                positive_pool,
                vertex_labels,
                positions,
                entity_label,
            )
        if none_pool:
            return self._build_q1_none(
                none_pool, vertex_labels, positions, entity_label
            )
        raise ValueError("Unable to craft a valid Q1 for the supplied graph.")

    def _generate_q2(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
        distance_cache: Dict[str, Dict[str, int]],
        vertex_labels: Dict[str, str],
        entity_label: str,
    ) -> Question2:
        positive_pool = self._collect_q2_positive(vertex_ids, distance_cache)
        none_pool = self._collect_q2_none(vertex_ids, distance_cache)

        if positive_pool and (not none_pool or self._rng.random() > 0.35):
            return self._build_q2_positive(positive_pool, vertex_labels, entity_label)
        if none_pool:
            return self._build_q2_none(none_pool, vertex_labels, entity_label)
        raise ValueError("Unable to craft a valid Q2 for the supplied graph.")

    def _generate_q3(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
        distance_cache: Dict[str, Dict[str, int]],
        vertex_labels: Dict[str, str],
        path_length: Optional[int],
        entity_label: str,
    ) -> Question3:
        if path_length is None or path_length <= 0:
            raise ValueError("Q3 requires a positive 'path_length' parameter.")

        positive_pool = self._collect_q3_positive(
            vertex_ids, distance_cache, path_length
        )
        none_pool = self._collect_q3_none(vertex_ids, distance_cache, path_length)

        if positive_pool and (not none_pool or self._rng.random() > 0.35):
            return self._build_q3_positive(
                positive_pool, vertex_labels, path_length, entity_label
            )
        if none_pool:
            return self._build_q3_none(
                none_pool, vertex_labels, path_length, entity_label
            )
        raise ValueError("Unable to craft a valid Q3 for the supplied graph.")

    # --- Q1 helpers -----------------------------------------------------

    def _collect_q1_positive(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
    ) -> List[Tuple[str, List[str], List[str]]]:
        candidates: List[Tuple[str, List[str], List[str]]] = []
        # When force_all_valid is True, need NUM_OPTIONS - 1 distractors (3)
        # Otherwise, need NUM_OPTIONS - 2 distractors (2)
        min_distractors = (self.NUM_OPTIONS - 1) if self._force_all_valid else (self._num_actual_options() - 1)
        for vertex_id in vertex_ids:
            neighbors = list(dict.fromkeys(adjacency.get(vertex_id, [])))
            if not neighbors:
                continue
            distractor_pool = self._exclusive_pool(vertex_id, neighbors, vertex_ids)
            if len(distractor_pool) < min_distractors:
                continue
            candidates.append((vertex_id, neighbors, distractor_pool))
        return candidates
        return candidates

    def _collect_q1_none(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
    ) -> List[Tuple[str, List[str]]]:
        candidates: List[Tuple[str, List[str]]] = []
        for vertex_id in vertex_ids:
            distractor_pool = self._exclusive_pool(
                vertex_id, adjacency.get(vertex_id, []), vertex_ids
            )
            if len(distractor_pool) < self._num_actual_options():
                continue
            candidates.append((vertex_id, distractor_pool))
        return candidates

    def _build_q1_positive(
        self,
        candidates: List[Tuple[str, List[str], List[str]]],
        vertex_labels: Dict[str, str],
        positions: Dict[str, Tuple[int, int]],
        entity_label: str,
    ) -> Question1:
        source_id, neighbors, distractors = self._rng.choice(candidates)
        correct_vertex = self._rng.choice(neighbors)
        option_ids = self._assemble_option_ids(distractors, correct_vertex)
        options = self._build_options(option_ids, vertex_labels)
        correct_index = option_ids.index(correct_vertex)
        prompt = (
            f"Which {entity_label} is reachable via exactly one directed edge starting from "
            f"{self._label_for(source_id, vertex_labels)}? Treat each undirected edge as two directed edges, one in each direction.{self._none_suffix()}"
        )
        steps = self._manhattan_distance(source_id, correct_vertex, positions)
        euclid = self._euclidean_distance(source_id, correct_vertex, positions)
        return Question1(
            question_type=QuestionType.Q1,
            prompt=prompt,
            options=options,
            correct_index=correct_index,
            source_vertex=source_id,
            variant="unique_neighbor",
            target_length_steps=steps,
            target_length_euclidean=euclid,
        )

    def _build_q1_none(
        self,
        candidates: List[Tuple[str, List[str]]],
        vertex_labels: Dict[str, str],
        positions: Dict[str, Tuple[int, int]],
        entity_label: str,
    ) -> Question1:
        source_id, distractors = self._rng.choice(candidates)
        annotated = [
            (
                vertex_id,
                self._manhattan_distance(source_id, vertex_id, positions),
                self._euclidean_distance(source_id, vertex_id, positions),
            )
            for vertex_id in distractors
        ]
        anchor = self._rng.choice(annotated)
        target_steps = anchor[1]
        tolerance = 0
        option_ids: List[str] = []
        max_steps = max((item[1] for item in annotated), default=0)
        while not option_ids and tolerance <= max_steps:
            pool = [
                vertex_id
                for vertex_id, steps, _ in annotated
                if abs(steps - target_steps) <= tolerance
            ]
            if len(pool) >= self._num_actual_options():
                option_ids = self._rng.sample(pool, self._num_actual_options())
                break
            tolerance += 1
        if not option_ids:
            option_ids = self._rng.sample(distractors, self._num_actual_options())
        options = self._build_options(option_ids, vertex_labels)
        prompt = (
            f"Which {entity_label} is reachable via exactly one directed edge starting from "
            f"{self._label_for(source_id, vertex_labels)}? Treat each undirected edge as two directed edges, one in each direction.{self._none_suffix()}"
        )
        return Question1(
            question_type=QuestionType.Q1,
            prompt=prompt,
            options=options,
            correct_index=len(options) - 1,
            source_vertex=source_id,
            variant="none_of_the_above",
            target_length_steps=target_steps,
            target_length_euclidean=anchor[2],
        )

    # --- Q2 helpers -----------------------------------------------------

    def _collect_q2_positive(
        self,
        vertex_ids: Sequence[str],
        distance_cache: Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, List[str], List[str]]]:
        candidates: List[Tuple[str, List[str], List[str]]] = []
        for vertex_id in vertex_ids:
            reachable = list(distance_cache.get(vertex_id, {}).keys())
            if not reachable:
                continue
            distractor_pool = self._exclusive_pool(vertex_id, reachable, vertex_ids)
            if len(distractor_pool) < self._num_actual_options() - 1:
                continue
            candidates.append((vertex_id, reachable, distractor_pool))
        return candidates

    def _collect_q2_none(
        self,
        vertex_ids: Sequence[str],
        distance_cache: Dict[str, Dict[str, int]],
    ) -> List[Tuple[str, List[str]]]:
        candidates: List[Tuple[str, List[str]]] = []
        for vertex_id in vertex_ids:
            reachable = list(distance_cache.get(vertex_id, {}).keys())
            if reachable:
                continue
            distractor_pool = self._exclusive_pool(vertex_id, [], vertex_ids)
            if len(distractor_pool) < self._num_actual_options():
                continue
            candidates.append((vertex_id, distractor_pool))
        return candidates

    def _build_q2_positive(
        self,
        candidates: List[Tuple[str, List[str], List[str]]],
        vertex_labels: Dict[str, str],
        entity_label: str,
    ) -> Question2:
        source_id, reachable, distractors = self._rng.choice(candidates)
        correct_vertex = self._rng.choice(reachable)
        option_ids = self._assemble_option_ids(distractors, correct_vertex)
        options = self._build_options(option_ids, vertex_labels)
        correct_index = option_ids.index(correct_vertex)
        prompt = (
            f"Which {entity_label} can be reached by a sequence of directed edges starting from "
            f"{self._label_for(source_id, vertex_labels)}? Treat undirected edges as two directed edges."
            f"{self._none_suffix()}"
        )
        return Question2(
            question_type=QuestionType.Q2,
            prompt=prompt,
            options=options,
            correct_index=correct_index,
            source_vertex=source_id,
            variant="reachable",
        )

    def _build_q2_none(
        self,
        candidates: List[Tuple[str, List[str]]],
        vertex_labels: Dict[str, str],
        entity_label: str,
    ) -> Question2:
        source_id, distractors = self._rng.choice(candidates)
        option_ids = self._rng.sample(distractors, self._num_actual_options())
        options = self._build_options(option_ids, vertex_labels)
        prompt = (
            f"Which {entity_label} can be reached by a sequence of directed edges starting from "
            f"{self._label_for(source_id, vertex_labels)}? Treat undirected edges as two directed edges."
            f"{self._none_suffix()}"
        )
        return Question2(
            question_type=QuestionType.Q2,
            prompt=prompt,
            options=options,
            correct_index=len(options) - 1,
            source_vertex=source_id,
            variant="none_of_the_above",
        )

    # --- Q3 helpers -----------------------------------------------------

    def _collect_q3_positive(
        self,
        vertex_ids: Sequence[str],
        distance_cache: Dict[str, Dict[str, int]],
        steps: int,
    ) -> List[Tuple[str, List[str], List[str]]]:
        candidates: List[Tuple[str, List[str], List[str]]] = []
        for vertex_id in vertex_ids:
            distances = distance_cache.get(vertex_id, {})
            exact_targets = [
                target for target, dist in distances.items() if dist == steps
            ]
            if not exact_targets:
                continue
            excluded = [target for target, dist in distances.items() if dist == steps]
            distractor_pool = self._exclusive_pool(vertex_id, excluded, vertex_ids)
            if len(distractor_pool) < self._num_actual_options() - 1:
                continue
            candidates.append((vertex_id, exact_targets, distractor_pool))
        return candidates

    def _collect_q3_none(
        self,
        vertex_ids: Sequence[str],
        distance_cache: Dict[str, Dict[str, int]],
        steps: int,
    ) -> List[Tuple[str, List[str]]]:
        candidates: List[Tuple[str, List[str]]] = []
        for vertex_id in vertex_ids:
            distances = distance_cache.get(vertex_id, {})
            exact_targets = [
                target for target, dist in distances.items() if dist == steps
            ]
            if exact_targets:
                continue
            distractor_pool = self._exclusive_pool(vertex_id, [], vertex_ids)
            if len(distractor_pool) < self._num_actual_options():
                continue
            candidates.append((vertex_id, distractor_pool))
        return candidates

    def _build_q3_positive(
        self,
        candidates: List[Tuple[str, List[str], List[str]]],
        vertex_labels: Dict[str, str],
        steps: int,
        entity_label: str,
    ) -> Question3:
        source_id, exact_targets, distractors = self._rng.choice(candidates)
        correct_vertex = self._rng.choice(exact_targets)
        option_ids = self._assemble_option_ids(distractors, correct_vertex)
        options = self._build_options(option_ids, vertex_labels)
        correct_index = option_ids.index(correct_vertex)
        prompt = (
            f"Which {entity_label} is exactly {steps} steps (directed edges) away from "
            f"{self._label_for(source_id, vertex_labels)}? Treat undirected edges as two directed edges."
            f"{self._none_suffix()}"
        )
        return Question3(
            question_type=QuestionType.Q3,
            prompt=prompt,
            options=options,
            correct_index=correct_index,
            source_vertex=source_id,
            steps=steps,
            variant="distance_match",
        )

    def _build_q3_none(
        self,
        candidates: List[Tuple[str, List[str]]],
        vertex_labels: Dict[str, str],
        steps: int,
        entity_label: str,
    ) -> Question3:
        source_id, distractors = self._rng.choice(candidates)
        option_ids = self._rng.sample(distractors, self._num_actual_options())
        options = self._build_options(option_ids, vertex_labels)
        prompt = (
            f"Which {entity_label} is exactly {steps} steps (directed edges) away from "
            f"{self._label_for(source_id, vertex_labels)}? Treat undirected edges as two directed edges."
            f"{self._none_suffix()}"
        )
        return Question3(
            question_type=QuestionType.Q3,
            prompt=prompt,
            options=options,
            correct_index=len(options) - 1,
            source_vertex=source_id,
            steps=steps,
            variant="none_of_the_above",
        )

    # --- Shared utilities -----------------------------------------------

    def _collect_vertex_ids(self, graph: GeneratedGraph) -> List[str]:
        raw_vertices = getattr(graph, "vertices", None)
        if raw_vertices is None:
            raw_vertices = graph["vertices"]
        vertex_ids: List[str] = []
        for vertex in raw_vertices:
            vertex_id = getattr(vertex, "id", None) or vertex.get("id")
            if vertex_id is None:
                raise ValueError("Every vertex needs an 'id' field.")
            vertex_ids.append(vertex_id)
        return vertex_ids

    def _collect_positions(self, graph: GeneratedGraph) -> Dict[str, Tuple[int, int]]:
        raw_vertices = getattr(graph, "vertices", None)
        if raw_vertices is None:
            raw_vertices = graph["vertices"]
        positions: Dict[str, Tuple[int, int]] = {}
        for vertex in raw_vertices:
            vertex_id = getattr(vertex, "id", None) or vertex.get("id")
            row = getattr(vertex, "row", None)
            col = getattr(vertex, "col", None)
            if row is None or col is None:
                # If the vertex is a dict, fall back to key lookup.
                row = vertex.get("row")
                col = vertex.get("col")
            if vertex_id is None or row is None or col is None:
                raise ValueError("Each vertex must define 'id', 'row', and 'col'.")
            positions[vertex_id] = (int(row), int(col))
        return positions

    def _collect_adjacency(
        self,
        graph: GeneratedGraph,
        vertex_ids: Sequence[str],
    ) -> Dict[str, List[str]]:
        raw_adjacency: Optional[Dict[str, List[str]]] = getattr(
            graph, "adjacency", None
        )
        if raw_adjacency is None:
            raw_adjacency = graph["adjacency"]
        return {
            vertex_id: list(raw_adjacency.get(vertex_id, []))
            for vertex_id in vertex_ids
        }

    def _build_distance_cache(
        self,
        vertex_ids: Sequence[str],
        adjacency: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, int]]:
        return {
            vertex_id: self._bfs_distances(vertex_id, adjacency)
            for vertex_id in vertex_ids
        }

    def _bfs_distances(
        self,
        source_id: str,
        adjacency: Dict[str, List[str]],
    ) -> Dict[str, int]:
        visited = {source_id}
        queue = deque([(source_id, 0)])
        distances: Dict[str, int] = {}
        while queue:
            current, dist = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                distances[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
        return distances

    def _manhattan_distance(
        self,
        source_id: str,
        target_id: str,
        positions: Dict[str, Tuple[int, int]],
    ) -> int:
        sx, sy = positions[source_id]
        tx, ty = positions[target_id]
        return abs(sx - tx) + abs(sy - ty)

    def _euclidean_distance(
        self,
        source_id: str,
        target_id: str,
        positions: Dict[str, Tuple[int, int]],
    ) -> float:
        sx, sy = positions[source_id]
        tx, ty = positions[target_id]
        return math.hypot(sx - tx, sy - ty)

    def _exclusive_pool(
        self,
        source_id: str,
        excluded_vertices: Sequence[str],
        vertex_ids: Sequence[str],
    ) -> List[str]:
        excluded = set(excluded_vertices)
        pool = [
            vertex_id
            for vertex_id in vertex_ids
            if vertex_id != source_id and vertex_id not in excluded
        ]
        if len(pool) < self._num_actual_options() and source_id not in excluded:
            pool.append(source_id)
        return list(dict.fromkeys(pool))

    def _assemble_option_ids(
        self,
        distractors: Sequence[str],
        correct_vertex: str,
    ) -> List[str]:
        # When force_all_valid is True, we need NUM_OPTIONS vertex options (no "None of the above")
        # Otherwise, we need NUM_OPTIONS - 1 vertex options (last one is "None of the above")
        num_distractors = (self.NUM_OPTIONS - 1) if self._force_all_valid else (self._num_actual_options() - 1)
        distractor_ids = self._rng.sample(distractors, num_distractors)
        option_ids = distractor_ids + [correct_vertex]
        self._rng.shuffle(option_ids)
        return option_ids

    def _build_options(
        self,
        option_ids: Sequence[str],
        vertex_labels: Dict[str, str],
    ) -> List[str]:
        options = [self._label_for(v_id, vertex_labels) for v_id in option_ids]
        if not self._force_all_valid:
            options.append(self.NONE_OPTION)
        return options

    def _label_for(self, vertex_id: str, vertex_labels: Dict[str, str]) -> str:
        return vertex_labels.get(vertex_id, vertex_id)

    def _num_actual_options(self) -> int:
        return self.NUM_OPTIONS - 1

    def _none_suffix(self) -> str:
        # return " (Select 'None' if no listed option applies.)"
        # None suffix not needed for the new None option format.
        return ""

    def _normalize_entity_label(self, label: Optional[str]) -> str:
        if not label:
            return "vertex"
        cleaned = label.strip()
        return cleaned if cleaned else "vertex"
