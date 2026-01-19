from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

try:
    from .graph_generator import GraphGenerator
    from .graph_to_image import GraphToImageRenderer
    from .graph_to_text import GraphToTextConverter
    from .question_generator import Question, QuestionGenerator, QuestionType
except ImportError:  # Fallback for direct execution
    from graph_generator import GraphGenerator  # type: ignore
    from graph_to_image import GraphToImageRenderer  # type: ignore
    from graph_to_text import GraphToTextConverter  # type: ignore
    from question_generator import Question, QuestionGenerator, QuestionType  # type: ignore

GridSize = Union[int, Tuple[int, int]]


@dataclass
class DatasetConfig:
    """Configuration block describing the dataset to be generated."""

    grid_size: GridSize
    vertex_set_name: str  # e.g. "number", "letter", "shape"
    vertex_labels: Dict[str, str]
    num_vertices_range: Tuple[int, int]
    path_length_range: Tuple[int, int]
    directed: bool
    graph_density: Literal["auto", "sparse", "dense"] = "auto"
    draw_grid: bool = True
    num_graphs: Optional[int] = None
    dataset_name: Optional[str] = None
    graph_seed: Optional[int] = None
    question_seed: Optional[int] = None
    question_type_targets: Optional[Dict[QuestionType, int]] = None
    max_none_answer_ratio: float = 0.25
    image_cell_pixels: int = 128
    image_margin_cells: float = 0.0
    image_supersample: int = 2
    shape_fill_solid: bool = True
    shape_scale: float = 0.85
    shape_stroke_width: float = 2.0
    arrow_stroke_width: float = 2.0


class DatasetGenerator:
    """High-level orchestrator that assembles datasets under ./collections/<name>."""

    def __init__(self, collections_dir: Union[str, Path] = "collections") -> None:
        self._collections_dir = Path(collections_dir)
        self._collections_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, config: DatasetConfig) -> Path:
        self._validate_config(config)
        dataset_name = config.dataset_name or self._default_dataset_name(config)
        dataset_dir = self._collections_dir / dataset_name
        if dataset_dir.exists():
            raise FileExistsError(
                f"Dataset directory '{dataset_dir}' already exists. "
                "Choose a different dataset_name or remove the existing folder."
            )

        imgs_dir = dataset_dir / "imgs"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir.mkdir(parents=True, exist_ok=True)
        data_path = dataset_dir / "data.jsonl"

        rows, cols = self._normalize_grid_size(config.grid_size)
        rng = random.Random(config.graph_seed)
        graph_generator = GraphGenerator(seed=config.graph_seed)
        question_generator = QuestionGenerator(seed=config.question_seed)
        text_converter = GraphToTextConverter()
        image_renderer = GraphToImageRenderer(
            output_dir=imgs_dir,
            cell_pixels=config.image_cell_pixels,
            margin_cells=config.image_margin_cells,
            supersample=config.image_supersample,
            shape_fill_solid=config.shape_fill_solid,
            shape_scale=config.shape_scale,
            shape_stroke_width=config.shape_stroke_width,
            arrow_stroke_width=config.arrow_stroke_width,
            draw_grid=config.draw_grid,
        )

        question_targets = (
            dict(config.question_type_targets) if config.question_type_targets else None
        )
        remaining_questions = (
            dict(question_targets) if question_targets is not None else None
        )
        if question_targets:
            none_targets = self._build_none_targets(
                question_targets, config.max_none_answer_ratio
            )
            remaining_none = dict(none_targets)
            remaining_standard = {
                qt: question_targets[qt] - none_targets.get(qt, 0)
                for qt in question_targets
            }
        else:
            none_targets = {}
            remaining_none = {}
            remaining_standard = {}

        attempts = 0
        graphs_written = 0
        graph_target = config.num_graphs or 0
        question_counter = 0
        attempt_limit = self._attempt_budget(config, question_targets)

        with data_path.open("w", encoding="utf-8") as data_file:
            while True:
                if question_targets:
                    assert remaining_questions is not None
                    if all(count <= 0 for count in remaining_questions.values()):
                        break
                else:
                    if graphs_written >= graph_target:
                        break

                attempts += 1
                if attempts > attempt_limit:
                    raise RuntimeError(
                        "Exceeded the maximum number of graph generation attempts "
                        f"({attempt_limit}). Consider relaxing the config (e.g., adjust "
                        "question targets, density, or vertex ranges)."
                    )
                num_vertices = rng.randint(*config.num_vertices_range)
                path_length = rng.randint(*config.path_length_range)

                try:
                    graph = graph_generator.generate_graph(
                        num_vertices=num_vertices,
                        grid_size=(rows, cols),
                        directed=config.directed,
                        density=config.graph_density,
                    )
                except ValueError:
                    continue

                question_payload = question_generator.try_generate(
                    graph,
                    config.vertex_labels,
                    path_length=path_length,
                    include_details=True,
                    entity_label=config.vertex_set_name,
                )

                if question_payload is None:
                    continue

                _answer_map, question_map = question_payload

                if question_targets:
                    usable_questions = self._select_questions_for_targets(
                        question_map,
                        remaining_questions,
                        remaining_standard,
                        remaining_none,
                    )
                    if not usable_questions:
                        continue
                else:
                    usable_questions = question_map

                text_result = text_converter.convert(graph, config.vertex_labels)
                image_identifier = f"graph_{graphs_written:05d}"
                image_result = image_renderer.render(
                    graph,
                    config.vertex_labels,
                    identifier=image_identifier,
                    label_set_hint=config.vertex_set_name,
                )
                relative_image_path = image_result.image_path.relative_to(dataset_dir)

                per_graph_rows = self._rows_from_questions(
                    usable_questions,
                    relative_image_path=str(relative_image_path),
                    textual_description=text_result.text,
                )

                if not per_graph_rows:
                    continue

                for row in per_graph_rows:
                    row["question_id"] = f"q{question_counter:06d}"
                    question_counter += 1
                    data_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                if question_targets:
                    for question_type, question in usable_questions.items():
                        remaining_questions[question_type] -= 1
                        if getattr(question, "variant", "") == "none_of_the_above":
                            remaining_none[question_type] -= 1
                        else:
                            remaining_standard[question_type] -= 1

                graphs_written += 1

        if question_targets:
            target_summary = ", ".join(
                f"{question_type.name}={count}"
                for question_type, count in question_targets.items()
            )
            print(
                f"Generated dataset '{dataset_name}' with {graphs_written} graphs "
                f"covering question targets ({target_summary}) after {attempts} attempts."
            )
        else:
            print(
                f"Generated dataset '{dataset_name}' with {graph_target} graphs "
                f"after {attempts} attempts."
            )
        return dataset_dir

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _rows_from_questions(
        self,
        question_map: Dict[QuestionType, Question],
        *,
        relative_image_path: str,
        textual_description: str,
    ) -> list:
        rows = []
        for question_type, question in question_map.items():
            options = [str(option) for option in question.options]
            answer = options[question.correct_index]
            rows.append(
                {
                    "question": question.prompt,
                    "options": options,
                    "answer": answer,
                    "answer_idx": question.correct_index,
                    "question_type": question_type.name,
                    "image": relative_image_path,
                    "textual_description": textual_description,
                    "edge_length_steps": getattr(question, "target_length_steps", None),
                    "edge_length": getattr(question, "target_length_euclidean", None),
                }
            )
        return rows

    def _select_questions_for_targets(
        self,
        question_map: Dict[QuestionType, Question],
        remaining: Dict[QuestionType, int],
        remaining_standard: Dict[QuestionType, int],
        remaining_none: Dict[QuestionType, int],
    ) -> Dict[QuestionType, Question]:
        filtered: Dict[QuestionType, Question] = {}
        for question_type, question in question_map.items():
            if remaining.get(question_type, 0) <= 0:
                continue
            variant = getattr(question, "variant", "")
            if variant == "none_of_the_above":
                if remaining_none.get(question_type, 0) <= 0:
                    continue
            else:
                if remaining_standard.get(question_type, 0) <= 0:
                    continue
            filtered[question_type] = question
        return filtered

    def _build_none_targets(
        self,
        question_targets: Dict[QuestionType, int],
        ratio: float,
    ) -> Dict[QuestionType, int]:
        if ratio <= 0:
            return {q_type: 0 for q_type in question_targets}
        if ratio >= 1:
            return dict(question_targets)
        none_counts: Dict[QuestionType, int] = {}
        for question_type, target in question_targets.items():
            desired = int(round(target * ratio))
            desired = min(max(desired, 0), target)
            none_counts[question_type] = desired
        return none_counts

    def _normalize_question_targets(
        self,
        question_targets: Dict[QuestionType, int],
    ) -> Dict[QuestionType, int]:
        normalized: Dict[QuestionType, int] = {}
        for raw_type, raw_count in question_targets.items():
            if isinstance(raw_type, QuestionType):
                question_type = raw_type
            elif isinstance(raw_type, str):
                try:
                    question_type = QuestionType[raw_type.upper()]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported question type key '{raw_type}'. "
                        "Use QuestionType enum members or their string names."
                    ) from exc
            else:
                raise TypeError(
                    "question_type_targets keys must be QuestionType or str instances."
                )

            count = int(raw_count)
            normalized[question_type] = count
        return dict(sorted(normalized.items(), key=lambda item: item[0].name))

    def _default_dataset_name(self, config: DatasetConfig) -> str:
        rows, cols = self._normalize_grid_size(config.grid_size)
        v_range = f"{config.num_vertices_range[0]}-{config.num_vertices_range[1]}"
        path_range = f"{config.path_length_range[0]}-{config.path_length_range[1]}"
        density_tag = config.graph_density
        directed_tag = "dir" if config.directed else "undir"
        vertex_tag = config.vertex_set_name.lower().replace(" ", "_")
        suffix_parts = [directed_tag]
        if config.num_graphs:
            suffix_parts.append(f"{config.num_graphs}g")
        if config.question_type_targets:
            counts_tag = "-".join(
                f"{question_type.name.lower()}{count}"
                for question_type, count in config.question_type_targets.items()
            )
            suffix_parts.append(f"qs{counts_tag}")
        suffix = "_".join(suffix_parts)
        return (
            f"{vertex_tag}_{rows}x{cols}_"
            f"v{v_range}_d{density_tag}_x{path_range}_{suffix}"
        )

    def _normalize_grid_size(self, grid_size: GridSize) -> Tuple[int, int]:
        if isinstance(grid_size, int):
            if grid_size <= 0:
                raise ValueError("grid_size must be positive.")
            return grid_size, grid_size
        if len(grid_size) != 2:
            raise ValueError("grid_size tuple must contain exactly two integers.")
        rows, cols = grid_size
        if rows <= 0 or cols <= 0:
            raise ValueError("grid_size dimensions must be positive.")
        return rows, cols

    def _validate_config(self, config: DatasetConfig) -> None:
        if config.question_type_targets:
            config.question_type_targets = self._normalize_question_targets(
                config.question_type_targets
            )
            if any(count <= 0 for count in config.question_type_targets.values()):
                raise ValueError("question_type_targets must map to positive integers.")
        if config.num_graphs is not None and config.num_graphs <= 0:
            raise ValueError("num_graphs must be a positive integer when provided.")
        if not config.question_type_targets and config.num_graphs is None:
            raise ValueError(
                "Either 'num_graphs' or 'question_type_targets' must be specified."
            )
        if not 0 <= config.max_none_answer_ratio <= 1:
            raise ValueError("max_none_answer_ratio must be within [0, 1].")
        if config.image_cell_pixels <= 0:
            raise ValueError("image_cell_pixels must be positive.")
        if config.image_supersample < 1:
            raise ValueError("image_supersample must be at least 1.")
        if config.image_margin_cells < 0:
            raise ValueError("image_margin_cells cannot be negative.")
        if config.shape_scale <= 0:
            raise ValueError("shape_scale must be positive.")
        if config.shape_stroke_width <= 0:
            raise ValueError("shape_stroke_width must be positive.")
        if config.arrow_stroke_width <= 0:
            raise ValueError("arrow_stroke_width must be positive.")

        rows, cols = self._normalize_grid_size(config.grid_size)
        expected_vertices = rows * cols
        if len(config.vertex_labels) < expected_vertices:
            raise ValueError(
                f"vertex_labels must cover at least {expected_vertices} entries "
                f"for a {rows}x{cols} grid."
            )
        for key, value in config.vertex_labels.items():
            if not isinstance(value, str):
                config.vertex_labels[key] = str(value)
        if config.num_vertices_range[0] > config.num_vertices_range[1]:
            raise ValueError("num_vertices_range must be in ascending order.")
        if config.path_length_range[0] > config.path_length_range[1]:
            raise ValueError("path_length_range must be in ascending order.")
        if config.graph_density not in ("auto", "sparse", "dense"):
            raise ValueError("graph_density must be one of: auto, sparse, dense.")
        if not isinstance(config.draw_grid, bool):
            raise ValueError("draw_grid must be a boolean flag.")

    def _attempt_budget(
        self,
        config: DatasetConfig,
        question_targets: Optional[Dict[QuestionType, int]],
    ) -> int:
        if question_targets:
            base = sum(question_targets.values())
        elif config.num_graphs:
            base = config.num_graphs
        else:
            base = 100
        return max(1000, base * 40)
