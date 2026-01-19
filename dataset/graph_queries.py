from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class VertexPosition:
    id: str
    row: int
    col: int


class GraphQueries:
    """Utility class for computing derived metrics from serialized dataset rows."""

    def __init__(self, data_path: Path | str) -> None:
        self._records: Dict[str, dict] = {}
        path = Path(data_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                question_id = payload.get("question_id")
                if question_id is None:
                    continue
                self._records[question_id] = payload

    def _vertex_positions(self, graph_blob: dict) -> Dict[str, VertexPosition]:
        vertices = graph_blob.get("vertices", [])
        return {
            vertex["id"]: VertexPosition(
                id=vertex["id"], row=int(vertex["row"]), col=int(vertex["col"])
            )
            for vertex in vertices
        }

    def _distance(self, positions: Dict[str, VertexPosition], source: str, target: str) -> Tuple[int, float]:
        src = positions.get(source)
        tgt = positions.get(target)
        if src is None or tgt is None:
            raise KeyError("Unknown vertex id for distance computation.")
        manhattan = abs(src.row - tgt.row) + abs(src.col - tgt.col)
        euclidean = math.hypot(src.row - tgt.row, src.col - tgt.col)
        return manhattan, euclidean

    def edge_distances(self, question_id: str) -> Tuple[Optional[int], Optional[float]]:
        record = self._records.get(question_id)
        if record is None:
            raise KeyError(f"Unknown question_id '{question_id}'.")
        if record.get("edge_variant") != "unique_neighbor":
            return None, None
        graph_blob = record.get("graph")
        source_vertex = record.get("source_vertex")
        answer_vertex = record.get("answer_vertex")
        if not graph_blob or not source_vertex or not answer_vertex:
            return None, None
        positions = self._vertex_positions(graph_blob)
        manhattan, euclidean = self._distance(positions, source_vertex, answer_vertex)
        return manhattan, euclidean

    def records(self) -> Iterable[dict]:
        return self._records.values()

    def get_record(self, question_id: str) -> Optional[dict]:
        return self._records.get(question_id)
