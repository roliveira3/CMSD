from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from .graph_generator import GeneratedGraph
except ImportError:  # Fallback for direct execution
    from graph_generator import GeneratedGraph  # type: ignore


@dataclass
class GraphToTextResult:
    """Container for any metadata produced alongside the textual description."""

    text: str
    debug_info: Optional[dict] = None


class GraphToTextConverter:
    """Placeholder converter that turns an abstract graph into a textual description."""
    def convert(self, graph: GeneratedGraph, vertex_labels: Optional[dict], conversion_type: str = "adjacency_list", use_cells: bool = False) -> GraphToTextResult:
        """Top-level conversion entry.

        conversion_type: one of 'adjacency_matrix', 'adjacency_list', 'edge_list'.
        use_cells: when True, the output refers to cell coordinates (r,c) and includes a cells mapping first.
                   when False, outputs use vertex labels (if provided) or vertex ids.
        vertex_labels: optional mapping from vertex id -> display label (string).
        """
        summary = self._summarize_graph(graph, conversion_type)

        top_intro = "Below we describe a diagram which consists of entities which can be connected to each other."

        if conversion_type == "adjacency_matrix":
            body = self._convert_matrix(graph, use_cells=use_cells, vertex_labels=vertex_labels)
        elif conversion_type == "adjacency_list":
            body = self._convert_list(graph, use_cells=use_cells, vertex_labels=vertex_labels)
        elif conversion_type == "edge_list":
            body = self._convert_edge_list(graph, use_cells=use_cells, vertex_labels=vertex_labels)
        else:
            raise ValueError(f"Unsupported conversion type: {conversion_type}")

        return GraphToTextResult(text=f"{top_intro}\n" + body, debug_info=summary)

    def _id_label(self, vid: str, vertex_labels: Optional[dict]) -> str:
        return vertex_labels[vid] if vertex_labels and vid in vertex_labels else vid

    def _cell_maps(self, graph: GeneratedGraph):
        vertices = graph.vertices
        id_to_cell = {v.id: (v.row, v.col) for v in vertices}
        cell_to_id = {(v.row, v.col): v.id for v in vertices}
        return id_to_cell, cell_to_id

    def get_cell_to_vertex_text(self, graph: GeneratedGraph, vertex_labels: Optional[dict] = None) -> str:
        """Return a compact single-line cell->label mapping. Sub-intro included.
        example output:
        cells:[
          (0,0):A,
          (0,1):B, 
          (0,2):None, 
          (0,3):D, 
          (1,0):E, ...
          ]
        """
        rows, cols = graph.grid_size
        id_to_cell, cell_to_id = self._cell_maps(graph)

        items = []
        for r in range(rows):
            for c in range(cols):
                vid = cell_to_id.get((r, c))
                if vid is None:
                    label = "No entity"
                else:
                    label = self._id_label(vid, vertex_labels)
                items.append(f"({r},{c}): {label}")
        return "cells:[\n" + ",\n".join(items) + "\n]"

    
    def _convert_matrix(self, graph: GeneratedGraph, use_cells: bool, vertex_labels: Optional[dict]) -> str:
        """
        Convert the graph to an adjacency matrix format.
        example output if use_cells is True:
        <cells mapping(via self.get_cell_to_vertex_text)>
        \t(0,0)\t(0,1)\t(1,0)\t(1,1)
        (0,0)\t0\t1\t0\t0
        (0,1)\t0\t0\t1\t0
        (1,0)\t0\t0\t0\t1
        (1,1)\t0\t0\t0\t0

        example output if use_cells is False:
        \tA\tB\tC\tD
        A\t0\t1\t0\t0
        B\t0\t0\t1\t0
        C\t0\t0\t0\t1
        D\t0\t0\t0\t0
        """
        vertices = graph.vertices
        edges = graph.edges
        directed = graph.directed

        if use_cells:
            verts_sorted = sorted(vertices, key=lambda v: (v.row, v.col))
            header_keys = [f"({v.row},{v.col})" for v in verts_sorted]
        else:
            verts_sorted = list(vertices)
            header_keys = [self._id_label(v.id, vertex_labels) for v in verts_sorted]

        ids = [v.id for v in verts_sorted]
        index = {vid: i for i, vid in enumerate(ids)}
        size = len(ids)
        matrix = [[0] * size for _ in range(size)]
        for e in edges:
            if e.source in index and e.target in index:
                matrix[index[e.source]][index[e.target]] = 1
                if not directed:
                    matrix[index[e.target]][index[e.source]] = 1
    
        intro = "The diagram and with the connections of the entities are laid out in 0/1 table: rows are sources, columns are destinations; 1 = direct link from source to destination, 0 = no connection between the two.\n"
        if use_cells:
            intro += f"The diagram consists of {graph.grid_size[0]} rows and {graph.grid_size[1]} columns. and we first describe what entity is in each cell and after the cell connections (if cells connect, the underlying entities connect) \n"
            intro += self.get_cell_to_vertex_text(graph, vertex_labels) + "\n"
        intro += f"The connections are: {'Oneway' if directed else 'Bothways'}.\n"
        intro += "Adjacency Matrix:\n"
        header = "\t" + "\t".join(header_keys)
        lines = [intro, header]
        id_to_cell, _ = self._cell_maps(graph)
        for vid, row in zip(ids, matrix):
            if use_cells:
                cell = id_to_cell[vid]
                row_label = f"({cell[0]},{cell[1]})"
            else:
                row_label = self._id_label(vid, vertex_labels)
            lines.append(f"{row_label}:\t" + "\t".join(str(c) for c in row))
        return "\n".join(lines)

    def _convert_list(self, graph: GeneratedGraph, use_cells: bool, vertex_labels: Optional[dict]) -> str:
        """

        example output if use_cells is True:
        <cells mapping(via self.get_cell_to_vertex_text)>
        Cell adjacency: each line is (r,c):[(r2,c2),...]
        
        example output if use_cells is False:
        List: each line shows entity:[targets]
        """
        vertices = graph.vertices
        adjacency = graph.adjacency
        
        if use_cells:

            id_to_cell, cell_to_id = self._cell_maps(graph)
            rows, cols = graph.grid_size

            intro = f"The diagram consists of {rows} rows and {cols} columns. and we first describe what entity is in each cell and after the cell connections (if cells connect, the underlying entities connect) \n"\
            "Each cell is followed by the list of cells it directly connects to.\n"\
            f"The connections are: {'Oneway' if graph.directed else 'Bothways'}.\n"
            intro += self.get_cell_to_vertex_text(graph, vertex_labels) + "\n"
            intro += "Connections:"

            lines = [intro]
            for r in range(rows):
                for c in range(cols):
                    vid = cell_to_id.get((r, c))
                    if vid is None:
                        targets_repr = "[]"
                    else:
                        targets = adjacency.get(vid, [])
                        target_cells = [id_to_cell[t] for t in targets]
                        targets_repr = "[" + ",".join(f"({rr},{cc})" for rr, cc in target_cells) + "]"
                    lines.append(f"({r},{c}):{targets_repr}")
            return "\n".join(lines)
        else:
            vertex_ids = [v.id for v in vertices]

            intro = "The diagram connections are listed below. each entity is followed by the list of entities it directly connects to.\n"\
            f"The connections are: {'Oneway' if graph.directed else 'Bothways'}.\n"
            intro += "Connections:\n"

            lines = [intro]
            for vid in vertex_ids:
                src_label = self._id_label(vid, vertex_labels)
                targets = adjacency.get(vid, [])
                target_labels = [self._id_label(t, vertex_labels) for t in targets]
                lines.append(f"{src_label}:[{','.join(target_labels)}]")
            return "\n".join(lines)

    def _convert_edge_list(self, graph: GeneratedGraph, use_cells: bool, vertex_labels: Optional[dict]) -> str:
        """
        example output if use_cells is True:
        <cells mapping(via self.get_cell_to_vertex_text)>
        connections:
        (0,0) connects to (0,1)
        (0,1) connects to (1,1)

        example output if use_cells is False:
        A connects to B
        B connects to C
        """
        edges = graph.edges
        intro = ""
        if use_cells:
            intro  += f"The diagram consists of {graph.grid_size[0]} rows and {graph.grid_size[1]} columns. and we first describe what entity is in each cell and after the cell connections (if cells connect, the underlying entities connect) \n"
            intro += self.get_cell_to_vertex_text(graph, vertex_labels) + "\n"
        intro += f"The connections are: {'Oneway' if graph.directed else 'Bothways'}.\n"
        intro += "Connections:\n"
        
        
        lines = [intro]
        if use_cells:
            
            id_to_cell, _ = self._cell_maps(graph)
            for e in edges:
                src = id_to_cell[e.source]
                tgt = id_to_cell[e.target]
                lines.append(f"({src[0]},{src[1]}) connects to ({tgt[0]},{tgt[1]})")
        else:
            for e in edges:
                src_label = self._id_label(e.source, vertex_labels)
                tgt_label = self._id_label(e.target, vertex_labels)
                lines.append(f"{src_label} connects to {tgt_label}")
        return "\n".join(lines)

  


    def _summarize_graph(self, graph: GeneratedGraph, conversion_type: str) -> dict:
        # Use direct attribute access; graph is expected to be a GeneratedGraph dataclass.
        vertices = graph.vertices
        edges = graph.edges
        grid_size = graph.grid_size

        return {
            "vertex_count": len(vertices),
            "edge_count": len(edges),
            "grid_rows": grid_size[0],
            "grid_cols": grid_size[1],
            "conversion_type": conversion_type,
        }
