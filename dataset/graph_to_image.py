from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

matplotlib.use("Agg")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - Pillow is an external dependency
    raise ImportError(
        "GraphToImageRenderer requires the 'Pillow' package. "
        "Install it via `pip install pillow`."
    ) from exc

try:
    from .graph_generator import Edge, GeneratedGraph, Vertex
    from .icon_catalog import IconDefinition, lookup_icon
except ImportError:  # Fallback for direct execution
    from graph_generator import Edge, GeneratedGraph, Vertex  # type: ignore
    from icon_catalog import IconDefinition, lookup_icon  # type: ignore


@dataclass
class GraphToImageResult:
    """Simple container that records where an illustrative asset is stored."""

    image_path: pathlib.Path


@dataclass
class IconRenderJob:
    icon: IconDefinition
    center: Tuple[float, float]
    node_radius: float


class GraphToImageRenderer:
    """Renders abstract graphs onto a grid while supporting text and icon vertices."""

    _PALETTE = [
        (69, 129, 255),
        (255, 121, 97),
        (101, 214, 194),
        (250, 200, 99),
        (188, 124, 255),
        (255, 173, 141),
        (89, 196, 143),
        (255, 107, 181),
        (120, 176, 255),
        (255, 198, 89),
    ]

    def __init__(
        self,
        output_dir: pathlib.Path | str = "artifacts/graphs",
        *,
        cell_pixels: int = 128,
        margin_cells: float = 0.0,
        supersample: int = 2,
        shape_fill_solid: bool = False,
        shape_scale: float = 0.85,
        shape_stroke_width: float = 2.0,
        arrow_stroke_width: float = 2.0,
        draw_grid: bool = True,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if cell_pixels <= 0:
            raise ValueError("cell_pixels must be a positive integer.")
        if margin_cells < 0:
            raise ValueError("margin_cells must be non-negative.")
        if supersample < 1:
            raise ValueError("supersample must be >= 1.")
        if shape_scale <= 0:
            raise ValueError("shape_scale must be positive.")
        if shape_stroke_width <= 0:
            raise ValueError("shape_stroke_width must be positive.")
        if arrow_stroke_width <= 0:
            raise ValueError("arrow_stroke_width must be positive.")

        self._cell_pixels = cell_pixels
        self._margin_cells = margin_cells
        self._supersample = supersample
        self._shape_fill_solid = shape_fill_solid
        self._shape_scale = shape_scale
        self._shape_stroke_width = shape_stroke_width
        self._arrow_stroke_width = arrow_stroke_width
        self._shape_renderers = self._build_shape_renderers()
        self._shape_aliases = self._build_shape_aliases()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

        self._background_color = (248, 249, 252, 255)
        self._grid_color = (212, 218, 233, 255)
        self._edge_color = (46, 54, 83, 255)
        self._edge_shadow = (255, 255, 255, 80)
        self._stroke_color = (32, 45, 72, 220)
        self._shape_outline_color = (0, 0, 0, 255)
        self._shape_fill_color = (0, 0, 0, 255)
        self._shape_hollow_fill = (0, 0, 0, 0)
        self._shape_outline_width = max(1, int(self._shape_stroke_width * self._supersample))
        self._arrow_clearance_ratio = 0.28
        self._node_fill_color = (255, 255, 255, 255)
        self._text_color = (0, 0, 0, 255)
        self._grid_visible = draw_grid
        self._grid_outer_width = 1
        self._grid_cell_width = 2

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render(
        self,
        graph: GeneratedGraph,
        vertex_labels: Dict[str, str],
        *,
        identifier: Optional[str] = None,
        label_set_hint: Optional[str] = None,
    ) -> GraphToImageResult:
        """Render the provided graph to disk and return its metadata."""
        graph_id = identifier or self._derive_identifier(graph)
        image_path = self._output_dir / f"{graph_id}.png"

        rows, cols = self._grid_size(graph)
        vertices = self._coerce_vertices(graph)
        edges = self._coerce_edges(graph)

        hi_res_cell = self._cell_pixels * self._supersample
        margin = self._margin_cells * hi_res_cell
        canvas_width = int(cols * hi_res_cell + margin * 2)
        canvas_height = int(rows * hi_res_cell + margin * 2)
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas, "RGBA")

        positions = self._place_vertices(vertices, margin, hi_res_cell)
        node_radius = hi_res_cell * 0.32
        classifications = self._classify_vertices(vertices, vertex_labels, label_set_hint)
        arrow_clearance = node_radius * self._arrow_clearance_ratio
        content_radii = self._compute_content_radii(vertices, classifications, node_radius, arrow_clearance)

        # Render everything with matplotlib in ONE pass for speed (3x faster)
        self._render_graph_matplotlib(
            canvas,
            edges,
            vertices,
            positions,
            node_radius,
            classifications,
            content_radii,
            arrow_clearance,
            directed=self._is_directed(graph),
        )

        if self._supersample > 1:
            downsampled = canvas.resize(
                (canvas_width // self._supersample, canvas_height // self._supersample),
                Image.LANCZOS,
            )
        else:
            downsampled = canvas

        final = Image.new("RGBA", downsampled.size, self._background_color)
        self._draw_crisp_grid(final, rows, cols)
        final.alpha_composite(downsampled)
        final.save(image_path)
        return GraphToImageResult(image_path=image_path)

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def _draw_crisp_grid(self, image: Image.Image, rows: int, cols: int) -> None:
        if not self._grid_visible:
            return

        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        cell = self._cell_pixels
        content_width = cols * cell
        content_height = rows * cell
        left_margin = max(0, (width - content_width) // 2)
        top_margin = max(0, (height - content_height) // 2)
        right_limit = left_margin + content_width
        bottom_limit = top_margin + content_height

        # Internal cell boundaries (uniform 2 px).
        for row in range(1, rows):
            y = top_margin + row * cell
            draw.line(
                [(left_margin, y), (right_limit, y)],
                fill=self._grid_color,
                width=self._grid_cell_width,
            )
        for col in range(1, cols):
            x = left_margin + col * cell
            draw.line(
                [(x, top_margin), (x, bottom_limit)],
                fill=self._grid_color,
                width=self._grid_cell_width,
            )

        # Outer border (1 px) around the grid footprint so every cell shares the same outline.
        if content_width > 0 and content_height > 0:
            draw.rectangle(
                [(left_margin, top_margin), (right_limit - 1, bottom_limit - 1)],
                outline=self._grid_color,
                width=self._grid_outer_width,
            )

    def _render_graph_matplotlib(
        self,
        canvas: Image.Image,
        edges: Sequence[Edge],
        vertices: Sequence[Vertex],
        positions: Dict[str, Tuple[float, float]],
        node_radius: float,
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]],
        content_radii: Dict[str, float],
        arrow_clearance: float,
        *,
        directed: bool,
    ) -> None:
        """Render edges, text labels, and icons in ONE matplotlib pass for speed.
        
        This is 3x faster than separate calls because it creates only one figure.
        """
        width, height = canvas.size
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y to match image coordinates
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis('off')
        
        # 1. Draw edges
        pair_counts: Dict[frozenset[str], int] = {}
        for edge in edges:
            pair = frozenset({edge.source, edge.target})
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        rendered_pairs: Set[frozenset[str]] = set()
        
        line_width_pixels = self._arrow_stroke_width * self._supersample
        line_width_points = line_width_pixels * 72 / dpi
        edge_color_mpl = self._rgba_to_mpl(self._edge_color)

        for edge in edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            start = positions[edge.source]
            end = positions[edge.target]
            if start == end:
                continue

            pair = frozenset({edge.source, edge.target})
            bidirectional = directed and pair_counts.get(pair, 0) > 1
            if bidirectional and pair in rendered_pairs:
                continue

            start_pad = self._content_padding(edge.source, content_radii, arrow_clearance)
            end_pad = self._content_padding(edge.target, content_radii, arrow_clearance)
            tapered = self._trim_edge(start, end, start_pad, end_pad)
            if tapered is None:
                continue
            trimmed_start, trimmed_end = tapered

            dx = trimmed_end[0] - trimmed_start[0]
            dy = trimmed_end[1] - trimmed_start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length == 0:
                continue
            ux, uy = dx / segment_length, dy / segment_length

            arrow_length, arrow_width = self._arrow_metrics(node_radius, segment_length)

            line_start = trimmed_start
            line_end = trimmed_end
            if bidirectional:
                line_start = (
                    line_start[0] + ux * arrow_length,
                    line_start[1] + uy * arrow_length,
                )
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )
            elif directed:
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )

            if math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1]) > 0:
                ax.plot([line_start[0], line_end[0]], 
                       [line_start[1], line_end[1]], 
                       color=edge_color_mpl, 
                       linewidth=line_width_points,
                       solid_capstyle='round',
                       solid_joinstyle='round',
                       antialiased=True)
            
            # Arrow heads
            if bidirectional:
                self._add_arrow_head_to_ax(ax, trimmed_start, trimmed_end, arrow_length, arrow_width)
                self._add_arrow_head_to_ax(ax, trimmed_end, trimmed_start, arrow_length, arrow_width)
                rendered_pairs.add(pair)
            elif directed:
                self._add_arrow_head_to_ax(ax, trimmed_start, trimmed_end, arrow_length, arrow_width)

        # 2. Draw text labels and icons
        text_color_mpl = self._rgba_to_mpl(self._text_color)
        
        for vertex in vertices:
            center = positions.get(vertex.id)
            if center is None:
                continue
            label, icon_def = classifications.get(vertex.id, (vertex.id, None))
            radius = content_radii.get(vertex.id, 0.0)
            if radius <= 0:
                continue
            
            if icon_def:
                # Draw icon
                marker_style = MarkerStyle(icon_def.marker)
                marker_path = marker_style.get_path().transformed(marker_style.get_transform())
                transform = (
                    Affine2D()
                    .scale(radius * icon_def.scale)
                    .rotate(math.pi)
                    .translate(center[0], center[1])
                )
                fill = self._shape_fill_color if self._shape_fill_solid else self._shape_hollow_fill
                patch = PathPatch(
                    marker_path,
                    facecolor=self._rgba_to_mpl(fill),
                    edgecolor=self._rgba_to_mpl(self._shape_outline_color),
                    linewidth=self._shape_outline_width / max(self._supersample, 1),
                    transform=transform + ax.transData,
                )
                ax.add_patch(patch)
            else:
                # Draw text
                font_size_pixels = max(14 * self._supersample, int(radius * 1.1))
                font_size_points = font_size_pixels * 72 / dpi
                ax.text(
                    center[0], center[1],
                    str(label),
                    color=text_color_mpl,
                    fontsize=font_size_points,
                    ha='center',
                    va='center',
                    fontfamily='sans-serif',
                    fontweight='normal',
                    clip_on=False,
                )

        # Render once and composite
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        overlay = Image.fromarray(buffer, mode='RGBA')
        canvas.alpha_composite(overlay)
        plt.close(fig)

    def _add_arrow_head_to_ax(
        self,
        ax: plt.Axes,
        start: Tuple[float, float],
        end: Tuple[float, float],
        arrow_length: float,
        arrow_width: float,
    ) -> None:
        """Add arrow head polygon to existing axes."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        arrow_length = min(arrow_length, length * 0.9)
        base_x = end[0] - ux * arrow_length
        base_y = end[1] - uy * arrow_length
        half_width = arrow_width * 0.5
        left = (base_x + uy * half_width, base_y - ux * half_width)
        right = (base_x - uy * half_width, base_y + ux * half_width)
        
        triangle = plt.Polygon([end, left, right], 
                              facecolor=self._rgba_to_mpl(self._edge_color),
                              edgecolor='none')
        ax.add_patch(triangle)

    def _draw_edges(
        self,
        draw: ImageDraw.ImageDraw,
        edges: Sequence[Edge],
        positions: Dict[str, Tuple[float, float]],
        node_radius: float,
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]],
        content_radii: Dict[str, float],
        arrow_clearance: float,
        *,
        directed: bool,
    ) -> None:
        pair_counts: Dict[frozenset[str], int] = {}
        for edge in edges:
            pair = frozenset({edge.source, edge.target})
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        rendered_pairs: Set[frozenset[str]] = set()
        line_width = max(1, int(self._arrow_stroke_width * self._supersample))

        for edge in edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            start = positions[edge.source]
            end = positions[edge.target]
            if start == end:
                continue

            pair = frozenset({edge.source, edge.target})
            bidirectional = directed and pair_counts.get(pair, 0) > 1
            if bidirectional and pair in rendered_pairs:
                continue

            start_pad = self._content_padding(edge.source, content_radii, arrow_clearance)
            end_pad = self._content_padding(edge.target, content_radii, arrow_clearance)
            tapered = self._trim_edge(start, end, start_pad, end_pad)
            if tapered is None:
                continue
            trimmed_start, trimmed_end = tapered

            dx = trimmed_end[0] - trimmed_start[0]
            dy = trimmed_end[1] - trimmed_start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length == 0:
                continue
            ux, uy = dx / segment_length, dy / segment_length

            arrow_length, arrow_width = self._arrow_metrics(node_radius, segment_length)

            line_start = trimmed_start
            line_end = trimmed_end
            if bidirectional:
                line_start = (
                    line_start[0] + ux * arrow_length,
                    line_start[1] + uy * arrow_length,
                )
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )
            elif directed:
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )

            if math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1]) <= 0:
                line_segment = [trimmed_start, trimmed_end]
            else:
                line_segment = [line_start, line_end]

            draw.line(line_segment, fill=self._edge_color, width=line_width, joint="curve")

            if bidirectional:
                self._draw_arrow_head(
                    draw, trimmed_start, trimmed_end, arrow_length, arrow_width
                )
                self._draw_arrow_head(
                    draw, trimmed_end, trimmed_start, arrow_length, arrow_width
                )
                rendered_pairs.add(pair)
            elif directed:
                self._draw_arrow_head(
                    draw, trimmed_start, trimmed_end, arrow_length, arrow_width
                )

    def _draw_edges_matplotlib(
        self,
        canvas: Image.Image,
        edges: Sequence[Edge],
        positions: Dict[str, Tuple[float, float]],
        node_radius: float,
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]],
        content_radii: Dict[str, float],
        arrow_clearance: float,
        *,
        directed: bool,
    ) -> None:
        """Draw edges using matplotlib for consistent line widths across all angles."""
        if not edges:
            return
            
        pair_counts: Dict[frozenset[str], int] = {}
        for edge in edges:
            pair = frozenset({edge.source, edge.target})
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        rendered_pairs: Set[frozenset[str]] = set()
        
        # Create matplotlib figure matching canvas size
        width, height = canvas.size
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y to match image coordinates
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis('off')
        
        # Line width in points (matplotlib uses points, not pixels)
        # 1 point = dpi/72 pixels, so we need to convert
        line_width_pixels = self._arrow_stroke_width * self._supersample
        line_width_points = line_width_pixels * 72 / dpi
        
        edge_color_mpl = self._rgba_to_mpl(self._edge_color)

        for edge in edges:
            if edge.source not in positions or edge.target not in positions:
                continue
            start = positions[edge.source]
            end = positions[edge.target]
            if start == end:
                continue

            pair = frozenset({edge.source, edge.target})
            bidirectional = directed and pair_counts.get(pair, 0) > 1
            if bidirectional and pair in rendered_pairs:
                continue

            start_pad = self._content_padding(edge.source, content_radii, arrow_clearance)
            end_pad = self._content_padding(edge.target, content_radii, arrow_clearance)
            tapered = self._trim_edge(start, end, start_pad, end_pad)
            if tapered is None:
                continue
            trimmed_start, trimmed_end = tapered

            dx = trimmed_end[0] - trimmed_start[0]
            dy = trimmed_end[1] - trimmed_start[1]
            segment_length = math.hypot(dx, dy)
            if segment_length == 0:
                continue
            ux, uy = dx / segment_length, dy / segment_length

            arrow_length, arrow_width = self._arrow_metrics(node_radius, segment_length)

            line_start = trimmed_start
            line_end = trimmed_end
            if bidirectional:
                line_start = (
                    line_start[0] + ux * arrow_length,
                    line_start[1] + uy * arrow_length,
                )
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )
            elif directed:
                line_end = (
                    line_end[0] - ux * arrow_length,
                    line_end[1] - uy * arrow_length,
                )

            if math.hypot(line_end[0] - line_start[0], line_end[1] - line_start[1]) > 0:
                # Draw line with matplotlib (consistent width at all angles)
                ax.plot([line_start[0], line_end[0]], 
                       [line_start[1], line_end[1]], 
                       color=edge_color_mpl, 
                       linewidth=line_width_points,
                       solid_capstyle='round',
                       solid_joinstyle='round',
                       antialiased=True)
            
            # Draw arrow heads using matplotlib patches for consistency
            if bidirectional:
                self._draw_arrow_head_matplotlib(
                    ax, trimmed_start, trimmed_end, arrow_length, arrow_width
                )
                self._draw_arrow_head_matplotlib(
                    ax, trimmed_end, trimmed_start, arrow_length, arrow_width
                )
                rendered_pairs.add(pair)
            elif directed:
                self._draw_arrow_head_matplotlib(
                    ax, trimmed_start, trimmed_end, arrow_length, arrow_width
                )
        
        # Render to numpy array and composite onto canvas
        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        overlay = Image.fromarray(buffer, mode='RGBA')
        canvas.alpha_composite(overlay)
        plt.close(fig)

    def _draw_arrow_head_matplotlib(
        self,
        ax: plt.Axes,
        start: Tuple[float, float],
        end: Tuple[float, float],
        arrow_length: float,
        arrow_width: float,
    ) -> None:
        """Draw arrow head using matplotlib polygon for consistent rendering."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        arrow_length = min(arrow_length, length * 0.9)
        base_x = end[0] - ux * arrow_length
        base_y = end[1] - uy * arrow_length
        half_width = arrow_width * 0.5
        left = (base_x + uy * half_width, base_y - ux * half_width)
        right = (base_x - uy * half_width, base_y + ux * half_width)
        
        # Draw filled triangle
        triangle = plt.Polygon([end, left, right], 
                              facecolor=self._rgba_to_mpl(self._edge_color),
                              edgecolor='none')
        ax.add_patch(triangle)

    def _trim_edge(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        start_padding: float,
        end_padding: float,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= start_padding + end_padding:
            return None
        ratio_start = start_padding / distance
        ratio_end = end_padding / distance
        return (
            (start[0] + dx * ratio_start, start[1] + dy * ratio_start),
            (end[0] - dx * ratio_end, end[1] - dy * ratio_end),
        )

    def _draw_arrow_head(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[float, float],
        end: Tuple[float, float],
        arrow_length: float,
        arrow_width: float,
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        arrow_length = min(arrow_length, length * 0.9)
        base_x = end[0] - ux * arrow_length
        base_y = end[1] - uy * arrow_length
        half_width = arrow_width * 0.5
        left = (base_x + uy * half_width, base_y - ux * half_width)
        right = (base_x - uy * half_width, base_y + ux * half_width)
        draw.polygon([end, left, right], fill=self._edge_color)

    def _draw_vertices(
        self,
        canvas: Image.Image,
        draw: ImageDraw.ImageDraw,
        vertices: Sequence[Vertex],
        positions: Dict[str, Tuple[float, float]],
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]],
        content_radii: Dict[str, float],
    ) -> None:
        icon_jobs: List[IconRenderJob] = []
        text_jobs: List[Tuple[Tuple[float, float], str, float]] = []  # (center, label, radius)
        
        for vertex in vertices:
            center = positions.get(vertex.id)
            if center is None:
                continue
            label, icon_def = classifications.get(vertex.id, (vertex.id, None))
            radius = content_radii.get(vertex.id, 0.0)
            if radius <= 0:
                continue
            if icon_def:
                icon_jobs.append(IconRenderJob(icon=icon_def, center=center, node_radius=radius))
            else:
                text_jobs.append((center, label, radius))
        
        self._render_icon_jobs(canvas, icon_jobs)
        self._render_text_jobs(canvas, text_jobs)

    def _draw_circle_node(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: float,
        label: str,
    ) -> None:
        self._draw_label(draw, center, label, radius)

    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        label: str,
        radius: float,
    ) -> None:
        if radius <= 0:
            return
        font_size = max(14 * self._supersample, int(radius * 1.1))
        font = self._get_font(font_size)
        text = str(label)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        if text_width > radius * 2:
            scale = (radius * 1.8) / max(text_width, 1)
            font_size = max(12 * self._supersample, int(font_size * scale))
            font = self._get_font(font_size)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        position = (
            center[0] - text_width / 2,
            center[1] - text_height / 2 - font_size * 0.05,
        )
        draw.text(position, text, fill=self._text_color, font=font)

    def _render_icon_jobs(self, canvas: Image.Image, icon_jobs: Sequence[IconRenderJob]) -> None:
        if not icon_jobs:
            return

        width, height = canvas.size
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis("off")

        for job in icon_jobs:
            marker_style = MarkerStyle(job.icon.marker)
            marker_path = marker_style.get_path().transformed(marker_style.get_transform())
            transform = (
                Affine2D()
                .scale(job.node_radius * job.icon.scale)
                .rotate(math.pi)
                .translate(job.center[0], job.center[1])
            )
            fill = self._shape_fill_color if self._shape_fill_solid else self._shape_hollow_fill
            patch = PathPatch(
                marker_path,
                facecolor=self._rgba_to_mpl(fill),
                edgecolor=self._rgba_to_mpl(self._shape_outline_color),
                linewidth=self._shape_outline_width / max(self._supersample, 1),
                transform=transform + ax.transData,
            )
            ax.add_patch(patch)

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        overlay = Image.fromarray(buffer, mode="RGBA")
        canvas.alpha_composite(overlay)
        plt.close(fig)

    def _render_text_jobs(
        self, 
        canvas: Image.Image, 
        text_jobs: List[Tuple[Tuple[float, float], str, float]]
    ) -> None:
        """Render text labels using matplotlib for scientifically accurate centering.
        
        Matplotlib properly handles:
        - Font metrics (ascender, descender, baseline)
        - Vertical alignment (va='center' uses true geometric center)
        - Horizontal alignment (ha='center' uses glyph bounding box)
        """
        if not text_jobs:
            return

        width, height = canvas.size
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip Y to match image coordinates
        ax.set_facecolor((0, 0, 0, 0))
        ax.axis('off')

        text_color_mpl = self._rgba_to_mpl(self._text_color)

        for center, label, radius in text_jobs:
            # Calculate font size based on radius
            # Use points (typographic unit) for resolution-independent sizing
            font_size_pixels = max(14 * self._supersample, int(radius * 1.1))
            font_size_points = font_size_pixels * 72 / dpi
            
            # Matplotlib's text with proper centering
            ax.text(
                center[0], center[1],
                str(label),
                color=text_color_mpl,
                fontsize=font_size_points,
                ha='center',  # Horizontal alignment: center
                va='center',  # Vertical alignment: center (geometric, not baseline)
                fontfamily='sans-serif',
                fontweight='normal',
                clip_on=False,
            )

        fig.canvas.draw()
        buffer = np.asarray(fig.canvas.buffer_rgba())
        overlay = Image.fromarray(buffer, mode='RGBA')
        canvas.alpha_composite(overlay)
        plt.close(fig)

    @staticmethod
    def _rgba_to_mpl(color: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        return tuple(component / 255 for component in color)

    # ------------------------------------------------------------------ #
    # Shape rendering
    # ------------------------------------------------------------------ #

    def _build_shape_renderers(self) -> Dict[str, Callable]:
        return {
            "circle": self._shape_circle,
            "ellipse": self._shape_ellipse,
            "square": self._shape_square,
            "rectangle": self._shape_rectangle,
            "diamond": self._shape_diamond,
            "triangle": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 3, fill, stroke),
            "pentagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 5, fill, stroke),
            "hexagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 6, fill, stroke),
            "heptagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 7, fill, stroke),
            "octagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 8, fill, stroke),
            "nonagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 9, fill, stroke),
            "decagon": lambda d, c, r, fill, stroke: self._shape_polygon(d, c, r, 10, fill, stroke),
            "star": lambda d, c, r, fill, stroke: self._shape_star(d, c, r, 5, fill, stroke),
            "star5": lambda d, c, r, fill, stroke: self._shape_star(d, c, r, 5, fill, stroke),
            "star6": lambda d, c, r, fill, stroke: self._shape_star(d, c, r, 6, fill, stroke),
            "star8": lambda d, c, r, fill, stroke: self._shape_star(d, c, r, 8, fill, stroke),
            "cross": self._shape_cross,
            "plus": self._shape_cross,
            "arrow": self._shape_arrow,
            "chevron": self._shape_chevron,
            "shield": self._shape_shield,
            "heart": self._shape_heart,
            "cloud": self._shape_cloud,
            "capsule": self._shape_capsule,
            "kite": self._shape_kite,
            "drop": self._shape_drop,
            "moon": self._shape_moon,
        }

    def _build_shape_aliases(self) -> Dict[str, str]:
        aliases = {}
        for name in list(self._shape_renderers.keys()):
            normalized = self._normalize_shape_key(name)
            aliases[normalized] = name
        aliases.update(
            {
                "rect": "rectangle",
                "diamondshape": "diamond",
                "rhombus": "diamond",
                "parallelogram": "diamond",
                "starfive": "star5",
                "starsix": "star6",
                "stareight": "star8",
                "arrowhead": "arrow",
                "crescent": "moon",
                "droplet": "drop",
                "shieldshape": "shield",
                "capsuleshape": "capsule",
                "kite shape": "kite",
            }
        )
        return aliases

    def _resolve_shape_renderer(
        self,
        label: str,
        label_set_hint: Optional[str],
    ) -> Optional[Callable]:
        if not label:
            return None
        key = self._normalize_shape_key(label)
        renderer_key = self._shape_aliases.get(key)
        if renderer_key:
            return self._shape_renderers.get(renderer_key)
        if label_set_hint and label_set_hint.lower() == "shape":
            # Treat all entries as shapes by falling back to circle if unknown.
            return self._shape_renderers.get("circle")
        return None

    def _shape_circle(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: float,
        fill: Tuple[int, int, int],
        stroke: Tuple[int, int, int, int],
    ) -> None:
        bbox = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        draw.ellipse(bbox, fill=fill, outline=stroke, width=self._shape_outline_width)

    def _shape_ellipse(self, draw, center, radius, fill, stroke) -> None:
        bbox = [
            center[0] - radius,
            center[1] - radius * 0.7,
            center[0] + radius,
            center[1] + radius * 0.7,
        ]
        draw.ellipse(bbox, fill=fill, outline=stroke, width=self._shape_outline_width)

    def _shape_rectangle(self, draw, center, radius, fill, stroke) -> None:
        bbox = [
            center[0] - radius,
            center[1] - radius * 0.8,
            center[0] + radius,
            center[1] + radius * 0.8,
        ]
        draw.rectangle(bbox, fill=fill, outline=stroke, width=self._shape_outline_width)

    def _shape_square(self, draw, center, radius, fill, stroke) -> None:
        bbox = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        draw.rectangle(bbox, fill=fill, outline=stroke, width=self._shape_outline_width)

    def _shape_diamond(self, draw, center, radius, fill, stroke) -> None:
        points = [
            (center[0], center[1] - radius),
            (center[0] + radius, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius, center[1]),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_polygon(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: float,
        sides: int,
        fill: Tuple[int, int, int],
        stroke: Tuple[int, int, int, int],
    ) -> None:
        if sides < 3:
            return
        points = []
        for idx in range(sides):
            angle = -math.pi / 2 + idx * (2 * math.pi / sides)
            points.append(
                (
                    center[0] + math.cos(angle) * radius,
                    center[1] + math.sin(angle) * radius,
                )
            )
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _outline_polygon(
        self,
        draw: ImageDraw.ImageDraw,
        points: List[Tuple[float, float]],
        stroke: Tuple[int, int, int, int],
    ) -> None:
        draw.line(points + [points[0]], fill=stroke, width=self._shape_outline_width, joint="curve")

    def _arrow_metrics(
        self,
        node_radius: float,
        segment_length: float,
    ) -> Tuple[float, float]:
        base_length = node_radius * 0.4
        arrow_length = min(base_length, segment_length * 0.25)
        arrow_width = base_length * 0.85
        return arrow_length, arrow_width

    def _shape_star(
        self,
        draw: ImageDraw.ImageDraw,
        center: Tuple[float, float],
        radius: float,
        points_count: int,
        fill: Tuple[int, int, int],
        stroke: Tuple[int, int, int, int],
    ) -> None:
        if points_count < 4:
            points_count = 5
        outer = []
        inner_radius = radius * 0.45
        for idx in range(points_count * 2):
            r = radius if idx % 2 == 0 else inner_radius
            angle = -math.pi / 2 + idx * math.pi / points_count
            outer.append(
                (
                    center[0] + math.cos(angle) * r,
                    center[1] + math.sin(angle) * r,
                )
            )
        draw.polygon(outer, fill=fill)
        self._outline_polygon(draw, outer, stroke)

    def _shape_cross(self, draw, center, radius, fill, stroke) -> None:
        arm = radius * 0.35
        points = [
            (center[0] - arm, center[1] - radius),
            (center[0] + arm, center[1] - radius),
            (center[0] + arm, center[1] - arm),
            (center[0] + radius, center[1] - arm),
            (center[0] + radius, center[1] + arm),
            (center[0] + arm, center[1] + arm),
            (center[0] + arm, center[1] + radius),
            (center[0] - arm, center[1] + radius),
            (center[0] - arm, center[1] + arm),
            (center[0] - radius, center[1] + arm),
            (center[0] - radius, center[1] - arm),
            (center[0] - arm, center[1] - arm),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_arrow(self, draw, center, radius, fill, stroke) -> None:
        shaft = radius * 0.45
        points = [
            (center[0] - shaft, center[1] - shaft * 0.4),
            (center[0] - shaft, center[1] + shaft * 0.4),
            (center[0], center[1] + shaft * 0.4),
            (center[0], center[1] + radius * 0.8),
            (center[0] + radius, center[1]),
            (center[0], center[1] - radius * 0.8),
            (center[0], center[1] - shaft * 0.4),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_chevron(self, draw, center, radius, fill, stroke) -> None:
        w = radius
        points = [
            (center[0] - w, center[1] - radius * 0.2),
            (center[0], center[1] - radius),
            (center[0] + w, center[1] - radius * 0.2),
            (center[0] + w * 0.6, center[1] + radius),
            (center[0], center[1] + radius * 0.2),
            (center[0] - w * 0.6, center[1] + radius),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_shield(self, draw, center, radius, fill, stroke) -> None:
        top = radius * 0.55
        points = [
            (center[0] - radius, center[1] - top),
            (center[0] + radius, center[1] - top),
            (center[0] + radius * 0.7, center[1] + radius * 0.3),
            (center[0], center[1] + radius),
            (center[0] - radius * 0.7, center[1] + radius * 0.3),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_heart(self, draw, center, radius, fill, stroke) -> None:
        offset = radius * 0.3
        left = [
            (center[0] - radius, center[1]),
            (center[0], center[1] + radius),
            (center[0], center[1] + radius * 0.7),
            (center[0] - offset, center[1] - radius * 0.2),
        ]
        right = [
            (center[0] + offset, center[1] - radius * 0.2),
            (center[0], center[1] + radius * 0.7),
            (center[0], center[1] + radius),
            (center[0] + radius, center[1]),
        ]
        points = left + right
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_cloud(self, draw, center, radius, fill, stroke) -> None:
        offsets = [
            (-radius * 0.7, 0),
            (-radius * 0.2, -radius * 0.4),
            (radius * 0.3, -radius * 0.3),
            (radius * 0.7, 0),
            (radius * 0.2, radius * 0.4),
            (-radius * 0.3, radius * 0.3),
        ]
        for offset in offsets:
            bbox = [
                center[0] + offset[0] - radius * 0.5,
                center[1] + offset[1] - radius * 0.35,
                center[0] + offset[0] + radius * 0.5,
                center[1] + offset[1] + radius * 0.35,
            ]
            draw.ellipse(bbox, fill=fill, outline=None)
        draw.ellipse(
            [
                center[0] - radius * 1.1,
                center[1] - radius * 0.4,
                center[0] + radius * 1.1,
                center[1] + radius * 0.8,
            ],
            outline=stroke,
            width=self._shape_outline_width,
        )

    def _shape_capsule(self, draw, center, radius, fill, stroke) -> None:
        width = radius * 1.4
        height = radius * 0.9
        draw.rounded_rectangle(
            [
                center[0] - width,
                center[1] - height,
                center[0] + width,
                center[1] + height,
            ],
            radius=height,
            fill=fill,
            outline=stroke,
            width=self._shape_outline_width,
        )

    def _shape_kite(self, draw, center, radius, fill, stroke) -> None:
        points = [
            (center[0], center[1] - radius),
            (center[0] + radius * 0.7, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius * 0.4, center[1]),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_drop(self, draw, center, radius, fill, stroke) -> None:
        points = [
            (center[0], center[1] - radius),
            (center[0] + radius * 0.7, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius * 0.7, center[1]),
        ]
        draw.polygon(points, fill=fill)
        self._outline_polygon(draw, points, stroke)

    def _shape_moon(self, draw, center, radius, fill, stroke) -> None:
        outer = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        inner = [
            center[0] - radius * 0.4,
            center[1] - radius,
            center[0] + radius * 1.1,
            center[1] + radius,
        ]
        draw.ellipse(outer, fill=fill, outline=stroke, width=self._shape_outline_width)
        draw.ellipse(inner, fill=self._background_color, outline=None)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _classify_vertices(
        self,
        vertices: Sequence[Vertex],
        vertex_labels: Dict[str, str],
        label_set_hint: Optional[str],
    ) -> Dict[str, Tuple[str, Optional[IconDefinition]]]:
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]] = {}
        for vertex in vertices:
            label = vertex_labels.get(vertex.id, vertex.id)
            icon_def = lookup_icon(label)
            classifications[vertex.id] = (label, icon_def)
        return classifications

    def _compute_content_radii(
        self,
        vertices: Sequence[Vertex],
        classifications: Dict[str, Tuple[str, Optional[IconDefinition]]],
        node_radius: float,
        arrow_clearance: float,
    ) -> Dict[str, float]:
        max_radius = node_radius * 0.95
        radii: Dict[str, float] = {}
        for vertex in vertices:
            _, icon_def = classifications.get(vertex.id, (vertex.id, None))
            if icon_def:
                base_radius = node_radius * self._shape_scale * max(icon_def.scale, 0.1)
            else:
                base_radius = node_radius * self._shape_scale * 0.75
            radii[vertex.id] = min(base_radius, max_radius)
        return radii

    def _grid_size(self, graph: GeneratedGraph) -> Tuple[int, int]:
        grid = getattr(graph, "grid_size", None) or graph["grid_size"]
        return int(grid[0]), int(grid[1])

    def _is_directed(self, graph: GeneratedGraph) -> bool:
        directed_attr = getattr(graph, "directed", None)
        if directed_attr is not None:
            return bool(directed_attr)
        return bool(graph["directed"])

    def _coerce_vertices(self, graph: GeneratedGraph) -> List[Vertex]:
        raw_vertices = getattr(graph, "vertices", None) or graph["vertices"]
        vertices: List[Vertex] = []
        for entry in raw_vertices:
            if isinstance(entry, Vertex):
                vertices.append(entry)
            else:
                vertices.append(Vertex(id=entry["id"], row=entry["row"], col=entry["col"]))
        return vertices

    def _coerce_edges(self, graph: GeneratedGraph) -> List[Edge]:
        raw_edges = getattr(graph, "edges", None) or graph["edges"]
        edges: List[Edge] = []
        for entry in raw_edges:
            if isinstance(entry, Edge):
                edges.append(entry)
            else:
                edges.append(Edge(source=entry["source"], target=entry["target"]))
        return edges

    def _place_vertices(
        self,
        vertices: Sequence[Vertex],
        margin: float,
        cell: float,
    ) -> Dict[str, Tuple[float, float]]:
        positions: Dict[str, Tuple[float, float]] = {}
        for vertex in vertices:
            x = margin + vertex.col * cell + cell / 2
            y = margin + vertex.row * cell + cell / 2
            positions[vertex.id] = (x, y)
        return positions

    def _color_for_vertex(self, vertex_id: str) -> Tuple[int, int, int]:
        idx = abs(hash(vertex_id)) % len(self._PALETTE)
        return self._PALETTE[idx]

    def _content_padding(
        self,
        vertex_id: str,
        content_radii: Dict[str, float],
        arrow_clearance: float,
    ) -> float:
        radius = content_radii.get(vertex_id)
        if radius is None:
            radius = arrow_clearance
        return radius + arrow_clearance * 0.5

    def _normalize_shape_key(self, label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalnum())

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        cached = self._font_cache.get(size)
        if cached:
            return cached
        for font_name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                font = ImageFont.truetype(font_name, size=size)
                self._font_cache[size] = font
                return font
            except OSError:
                continue
        font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    def _derive_identifier(self, graph: GeneratedGraph) -> str:
        rows, cols = self._grid_size(graph)
        vertex_count = len(self._coerce_vertices(graph))
        edge_count = len(self._coerce_edges(graph))
        return f"graph_{rows}x{cols}_{vertex_count}v_{edge_count}e"
