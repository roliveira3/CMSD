from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

IconMarker = Union[str, Path]
RGBAColor = Tuple[int, int, int, int]


@dataclass(frozen=True)
class IconDefinition:
    """Describes a reusable icon sources for semantic shape rendering."""

    name: str
    marker: IconMarker
    scale: float = 1.0
    fill_color: Optional[RGBAColor] = None


_NORMALIZE_RE = re.compile(r"[^0-9a-z]+")
_GLYPH_FONT = FontProperties(family="DejaVu Sans")


def normalize_icon_key(label: str) -> str:
    if not label:
        return ""
    return _NORMALIZE_RE.sub("", label.lower())


def _normalized_path(path: TextPath) -> Path:
    vertices = path.vertices
    min_x = float(vertices[:, 0].min())
    max_x = float(vertices[:, 0].max())
    min_y = float(vertices[:, 1].min())
    max_y = float(vertices[:, 1].max())
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y) or 1.0
    normalized = vertices.copy()
    normalized[:, 0] = (normalized[:, 0] - center_x) / span
    normalized[:, 1] = (normalized[:, 1] - center_y) / span
    return Path(normalized, path.codes)


def _glyph_marker(symbol: str) -> Path:
    text_path = TextPath((0, 0), symbol, prop=_GLYPH_FONT, size=1.0)
    return _normalized_path(text_path)


def _star_path(points: int) -> Path:
    vertices: list[Tuple[float, float]] = []
    codes: list[int] = []
    angle_step = math.pi / points
    for idx in range(points * 2):
        radius = 1.0 if idx % 2 == 0 else 0.45
        angle = -math.pi / 2 + idx * angle_step
        vertices.append((math.cos(angle) * radius, math.sin(angle) * radius))
        codes.append(Path.MOVETO if idx == 0 else Path.LINETO)
    vertices.append(vertices[0])
    codes.append(Path.CLOSEPOLY)
    return Path(vertices, codes)


def _crescent_path() -> Path:
    outer: list[Tuple[float, float]] = []
    inner: list[Tuple[float, float]] = []
    steps = 64
    offset = 0.35
    for idx in range(steps + 1):
        angle = 2 * math.pi * idx / steps
        outer.append((math.cos(angle), math.sin(angle)))
    for idx in range(steps, -1, -1):
        angle = 2 * math.pi * idx / steps
        inner.append((math.cos(angle) * 0.7 + offset, math.sin(angle) * 0.7))
    vertices = outer + inner
    codes = [Path.MOVETO] + [Path.LINETO] * (len(outer) - 1)
    codes += [Path.LINETO] * len(inner)
    codes[-1] = Path.CLOSEPOLY
    return Path(vertices, codes)


ICON_SPECS: Sequence[IconDefinition] = [
    IconDefinition("Circle", "o", scale=0.93),
    IconDefinition("Square", "s", scale=0.88),
    IconDefinition("Triangle", "^", scale=0.88),
    IconDefinition("Pentagon", "p", scale=0.9),
    IconDefinition("Hexagon", "h", scale=0.9),
    IconDefinition("Star", _star_path(5), scale=0.95),
    IconDefinition("Cross", "X", scale=0.9),
    IconDefinition("Heart", _glyph_marker("\N{BLACK HEART SUIT}"), scale=0.92),
    IconDefinition("Spade", r"$\spadesuit$", scale=0.92),
    IconDefinition("Club", r"$\clubsuit$", scale=0.92),
    IconDefinition("Diamond", _glyph_marker("\N{BLACK DIAMOND SUIT}"), scale=0.92),
]

ICON_LOOKUP: Dict[str, IconDefinition] = {
    normalize_icon_key(spec.name): spec for spec in ICON_SPECS
}
ICON_LABELS: Sequence[str] = [spec.name for spec in ICON_SPECS]


def lookup_icon(label: Optional[str]) -> Optional[IconDefinition]:
    if not label:
        return None
    key = normalize_icon_key(label)
    return ICON_LOOKUP.get(key)
