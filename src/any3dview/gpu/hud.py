"""OpenGL screen-space HUD with a cached Pillow-generated text atlas."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

try:
    import moderngl
except ImportError as error:  # pragma: no cover - malformed optional install
    raise ImportError("ANY3dView GPU HUD requires the 'gpu' extra") from error

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:  # pragma: no cover - malformed optional install
    raise ImportError("ANY3dView GPU HUD requires Pillow from the 'gpu' extra") from error

from ..core import parse_color


_VERTEX = """
#version 330
uniform vec2 u_viewport;
in vec2 in_pixel;
in float in_depth;
in vec2 in_uv;
in vec4 in_color;
out vec2 v_uv;
out vec4 v_color;
void main() {
    vec2 ndc = vec2(
        2.0 * in_pixel.x / u_viewport.x - 1.0,
        1.0 - 2.0 * in_pixel.y / u_viewport.y
    );
    gl_Position = vec4(ndc, 2.0 * in_depth - 1.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
"""

_FRAGMENT = """
#version 330
uniform sampler2D u_atlas;
uniform vec2 u_atlas_size;
in vec2 v_uv;
in vec4 v_color;
out vec4 frag_color;
void main() {
    float coverage = texture(u_atlas, v_uv / u_atlas_size).r;
    float alpha = coverage * v_color.a;
    if (alpha <= 0.001) discard;
    frag_color = vec4(v_color.rgb, alpha);
}
"""


def _rgba(color: object, alpha: float = 1.0) -> tuple[float, float, float, float]:
    parsed = parse_color(str(color)) or (31, 41, 55)
    return (
        parsed[0] / 255.0,
        parsed[1] / 255.0,
        parsed[2] / 255.0,
        max(0.0, min(1.0, float(alpha))),
    )


@dataclass(frozen=True, slots=True)
class _AtlasEntry:
    width: int
    height: int
    x: int
    y: int
    uv: tuple[float, float, float, float]


class _TextAtlas:
    """Fixed-size shelf-packed alpha atlas; text/font pairs are cached."""

    def __init__(self, context: Any, size: int = 2048) -> None:
        self.context = context
        self.size = int(size)
        self.max_size = max(
            self.size,
            int(context.info.get("GL_MAX_TEXTURE_SIZE", 8192)),
        )
        self.fonts: dict[tuple[object, ...], Any] = {}
        self._reset()

    def _reset(self) -> None:
        previous = getattr(self, "texture", None)
        if previous is not None:
            previous.release()
        self.image = Image.new("L", (self.size, self.size), 0)
        self.image.paste(255, (0, 0, 2, 2))
        self.texture = self.context.texture((self.size, self.size), 1, dtype="f1")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.entries: dict[tuple[str, tuple[object, ...]], _AtlasEntry] = {}
        self.x = 4
        self.y = 4
        self.row_height = 0
        self.dirty = True

    def begin_frame(self) -> None:
        # Animated/dynamic labels must not accumulate forever.  Rebuilding at
        # a frame boundary is safe because the HUD vertex batch was cleared.
        if len(self.entries) > 4096 or self.y + self.row_height > self.size * 0.8:
            self._reset()

    def _grow(self) -> bool:
        new_size = min(self.max_size, self.size * 2)
        if new_size <= self.size:
            return False
        old_image = self.image
        old_entries = self.entries
        old_texture = self.texture
        self.size = new_size
        self.image = Image.new("L", (self.size, self.size), 0)
        self.image.paste(old_image, (0, 0))
        self.texture = self.context.texture((self.size, self.size), 1, dtype="f1")
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.entries = old_entries
        old_texture.release()
        self.dirty = True
        return True

    @property
    def white_uv(self) -> tuple[float, float]:
        return 1.0, 1.0

    @staticmethod
    def _font_key(font: object) -> tuple[object, ...]:
        if isinstance(font, (tuple, list)):
            values = tuple(font)
        else:
            values = (str(font), 9, "")
        family = str(values[0]) if values else "Segoe UI"
        size = max(6, int(values[1])) if len(values) > 1 else 9
        style = str(values[2]).lower() if len(values) > 2 else ""
        return family, size, style

    def _font(self, font: object) -> tuple[tuple[object, ...], Any]:
        key = self._font_key(font)
        existing = self.fonts.get(key)
        if existing is not None:
            return key, existing
        family, size, style = key
        candidates: list[str] = [str(family)]
        if str(family).casefold() in {"segoe ui", "segoeui"}:
            filename = "segoeuib.ttf" if "bold" in str(style) else "segoeui.ttf"
            candidates.insert(0, str(Path("C:/Windows/Fonts") / filename))
        candidates.extend(
            ["DejaVuSans-Bold.ttf" if "bold" in str(style) else "DejaVuSans.ttf"]
        )
        loaded = None
        for candidate in candidates:
            try:
                loaded = ImageFont.truetype(candidate, int(size))
                break
            except (OSError, ValueError):
                continue
        if loaded is None:
            loaded = ImageFont.load_default()
        self.fonts[key] = loaded
        return key, loaded

    def text(self, value: object, font: object) -> _AtlasEntry:
        text = str(value)
        font_key, loaded = self._font(font)
        key = text, font_key
        existing = self.entries.get(key)
        if existing is not None:
            return existing
        scratch = ImageDraw.Draw(self.image)
        bbox = scratch.textbbox((0, 0), text or " ", font=loaded, stroke_width=0)
        padding = 2
        width = max(1, bbox[2] - bbox[0]) + 2 * padding
        height = max(1, bbox[3] - bbox[1]) + 2 * padding
        if self.x + width >= self.size:
            self.x = 4
            self.y += self.row_height + 2
            self.row_height = 0
        if self.y + height >= self.size:
            if self._grow():
                return self.text(value, font)
            # A single frame can theoretically exceed the hardware texture
            # limit.  Keep rendering deterministically instead of taking down
            # the viewer; subsequent frame-boundary eviction recovers space.
            white = self.white_uv
            return _AtlasEntry(2, 2, 0, 0, (*white, *white))
        scratch.text(
            (self.x + padding - bbox[0], self.y + padding - bbox[1]),
            text,
            fill=255,
            font=loaded,
        )
        entry = _AtlasEntry(
            width,
            height,
            self.x,
            self.y,
            (
                float(self.x),
                float(self.y),
                float(self.x + width),
                float(self.y + height),
            ),
        )
        self.entries[key] = entry
        self.x += width + 2
        self.row_height = max(self.row_height, height)
        self.dirty = True
        return entry

    def bind(self, location: int = 0) -> None:
        if self.dirty:
            self.texture.write(self.image.tobytes())
            self.dirty = False
        self.texture.use(location=location)

    def release(self) -> None:
        self.texture.release()


class GPUHudRenderer:
    """Small dynamic 2D batch independent of retained scene resources."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.program = context.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        self.buffer = context.buffer(reserve=8 * 4 * 1024)
        self.vao = context.vertex_array(
            self.program,
            [(self.buffer, "2f 1f 2f 4f", "in_pixel", "in_depth", "in_uv", "in_color")],
        )
        self.atlas = _TextAtlas(context)
        self.viewport = (1, 1)
        self.vertices: list[tuple[float, ...]] = []
        self.depth_vertices: list[tuple[float, ...]] = []
        self.uploads = 0

    def begin(self, viewport: tuple[int, int]) -> None:
        self.viewport = max(1, int(viewport[0])), max(1, int(viewport[1]))
        self.vertices.clear()
        self.depth_vertices.clear()
        self.atlas.begin_frame()

    def _vertex(
        self,
        point: tuple[float, float],
        uv: tuple[float, float],
        color: tuple[float, float, float, float],
        depth: Optional[float] = None,
    ) -> tuple[float, ...]:
        return (
            point[0], point[1],
            0.0 if depth is None else max(0.0, min(1.0, float(depth))),
            uv[0], uv[1], *color,
        )

    def _append(
        self,
        point: tuple[float, float],
        uv: tuple[float, float],
        color: tuple[float, float, float, float],
        depth: Optional[float],
    ) -> None:
        target = self.vertices if depth is None else self.depth_vertices
        target.append(self._vertex(point, uv, color, depth))

    def quad(
        self,
        rect: tuple[float, float, float, float],
        color: object,
        *,
        alpha: float = 1.0,
        uv: Optional[tuple[float, float, float, float]] = None,
        depth: Optional[float] = None,
    ) -> None:
        x0, y0, x1, y1 = (float(value) for value in rect)
        if uv is None:
            white = self.atlas.white_uv
            u0 = u1 = white[0]
            v0 = v1 = white[1]
        else:
            u0, v0, u1, v1 = uv
        rgba = _rgba(color, alpha)
        corners = (
            ((x0, y0), (u0, v0)), ((x1, y0), (u1, v0)),
            ((x1, y1), (u1, v1)), ((x0, y1), (u0, v1)),
        )
        for index in (0, 1, 2, 0, 2, 3):
            point, texture = corners[index]
            self._append(point, texture, rgba, depth)

    def line(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        color: object,
        *,
        width: float = 1.0,
        alpha: float = 1.0,
        depth: Optional[float] = None,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        length = max(1.0e-9, math.hypot(dx, dy))
        half = max(0.5, float(width) * 0.5)
        nx, ny = -dy * half / length, dx * half / length
        rgba = _rgba(color, alpha)
        uv = self.atlas.white_uv
        points = (
            (x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
            (x1 - nx, y1 - ny), (x0 - nx, y0 - ny),
        )
        for index in (0, 1, 2, 0, 2, 3):
            self._append(points[index], uv, rgba, depth)

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: object,
        *,
        width: float = 1.0,
        closed: bool = False,
        alpha: float = 1.0,
        depth: Optional[float] = None,
    ) -> None:
        if len(points) < 2:
            return
        stop = len(points) if closed else len(points) - 1
        for index in range(stop):
            self.line(
                points[index], points[(index + 1) % len(points)], color,
                width=width, alpha=alpha, depth=depth,
            )

    def rectangle(
        self,
        rect: tuple[float, float, float, float],
        outline: object,
        *,
        width: float = 1.0,
        fill: Optional[object] = None,
        fill_alpha: float = 0.2,
        depth: Optional[float] = None,
    ) -> None:
        x0, y0, x1, y1 = rect
        if fill is not None:
            self.quad((x0, y0, x1, y1), fill, alpha=fill_alpha, depth=depth)
        self.polyline(
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
            outline, width=width, closed=True, depth=depth,
        )

    def circle(
        self,
        center: tuple[float, float],
        radius: float,
        color: object,
        *,
        width: float = 1.0,
        depth: Optional[float] = None,
    ) -> None:
        total = 20
        points = tuple(
            (
                center[0] + radius * math.cos(2 * math.pi * index / total),
                center[1] + radius * math.sin(2 * math.pi * index / total),
            )
            for index in range(total)
        )
        self.polyline(points, color, width=width, closed=True, depth=depth)

    def text(
        self,
        point: tuple[float, float],
        value: object,
        color: object = "black",
        *,
        font: object = ("Segoe UI", 9, ""),
        anchor: str = "center",
        depth: Optional[float] = None,
    ) -> None:
        entry = self.atlas.text(value, font)
        x, y = float(point[0]), float(point[1])
        normalized = str(anchor).casefold()
        if "e" in normalized and normalized != "center":
            x -= entry.width
        elif "w" not in normalized:
            x -= 0.5 * entry.width
        if "s" in normalized:
            y -= entry.height
        elif "n" not in normalized:
            y -= 0.5 * entry.height
        self.quad(
            (x, y, x + entry.width, y + entry.height), color,
            uv=entry.uv, depth=depth,
        )

    def render(self, target: Any = None) -> None:
        if not self.vertices and not self.depth_vertices:
            return
        framebuffer = target or self.context.screen
        framebuffer.use()
        self.context.viewport = (0, 0, *self.viewport)
        self.context.disable(moderngl.CULL_FACE)
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.context.depth_mask = False
        self.atlas.bind(0)
        self.program["u_atlas"].value = 0
        self.program["u_atlas_size"].value = (
            float(self.atlas.size), float(self.atlas.size)
        )
        self.program["u_viewport"].value = tuple(float(v) for v in self.viewport)

        def draw(vertices: list[tuple[float, ...]], depth_test: bool) -> None:
            if not vertices:
                return
            values = np.asarray(vertices, dtype=np.float32)
            payload = values.tobytes()
            if self.buffer.size < len(payload):
                self.buffer.orphan(max(len(payload), self.buffer.size * 2))
            self.buffer.write(payload)
            self.uploads += 1
            if depth_test:
                self.context.enable(moderngl.DEPTH_TEST)
            else:
                self.context.disable(moderngl.DEPTH_TEST)
            self.vao.render(mode=moderngl.TRIANGLES, vertices=len(values))

        draw(self.depth_vertices, True)
        draw(self.vertices, False)
        self.context.depth_mask = True
        self.context.disable(moderngl.BLEND)

    def release(self) -> None:
        self.atlas.release()
        self.vao.release()
        self.buffer.release()
        self.program.release()


__all__ = ["GPUHudRenderer"]
