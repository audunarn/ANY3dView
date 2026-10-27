"""Frozen pre-style-reuse painter oracle; deliberately independent of optimized state handling."""
import numpy as np
from any3dview.qt import (QPainter,QColor,QPen,QPolygonF,QPointF,Qt,Point3D,_FrameProjection)

def reference_paint_software(self, painter):
    self._prune_software_prep()
    pixel_ratio = self.pixel_ratio
    painter.setRenderHint(QPainter.Antialiasing)
    ratio = self.surface.devicePixelRatioF()
    painter.scale(1/ratio, 1/ratio)
    width, height = self.viewport_size
    painter.fillRect(0, 0, width, height, QColor(self.bg))
    # One camera basis/viewport/scale evaluation serves every projection
    # in the frame; the retained preparation below removes the
    # camera-invariant per-primitive work from repeated frames.
    frame = _FrameProjection(self)
    section = self._section_plane if self._section_plane is not None and self._section_plane.enabled else None
    faces, lines, points, outlines = [], [], [], []
    for entry in self._display_entries().values():
        handle = entry["handle"]
        if handle.removed or not handle.visible:
            continue
        appearance = dict(entry["appearance"],scalar_range=handle.appearance.scalar_range,colormap=handle.appearance.colormap)
        if appearance.get("depth_only"):
            continue
        for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:
            prep = self._software_chunk_prep(entry, handle, chunk_id, mesh, section)
            world = prep.world
            screen_all, visible_all = self._project_world_points(world, frame)
            triangle_visible = None
            if section is None and len(mesh.triangles):
                triangle_visible = visible_all[np.asarray(mesh.triangles)].all(axis=1)
            visible_triangles = []
            for position, index in enumerate(prep.triangles):
                if section is not None:
                    verts = prep.section_polygons[position]
                    screen, visible = self._project_world_points(verts, frame)
                    if len(screen)<3 or not visible.all():
                        continue
                else:
                    if not triangle_visible[index]:
                        continue
                    screen = screen_all[mesh.triangles[index]]
                visible_triangles.append(index)
                faces.append((sum(p[2] for p in screen)/len(screen), screen, prep.colors[position], appearance))
            if self.show_mesh_lines and (appearance.get("mesh_lines") or entry.get("outline_groups") is None):
                if section is None and visible_all.all():
                    # All admitted triangles are inside near/far: reuse their
                    # exact semantic boundaries and this frame's projection.
                    segments = ((tuple(screen_all[a]), tuple(screen_all[b]))
                                for loop in prep.outline_loops
                                for a, b in ([loop] if len(loop) == 2 else zip(loop, (*loop[1:], loop[0]))))
                else:
                    segments = self._surface_outline_segments(entry, chunk_id, mesh, visible_triangles, world, frame)
                for screen in segments:
                    outlines.append((sum(point[2] for point in screen) / len(screen), screen, appearance))
            for segment, selected in zip(prep.line_segments, prep.line_selected):
                a,b=(self._project_point_with_frame(Point3D(*p), frame) for p in segment)
                if a is not None and b is not None:
                    style=dict(appearance)
                    if selected:style["line_color"]=self._highlight_fill
                    lines.append((a,b,style))
            for node, selected in zip(prep.points, prep.point_selected):
                if not visible_all[node]:continue
                style=dict(appearance)
                if selected:style["point_color"]=self._highlight_fill
                points.append((tuple(screen_all[node]),style))
    surfaces = [(depth, 0, screen, color, appearance) for depth, screen, color, appearance in faces]
    surfaces.extend((depth, 1, screen, None, appearance) for depth, screen, appearance in outlines)
    # Keep the software painter's depth ordering: nearer fills must cover
    # more distant boundaries. At equal depth, fill before its own outline.
    for _, kind, screen, color, appearance in sorted(surfaces, key=lambda surface: (-float(f"{surface[0]:.12g}"), surface[1])):
        polygon = QPolygonF([QPointF(*point[:2]) for point in screen])
        if kind == 0:
            # Antialiasing each tessellation triangle independently leaves
            # pale seams where their partially covered edge pixels meet.
            painter.setRenderHint(QPainter.Antialiasing, False)
            fill=QColor(color); fill.setAlphaF(appearance["opacity"])
            painter.setBrush(fill)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(polygon)
        else:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(appearance["line_color"]), appearance["line_width"]*pixel_ratio))
            painter.drawPolyline(polygon)
    painter.setRenderHint(QPainter.Antialiasing, True)
    for a,b,appearance in lines:
        painter.setPen(QPen(QColor(appearance["line_color"]),appearance["line_width"]*pixel_ratio))
        painter.drawLine(QPointF(*a[:2]),QPointF(*b[:2]))
    for p,appearance in points:
        painter.setBrush(QColor(appearance["point_color"]))
        painter.setPen(Qt.NoPen)
        r=appearance["point_size"]*pixel_ratio/2
        painter.drawEllipse(QPointF(*p[:2]),r,r)
    self._hud.painter=painter
    self._render_hud(self.viewport_size)
    self._hud.painter=None
    self._renderer.frame_count+=1
    self._renderer.draw_calls=len(faces)+len(lines)+len(points)
    painter.end()
