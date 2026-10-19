"""qt.py patch: GPU hover coalescing, queued update notification, software prep cache."""
import io

PATH = r"C:\Github\ANY3dView\src\any3dview\qt.py"
CRLF = "\r\n"

def read():
    with io.open(PATH, "r", newline="") as handle:
        return handle.read()

def write(text):
    with io.open(PATH, "w", newline="") as handle:
        handle.write(text)

def lines(*rows):
    return CRLF.join(rows)

text = read()

def replace_once(old, new):
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"MATCH FAILURE ({count}) for: {old[:100]!r}")
    text = text.replace(old, new)

# 1. GPU hover: coalesce unpressed moves to one latest request per pass.
replace_once(
    lines(
        "        elif self.owner.backend_name == \"software\":",
        "            self.owner._queue_hover(e)",
        "        else:",
        "            self.owner._hover_select(e)",
    ),
    lines(
        "        elif self.owner.backend_name == \"software\":",
        "            self.owner._queue_hover(e)",
        "        else:",
        "            # Hover is optional feedback: keep the latest event and",
        "            # evaluate it once per event-loop pass instead of running",
        "            # a synchronous GPU pick per move. Exact clicks bypass",
        "            # this path and stale requests are canceled below.",
        "            self.owner._queue_hover(e, 0)",
    ),
)

replace_once(
    lines(
        "    def _queue_hover(self, event):",
        "        if self._closed:",
        "            return",
    ),
    lines(
        "    def _queue_hover(self, event, interval=120):",
        "        if self._closed:",
        "            return",
    ),
)
replace_once(
    "        self._hover_timer.start(120)\r\n",
    "        self._hover_timer.start(interval)\r\n",
)

# 2. Queued scheduler notification replaces the idle 2 ms poll.
replace_once(
    lines(
        "    def _start_update_pump(self):",
        "        if self._closed or self._update_poll_id is not None:",
        "            return",
        "        self._poll_updates()",
        "        if self._closed:",
        "            return",
        "        # Reuse one Qt timer. The shared Tk implementation uses after(), which",
        "        # otherwise creates/deletes a QObject on every 2 ms poll in this host.",
        "        timer = QTimer(self)",
        "        self._update_poll_id = timer",
        "        self._timers.add(timer)",
        "        timer.timeout.connect(self._poll_updates)",
        "        timer.start(2)",
    ),
    lines(
        "    def _start_update_pump(self):",
        "        if self._closed or self._update_poll_id is not None:",
        "            return",
        "        # Drain work that arrived before a wakeup existed; a callback",
        "        # failure here escapes the constructor with no notifier armed.",
        "        self._poll_updates()",
        "        if self._closed:",
        "            return",
        "        notifier = _UpdateNotifier(self, self)",
        "        self._update_notifier = notifier",
        "        self._update_poll_id = notifier",
        "        # The scheduler invokes this hook from any submitting thread;",
        "        # postEvent is Qt's thread-safe owner-thread handoff, so no",
        "        # idle polling timer is needed.",
        "        self._update_scheduler.set_notifier(",
        "            lambda: QCoreApplication.postEvent(notifier, _UpdateEvent()))",
        "        # A submission racing setup gets no wakeup; drain once more so",
        "        # it cannot wait for the next submit.",
        "        self._poll_updates()",
    ),
)

# 3. Constructor state for the software preparation cache and notifier.
replace_once(
    lines(
        "        self._hover_timer = None",
        "        self._pending_hover = None",
        "        self._host = self",
    ),
    lines(
        "        self._hover_timer = None",
        "        self._pending_hover = None",
        "        self._software_prep = {}",
        "        self._update_notifier = None",
        "        self._host = self",
    ),
)

# 4. Replace _paint_software with the retained-preparation version.
start = text.index("    def _paint_software(self, painter):")
end = text.index("class QtViewerHostAdapter:")
old_block = text[start:end]
assert "_renderer.frame_count+=1" in old_block, "unexpected block"

new_block = lines(
    "    def _software_prep_signature(self, entry, handle):",
    "        appearance = entry[\"appearance\"]",
    "        return (",
    "            handle.generations,",
    "            getattr(self._renderer, \"mask_revision\", 0),",
    "            None if self._section_plane is None else self._section_plane.key,",
    "            self._highlight_fill,",
    "            handle.appearance.scalar_range,",
    "            id(handle.appearance.colormap),",
    "            appearance.get(\"color\"),",
    "            appearance.get(\"face_colors\"),",
    "            appearance.get(\"invalid_color\"),",
    "        )",
    "",
    "    def _software_chunk_prep(self, entry, handle, chunk_id, mesh, section):",
    "        \"\"\"Build or reuse the camera-invariant preparation for one chunk.",
    "",
    "        The retained record is invalidated by its signature (handle",
    "        generations, semantic-mask revision, section plane, highlight",
    "        fill and static color style); projection and submission stay",
    "        per frame, so cached and direct paints give identical pixels.",
    "        \"\"\"",
    "        signature = self._software_prep_signature(entry, handle)",
    "        key = (id(handle), chunk_id)",
    "        prep = self._software_prep.get(key)",
    "        if prep is not None and prep.handle is handle and prep.signature == signature:",
    "            return prep",
    "        appearance = dict(entry[\"appearance\"],scalar_range=handle.appearance.scalar_range,colormap=handle.appearance.colormap)",
    "        positions = mesh.positions",
    "        if mesh.displacements is not None:",
    "            positions = positions + handle.deformation_scale*mesh.displacements",
    "        world = positions @ handle.transform[:3,:3].T + handle.transform[:3,3]",
    "        mask = self._renderer.masks.get(id(handle), {}) if chunk_id is None else self._renderer.chunk_masks.get((id(handle),chunk_id), {})",
    "        hidden = set(mask.get(\"hidden_elements\", ()))",
    "        highlighted = set(mask.get(\"selected_elements\", ())) | set(handle.selected_elements)",
    "        preselected = set(mask.get(\"preselected_elements\", ()))",
    "        colors = appearance.get(\"face_colors\")",
    "        scalars = mesh.element_scalars if mesh.element_scalars is not None else mesh.node_scalars",
    "        scalar_limits = None",
    "        if scalars is not None and not colors:",
    "            finite = scalars[np.isfinite(scalars)]",
    "            scalar_limits = (appearance.get(\"scalar_range\") or (float(finite.min()),float(finite.max()))) if len(finite) else (0.,1.)",
    "        triangles, triangle_colors = [], []",
    "        section_polygons = [] if section is not None else None",
    "        for index, triangle in enumerate(mesh.triangles):",
    "            element = int(mesh.triangle_to_element[index]) if mesh.triangle_to_element is not None else index",
    "            if mesh.active_elements is not None and not mesh.active_elements[element]:",
    "                continue",
    "            if element in hidden:",
    "                continue",
    "            if section is not None:",
    "                section_polygons.append(np.array([p.to_tuple() for p in section.clip_polygon(world[triangle])]))",
    "            color = self._highlight_fill if element in highlighted else (colors[index if len(colors)==mesh.triangle_count else element] if colors else appearance[\"color\"])",
    "            if scalars is not None and not colors and element not in highlighted:",
    "                value=float(scalars[element]) if mesh.element_scalars is not None else float(np.mean(scalars[triangle]))",
    "                color=scalar_color(value,*scalar_limits,appearance.get('colormap')) if np.isfinite(value) else appearance[\"invalid_color\"]",
    "            if element in preselected and element not in highlighted:color=\"#ffd166\"",
    "            triangles.append(index)",
    "            triangle_colors.append(color)",
    "        line_segments, line_selected = [], []",
    "        if mesh.lines is not None and not appearance.get(\"mesh_lines\"):",
    "            hidden_lines=set(mask.get(\"hidden_lines\",()))",
    "            selected_lines=set(mask.get(\"selected_lines\",()))",
    "            for index,(a,b) in enumerate(mesh.lines):",
    "                if index in hidden_lines:continue",
    "                segment=(Point3D(*world[a]),Point3D(*world[b]))",
    "                if self._section_plane is not None:segment=self._section_plane.clip_segment(*segment)",
    "                if segment is None:continue",
    "                line_segments.append((segment[0].to_tuple(), segment[1].to_tuple()))",
    "                line_selected.append(index in selected_lines)",
    "        points, point_selected = [], []",
    "        if mesh.point_indices is not None:",
    "            hidden_points=set(mask.get(\"hidden_points\",()))",
    "            selected_points=set(mask.get(\"selected_points\",()))",
    "            for index,i in enumerate(mesh.point_indices):",
    "                if index in hidden_points:continue",
    "                if self._section_plane is not None and not self._section_plane.contains(world[i]):continue",
    "                points.append(int(i))",
    "                point_selected.append(index in selected_points)",
    "        prep = _SoftwareChunkPrep(handle, signature, world, triangles, triangle_colors,",
    "                                   section_polygons, line_segments, line_selected,",
    "                                   points, point_selected)",
    "        self._software_prep[key] = prep",
    "        return prep",
    "",
    "    def _paint_software(self, painter):",
    "        painter.setRenderHint(QPainter.Antialiasing)",
    "        ratio = self.surface.devicePixelRatioF()",
    "        painter.scale(1/ratio, 1/ratio)",
    "        width, height = self.viewport_size",
    "        painter.fillRect(0, 0, width, height, QColor(self.bg))",
    "        # One camera basis/viewport/scale evaluation serves every projection",
    "        # in the frame; the retained preparation below removes the",
    "        # camera-invariant per-primitive work from repeated frames.",
    "        frame = _FrameProjection(self)",
    "        section = self._section_plane if self._section_plane is not None and self._section_plane.enabled else None",
    "        faces, lines, points, outlines = [], [], [], []",
    "        for entry in self._display_entries().values():",
    "            handle = entry[\"handle\"]",
    "            if handle.removed or not handle.visible:",
    "                continue",
    "            appearance = dict(entry[\"appearance\"],scalar_range=handle.appearance.scalar_range,colormap=handle.appearance.colormap)",
    "            if appearance.get(\"depth_only\"):",
    "                continue",
    "            for chunk_id, mesh in [(None, handle.mesh), *handle.chunks]:",
    "                prep = self._software_chunk_prep(entry, handle, chunk_id, mesh, section)",
    "                world = prep.world",
    "                screen_all, visible_all = self._project_world_points(world, frame)",
    "                triangle_visible = None",
    "                if section is None and len(mesh.triangles):",
    "                    triangle_visible = visible_all[np.asarray(mesh.triangles)].all(axis=1)",
    "                visible_triangles = []",
    "                for position, index in enumerate(prep.triangles):",
    "                    if section is not None:",
    "                        verts = prep.section_polygons[position]",
    "                        screen, visible = self._project_world_points(verts, frame)",
    "                        if len(screen)<3 or not visible.all():",
    "                            continue",
    "                    else:",
    "                        if not triangle_visible[index]:",
    "                            continue",
    "                        screen = screen_all[mesh.triangles[index]]",
    "                    visible_triangles.append(index)",
    "                    faces.append((sum(p[2] for p in screen)/len(screen), screen, prep.colors[position], appearance))",
    "                if self.show_mesh_lines and (appearance.get(\"mesh_lines\") or entry.get(\"outline_groups\") is None):",
    "                    for screen in self._surface_outline_segments(entry, chunk_id, mesh, visible_triangles, world, frame):",
    "                        outlines.append((sum(point[2] for point in screen) / len(screen), screen, appearance))",
    "                for segment, selected in zip(prep.line_segments, prep.line_selected):",
    "                    a,b=(self._project_point_with_frame(Point3D(*p), frame) for p in segment)",
    "                    if a is not None and b is not None:",
    "                        style=dict(appearance)",
    "                        if selected:style[\"line_color\"]=self._highlight_fill",
    "                        lines.append((a,b,style))",
    "                for node, selected in zip(prep.points, prep.point_selected):",
    "                    if not visible_all[node]:continue",
    "                    style=dict(appearance)",
    "                    if selected:style[\"point_color\"]=self._highlight_fill",
    "                    points.append((tuple(screen_all[node]),style))",
    "        surfaces = [(depth, 0, screen, color, appearance) for depth, screen, color, appearance in faces]",
    "        surfaces.extend((depth, 1, screen, None, appearance) for depth, screen, appearance in outlines)",
    "        # Keep the software painter's depth ordering: nearer fills must cover",
    "        # more distant boundaries. At equal depth, fill before its own outline.",
    "        for _, kind, screen, color, appearance in sorted(surfaces, key=lambda surface: (-float(f\"{surface[0]:.12g}\"), surface[1])):",
    "            polygon = QPolygonF([QPointF(*point[:2]) for point in screen])",
    "            if kind == 0:",
    "                # Antialiasing each tessellation triangle independently leaves",
    "                # pale seams where their partially covered edge pixels meet.",
    "                painter.setRenderHint(QPainter.Antialiasing, False)",
    "                fill=QColor(color); fill.setAlphaF(appearance[\"opacity\"])",
    "                painter.setBrush(fill)",
    "                painter.setPen(Qt.NoPen)",
    "                painter.drawPolygon(polygon)",
    "            else:",
    "                painter.setRenderHint(QPainter.Antialiasing, True)",
    "                painter.setBrush(Qt.NoBrush)",
    "                painter.setPen(QPen(QColor(appearance[\"line_color\"]), appearance[\"line_width\"]*self.pixel_ratio))",
    "                painter.drawPolyline(polygon)",
    "        painter.setRenderHint(QPainter.Antialiasing, True)",
    "        for a,b,appearance in lines:",
    "            painter.setPen(QPen(QColor(appearance[\"line_color\"]),appearance[\"line_width\"]*self.pixel_ratio))",
    "            painter.drawLine(QPointF(*a[:2]),QPointF(*b[:2]))",
    "        for p,appearance in points:",
    "            painter.setBrush(QColor(appearance[\"point_color\"]))",
    "            painter.setPen(Qt.NoPen)",
    "            r=appearance[\"point_size\"]*self.pixel_ratio/2",
    "            painter.drawEllipse(QPointF(*p[:2]),r,r)",
    "        self._hud.painter=painter",
    "        self._render_hud(self.viewport_size)",
    "        self._hud.painter=None",
    "        self._renderer.frame_count+=1",
    "        self._renderer.draw_calls=len(faces)+len(lines)+len(points)",
    "        painter.end()",
    "",
    "",
)

text = text[:start] + new_block + text[end:]
write(text)
print("qt.py patched: hover, pump, prep cache")
