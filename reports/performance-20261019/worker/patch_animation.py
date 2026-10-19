"""One-shot patcher: rewrite _show_animation_frame with handle reuse."""
import io

path = r'src/any3dview/retained_viewer.py'
data = open(path, 'rb').read().decode('utf-8')

start = data.index('    def _show_animation_frame(self, index: int) -> None:')
end = data.index('    def play_animation(self, fps: float = 30')
old_method = data[start:end]
assert old_method.count('def ') == 1, old_method[:200]
assert old_method.rstrip().endswith('self._apply_highlight_masks()'), old_method[-100:]

new_method = '''    def _animation_entry(self, handle, item):
        """Per-frame display entry for one animation mesh handle."""

        return {
            "handle": handle,
            "owners": item["owners"],
            "owner_resolver": item["owner_resolver"],
            "appearance": item["appearance"],
            "outline_groups": item.get("outline_groups"),
            "outline_topology": handle.mesh.triangles,
            "layer": item["layer"],
            "tags": item["tags"],
            "item": item["item"],
            "face_colors": item["appearance"].get("face_colors"),
            "semantic_elements": {
                key: np.asarray(value).copy()
                for key, value in item["semantic_elements"].items()
            },
            "hit_elements": {
                key: np.asarray(value).copy()
                for key, value in item["hit_elements"].items()
            },
            "semantic_lines": {
                key: np.asarray(value).copy()
                for key, value in item["semantic_lines"].items()
            },
            "hit_lines": {
                key: np.asarray(value).copy()
                for key, value in item["hit_lines"].items()
            },
            "semantic_points": {
                key: np.asarray(value).copy()
                for key, value in item["semantic_points"].items()
            },
            "hit_points": {
                key: np.asarray(value).copy()
                for key, value in item["hit_points"].items()
            },
            "chunk_semantics": self._copy_chunk_semantics(
                item.get("chunk_semantics", {})
            ),
        }

    def _animation_mesh_compatible(self, current, replacement) -> bool:
        """Exact topology guard for reusing a handle with new frame arrays."""

        for field in _ANIMATION_TOPOLOGY_FIELDS:
            if not _arrays_equal(getattr(current, field), getattr(replacement, field)):
                return False
        if np.asarray(current.positions).shape != np.asarray(replacement.positions).shape:
            return False
        for field in _ANIMATION_VALUE_FIELDS:
            if not _value_shapes_compatible(
                getattr(current, field), getattr(replacement, field)
            ):
                return False
        return True

    def _animation_item_compatible(self, handle, item) -> bool:
        if not self._animation_mesh_compatible(handle.mesh, item["mesh"]):
            return False
        chunks = item["chunks"]
        existing = handle.chunks
        if len(chunks) != len(existing):
            return False
        for (chunk_id, chunk, _owners, _resolver), (current_id, current) in zip(chunks, existing):
            if chunk_id != current_id or not self._animation_mesh_compatible(current, chunk):
                return False
        entry = self._animation_entries.get(id(handle))
        if entry is None:
            return False
        # Static appearance must match exactly; scalar range and palette are
        # dynamic and flow through the handle's appearance mutator instead.
        appearance = item["appearance"]
        current_appearance = entry["appearance"]
        return all(
            appearance.get(key) == current_appearance.get(key)
            for key in _ANIMATION_STATIC_APPEARANCE_KEYS
        )

    def _apply_animation_item(self, handle, item) -> None:
        """Push one frame's dynamic arrays through the handle mutators."""

        mesh = item["mesh"]
        current = handle.mesh
        if not _arrays_equal(current.positions, mesh.positions):
            handle.update_positions(mesh.positions)
        if not _arrays_equal(current.displacements, mesh.displacements):
            handle.update_displacements(mesh.displacements)
        if (not _arrays_equal(current.node_scalars, mesh.node_scalars)
                or not _arrays_equal(current.element_scalars, mesh.element_scalars)):
            handle.update_scalar_fields(
                node_scalars=mesh.node_scalars,
                element_scalars=mesh.element_scalars,
            )
        if not _arrays_equal(current.active_elements, mesh.active_elements):
            handle.set_active_elements(mesh.active_elements)
        for chunk_id, _chunk, owners, resolver in item["chunks"]:
            if handle.chunk_ownership(chunk_id) != (owners, resolver):
                handle.set_chunk_ownership(chunk_id, owners, owner_resolver=resolver)
        if not _arrays_equal(handle.transform, item["transform"]):
            handle.set_transform(item["transform"])
        handle.set_deformation_scale(item["deformation_scale"])
        if not _arrays_equal(handle.selected_elements, item["selected"]):
            handle.set_selected_elements(item["selected"])
        handle.set_visible(item["visible"])
        handle.update_appearance(
            scalar_range=item["appearance"].get("scalar_range"),
            colormap=item["appearance"].get("colormap"),
        )

    def _update_animation_handles(self, frame) -> bool:
        """Reuse the live animation handles when the frame is compatible.

        Compatibility is verified for every item before any mutation, so an
        incompatible frame leaves the previous frame's scene untouched for
        the full rebuild path.  Live entries stay detached from the
        renderer exactly as the rebuild path detaches them.
        """

        items = frame["meshes"]
        handles = self._animation_handles
        if not handles or len(items) != len(handles):
            return False
        for handle, item in zip(handles, items):
            if not self._animation_item_compatible(handle, item):
                return False
        for entry in self._entries.values():
            self._renderer.remove_mesh(entry["handle"])
        entries = {}
        for handle, item in zip(handles, items):
            self._apply_animation_item(handle, item)
            entries[id(handle)] = self._animation_entry(handle, item)
        self._animation_entries = entries
        return True

    def _finish_animation_frame(self, frame) -> None:
        self._world_text = [dict(value) for value in frame["world_text"]]
        self._thickness_legend = (
            None if frame["legend"] is None else dict(frame["legend"])
        )
        self._highlighted_tags = frame["highlighted_tags"]
        self._preselected_key = frame["preselected_key"]
        self._preselection_from_hit = frame["preselection_from_hit"]
        self._highlight_fill = frame["highlight_fill"]
        self._highlight_outline = frame["highlight_outline"]
        self._selection_index = None
        self._selection_index_key = None
        self._apply_highlight_masks()

    def _show_animation_frame(self, index: int) -> None:
        frame = self._animation_cache[index]
        if self._animation_frame_active and self._animation_handles:
            try:
                if self._update_animation_handles(frame):
                    self._finish_animation_frame(frame)
                    return
            except Exception:
                # A failed in-place mutation must not leave a partially
                # updated live scene; the full rebuild below replaces the
                # handles wholesale and restores a valid frame.
                pass
        for entry in self._entries.values():
            self._renderer.remove_mesh(entry["handle"])
        for handle in self._animation_handles:
            self._renderer.remove_mesh(handle)
        self._animation_handles.clear()
        self._animation_entries.clear()
        self._animation_frame_active = True
        for item in frame["meshes"]:
            handle = MeshHandle(item["mesh"],scalar_range=item["appearance"].get("scalar_range"),colormap=item["appearance"].get("colormap"))
            for key, mesh, owners, resolver in item["chunks"]:
                handle.add_chunk(
                    key,
                    mesh,
                    owners=owners,
                    owner_resolver=resolver,
                )
            handle.set_transform(item["transform"])
            handle.set_deformation_scale(item["deformation_scale"])
            handle.set_selected_elements(item["selected"])
            handle.set_visible(item["visible"])
            self._renderer.add_mesh(handle, **{
                key: value for key, value in item["appearance"].items()
                if key in _ANIMATION_RENDERER_APPEARANCE_KEYS
            })
            self._animation_handles.append(handle)
            self._animation_entries[id(handle)] = self._animation_entry(handle, item)
        self._finish_animation_frame(frame)


'''

data = data[:start] + new_method + data[end:]
open(path, 'wb').write(data.encode('utf-8'))
print('ok')
