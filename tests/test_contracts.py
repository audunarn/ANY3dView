from __future__ import annotations

import importlib
import inspect
import sys

import pytest

import any3dview
from any3dview import Pick, Point3D, SectionPlane, ViewerCapabilities, ViewerState


def state(**changes):
    values = dict(
        camera_position=Point3D(3, 4, 5),
        camera_target=Point3D(0, 0, 0),
        camera_world_up=Point3D(0, 0, 1),
        fov=0.7,
        near=0.01,
        far=1000.0,
        section_plane=SectionPlane((1, 0, 0), 2.0),
        background="#ffffff",
    )
    values.update(changes)
    return ViewerState(**values)


def test_shared_pick_and_viewer_state_are_toolkit_neutral_values():
    picked = Pick("face:9", 12, 30, 40, shift=True)
    value = state()

    assert picked.tag == "face:9"
    assert picked.shift and not picked.ctrl
    assert value.section_plane is not None
    assert value.section_plane.contains((3, 0, 0))
    assert value.interaction_profile == "legacy"


def test_expanded_capabilities_are_backward_compatible_defaults():
    value = ViewerCapabilities(dynamic_arrays=True)

    assert value.dynamic_arrays
    assert not value.legacy_primitives
    assert not value.image_capture
    assert not value.stippled_transparency


def test_base_import_does_not_load_optional_gui_modules():
    optional = {"tkinter", "moderngl", "tkinter_gl", "PIL", "anytk3d"}
    before = {name for name in sys.modules if name.split(".")[0] in optional}
    importlib.reload(any3dview)
    after = {name for name in sys.modules if name.split(".")[0] in optional}

    assert after == before


def test_gpu_legacy_public_surface_contains_tk_compatibility_methods():
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    required = {
        "add_faces", "add_line", "add_markers", "add_text", "add_mesh",
        "add_box", "add_sphere", "add_arrow", "add_cylinder",
        "set_thickness_legend", "set_pick_callback", "set_highlight",
        "begin_animation_cache", "capture_animation_frame", "play_animation",
        "capture_image", "export_view_state", "apply_view_state",
        "project_point", "project_points", "unproject_to_plane",
    }
    assert required <= set(vars(Any3DView))
    signature = inspect.signature(Any3DView.add_mesh_arrays)
    assert signature.parameters["point_color"].default == "#2563eb"
    assert signature.parameters["point_size"].default == 6
    legend_signature = inspect.signature(Any3DView.set_thickness_legend)
    assert legend_signature.parameters["font_size"].default == 10


def test_gpu_legend_retains_requested_text_size(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._thickness_legend = None
    monkeypatch.setattr(viewer, "redraw", lambda: None)

    viewer.set_thickness_legend(
        (0.0, 1.0), title="Displacement", width=220, font_size=12
    )

    assert viewer._thickness_legend["width"] == 220
    assert viewer._thickness_legend["font_size"] == 12


def test_gpu_legacy_face_conversion_batches_equal_colours(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    captured = []
    monkeypatch.setattr(
        viewer,
        "add_mesh_arrays",
        lambda mesh, **options: captured.append((mesh, options)),
    )
    polygons = (
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        ((0, 0, 1), (1, 0, 1), (0, 1, 1)),
        ((0, 0, 2), (1, 0, 2), (0, 1, 2)),
    )

    viewer.add_faces(polygons, colors=("red", "blue", "red"), outline="black")

    assert len(captured) == 2
    assert sum(mesh.triangle_count for mesh, _options in captured) == 3
    assert sum(len(mesh.lines) for mesh, _options in captured) == 9


def test_gpu_ring_stiffener_matches_web_and_flange_section_geometry(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._thickness_legend = None
    captured = []
    monkeypatch.setattr(
        viewer,
        "add_faces",
        lambda faces, **options: captured.append((tuple(faces), options)),
    )

    viewer.add_ring_stiffener(
        2.0,
        0.5,
        web_height=0.25,
        web_thickness=0.04,
        flange_width=0.3,
        flange_thickness=0.06,
        segments=8,
    )

    assert [len(faces) for faces, _options in captured] == [24, 8]
    assert [options["layer"] for _faces, options in captured] == [20, 21]
    web_points = [point for face in captured[0][0] for point in face]
    flange_points = [point for face in captured[1][0] for point in face]
    assert {round(point.z, 6) for point in web_points} == {0.48, 0.52}
    assert {round(point.z, 6) for point in flange_points} == {0.35, 0.65}
    assert max((point.x ** 2 + point.y ** 2) ** 0.5 for point in web_points) == pytest.approx(2.25)
    assert max((point.x ** 2 + point.y ** 2) ** 0.5 for point in flange_points) == pytest.approx(2.28)


def test_gpu_back_color_uses_one_two_sided_retained_batch(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    captured = []
    monkeypatch.setattr(
        viewer,
        "add_mesh_arrays",
        lambda mesh, **options: captured.append((mesh, options)),
    )
    viewer.add_polygon(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        color="#ff0000",
        back_color="#0000ff",
    )

    assert len(captured) == 1
    assert captured[0][1]["back_color"] == "#0000ff"


def test_gpu_transparent_cylinder_automatically_retains_back_shell(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    captured = []
    monkeypatch.setattr(
        viewer,
        "add_shape",
        lambda mesh, position=None, **options: captured.append(options),
    )
    viewer.add_cylinder(1.0, 2.0, opacity=0.5, show_backfaces=None)

    assert captured[0]["cull_backface"] is False
    assert captured[0]["two_sided_shell"] is True


def test_gpu_semantic_highlight_does_not_overwrite_application_selection(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import MeshArrays, MeshHandle, PackedOwnerTable, PickBinding
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
    )
    handle = MeshHandle(mesh)
    handle.set_selected_elements((0,))
    calls = []

    class Renderer:
        def set_highlighted_elements(self, value, elements):
            calls.append((value, tuple(elements)))

    viewer = Any3DView.__new__(Any3DView)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(handle): {
            "handle": handle,
            "owners": PackedOwnerTable.from_owners(
                triangles=(PickBinding.one("face:1"),)
            ),
            "owner_resolver": None,
            "tags": frozenset(),
        }
    }
    viewer._highlighted_tags = frozenset()
    viewer._highlight_fill = "#ff8c00"
    viewer._highlight_outline = "#b45309"
    viewer._preselected_key = None
    monkeypatch.setattr(viewer, "redraw", lambda: None)

    viewer.set_highlight(("face:1",))
    viewer.clear_highlight()

    assert handle.selected_elements.tolist() == [0]
    assert calls[-2][1] == (0,)
    assert calls[-1][1] == ()


def test_gpu_large_visible_query_stays_on_cached_integer_pick_path(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview import PickOwner, SelectionConfig, SelectionHit
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig(click_radius_px=9)
    expected = (SelectionHit(PickOwner("element:7"), primitive=7, depth=1.0),)
    calls = []
    monkeypatch.setattr(
        viewer,
        "_gpu_point_hits",
        lambda x, y, selection_filter, radius=0: (
            calls.append((x, y, radius)) or expected
        ),
    )
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)
    monkeypatch.setattr(
        viewer,
        "_projected_selection_index",
        lambda: (_ for _ in ()).throw(AssertionError("CPU index must stay cold")),
    )

    assert viewer.query_point(13, 17) == expected
    assert calls == [(13, 17, 9)]


def test_gpu_large_visible_query_falls_back_if_id_backend_is_unavailable(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview import PickOwner, SelectionConfig, SelectionHit
    from any3dview.gpu import Any3DView

    expected = (SelectionHit(PickOwner("fallback"), primitive=1, depth=1.0),)

    class Index:
        @staticmethod
        def point_hits(*_args, **_kwargs):
            return expected

    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig()
    monkeypatch.setattr(viewer, "_gpu_point_hits", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)
    monkeypatch.setattr(viewer, "_projected_selection_index", lambda: Index())

    assert viewer.query_point(2, 3) == expected


def test_gpu_large_click_expands_stack_on_second_click(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview import (
        PickOwner,
        SelectionConfig,
        SelectionDepth,
        SelectionHit,
        SelectionOperation,
    )
    from any3dview.gpu import Any3DView

    front = SelectionHit(PickOwner("front"), primitive=0, depth=1.0)
    behind = SelectionHit(PickOwner("behind"), primitive=1, depth=2.0, visible=False)
    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig()
    viewer._cycle_candidates = ()
    viewer._cycle_anchor = None
    viewer._cycle_time = 0.0
    viewer._cycle_index = -1
    viewer._selection_callback = None
    viewer._pick_callback = None
    viewer._pick_prefix = ""
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)
    monkeypatch.setattr(
        viewer,
        "query_point",
        lambda *_args, config, **_kwargs: (
            (front, behind) if config.depth is SelectionDepth.THROUGH else (front,)
        ),
    )

    assert viewer._emit_click((20, 20), SelectionOperation.REPLACE) == (front,)
    assert viewer._emit_click((20, 20), SelectionOperation.REPLACE) == (behind,)


def test_gpu_large_hud_selection_does_not_build_projected_index(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    class Hud:
        def begin(self, _viewport):
            pass

        def render(self):
            pass

    viewer = Any3DView.__new__(Any3DView)
    viewer._hud = Hud()
    viewer._world_text = []
    viewer._section_plane = None
    viewer._show_axis_indicator = False
    viewer.show_axis_ruler = False
    viewer._thickness_legend = None
    viewer._highlighted_tags = frozenset(("element:1",))
    viewer._preselected_key = None
    viewer._selection_dragging = False
    viewer._selection_press = None
    viewer._selection_current = None
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1_000_000)
    monkeypatch.setattr(
        viewer,
        "_projected_selection_index",
        lambda: (_ for _ in ()).throw(AssertionError("camera-time CPU projection")),
    )

    viewer._render_hud((800, 600))


def test_gpu_camera_drag_does_not_build_projected_highlight_index(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    class Hud:
        def begin(self, _viewport):
            pass

        def render(self):
            pass

    viewer = Any3DView.__new__(Any3DView)
    viewer._hud = Hud()
    viewer._world_text = []
    viewer._section_plane = None
    viewer._show_axis_indicator = False
    viewer.show_axis_ruler = False
    viewer._thickness_legend = None
    viewer._highlighted_tags = frozenset(("geometry.face:1",))
    viewer._preselected_key = None
    viewer._selection_dragging = False
    viewer._selection_press = None
    viewer._selection_current = None
    viewer._drag = "orbit"
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 100)
    monkeypatch.setattr(
        viewer,
        "_projected_selection_index",
        lambda: (_ for _ in ()).throw(AssertionError("camera-time CPU projection")),
    )

    viewer._render_hud((800, 600))


def test_gpu_wheel_zoom_does_not_build_projected_highlight_index(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview.gpu import Any3DView

    class Hud:
        def begin(self, _viewport):
            pass

        def render(self):
            pass

    viewer = Any3DView.__new__(Any3DView)
    viewer._hud = Hud()
    viewer._world_text = []
    viewer._section_plane = None
    viewer._show_axis_indicator = False
    viewer.show_axis_ruler = False
    viewer._thickness_legend = None
    viewer._highlighted_tags = frozenset(("geometry.face:1",))
    viewer._preselected_key = None
    viewer._selection_dragging = False
    viewer._selection_press = None
    viewer._selection_current = None
    viewer._drag = ""
    viewer._wheel_finish_after_id = "after#zoom"
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 100)
    monkeypatch.setattr(
        viewer,
        "_projected_selection_index",
        lambda: (_ for _ in ()).throw(AssertionError("wheel-time CPU projection")),
    )

    viewer._render_hud((800, 600))


def test_gpu_semantic_masks_keep_line_point_and_preselection_distinct(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import MeshArrays, MeshHandle, PackedOwnerTable, PickBinding
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0]], np.float32),
        np.empty((0, 3), np.uint32),
        lines=np.asarray([[0, 1]], np.uint32),
        point_indices=np.asarray([0, 1], np.uint32),
    )
    handle = MeshHandle(mesh)
    captured = []

    class Renderer:
        def set_semantic_masks(self, _handle, **masks):
            captured.append({key: tuple(value) for key, value in masks.items()})

    viewer = Any3DView.__new__(Any3DView)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(handle): {
            "handle": handle,
            "owners": PackedOwnerTable.from_owners(
                lines=(PickBinding.one("edge:1"),),
                points=(PickBinding.one("node:1"), PickBinding.one("node:2")),
            ),
            "owner_resolver": None,
            "tags": frozenset(),
            "hit_points": {"node:2": np.asarray([1], np.uint32)},
        }
    }
    viewer._highlighted_tags = frozenset(("edge:1",))
    viewer._preselected_key = "node:2"
    monkeypatch.setattr(viewer, "redraw", lambda: None)

    viewer._apply_highlight_masks()

    assert captured[-1]["selected_lines"] == (0,)
    assert captured[-1]["preselected_points"] == (1,)
    assert captured[-1]["selected_points"] == ()
    assert captured[-1]["preselected_lines"] == ()


def test_gpu_persistent_highlight_resolves_complete_owner_not_partial_hit(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import MeshArrays, MeshHandle, PackedOwnerTable, PickBinding
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], np.uint32),
    )
    handle = MeshHandle(mesh)
    captured = []

    class Renderer:
        def set_semantic_masks(self, _handle, **masks):
            captured.append({key: tuple(value) for key, value in masks.items()})

    viewer = Any3DView.__new__(Any3DView)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(handle): {
            "handle": handle,
            "owners": PackedOwnerTable.from_owners(
                triangles=(PickBinding.one("face:1"), PickBinding.one("face:1"))
            ),
            "owner_resolver": None,
            "tags": frozenset(),
            "hit_elements": {"face:1": np.asarray([0], np.uint32)},
        }
    }
    viewer._highlighted_tags = frozenset(("face:1",))
    viewer._preselected_key = None
    monkeypatch.setattr(viewer, "redraw", lambda: None)

    viewer._apply_highlight_masks()

    assert captured[-1]["selected_elements"] == (0, 1)


def test_gpu_visibility_compiles_owner_masks_without_geometry_updates(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import (
        MeshArrays,
        MeshHandle,
        PackedOwnerTable,
        PickBinding,
        SemanticRef,
        VisibilityState,
    )
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0]], np.float32),
        np.empty((0, 3), np.uint32),
        lines=np.asarray([[0, 1]], np.uint32),
        point_indices=np.asarray([0], np.uint32),
    )
    handle = MeshHandle(mesh)
    captured = []

    class Renderer:
        pick_dirty = False

        def set_semantic_masks(self, _handle, **masks):
            captured.append({key: tuple(value) for key, value in masks.items()})

    viewer = Any3DView.__new__(Any3DView)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(handle): {
            "handle": handle,
            "owners": PackedOwnerTable.from_owners(
                lines=(PickBinding.one("edge:1", "edge"),),
                points=(PickBinding.one("node:1", "node"),),
            ),
            "owner_resolver": None,
            "tags": frozenset(),
        }
    }
    viewer._highlighted_tags = frozenset()
    viewer._semantic_selection = ()
    viewer._preselected_key = None
    viewer._visibility_state = VisibilityState(hidden=(
        SemanticRef("application", "edge", "edge:1"),
    ))
    monkeypatch.setattr(viewer, "redraw", lambda: None)
    generations = handle.generations

    viewer._apply_highlight_masks()

    assert captured[-1]["hidden_lines"] == (0,)
    assert captured[-1]["hidden_points"] == ()
    assert handle.generations == generations


def test_gpu_clear_keep_canvas_cancels_redraw_and_resets_transient_selection(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview import SelectionOperation
    from any3dview.gpu import Any3DView

    class Host:
        cancelled = 0
        requested = 0

        def cancel_redraw(self):
            self.cancelled += 1

        def request_redraw(self):
            self.requested += 1

    viewer = Any3DView.__new__(Any3DView)
    viewer._host = Host()
    viewer._closed = False
    viewer._suspend_redraw = False
    viewer._entries = {}
    viewer._world_text = []
    viewer._thickness_legend = {"title": "old"}
    viewer._opaque_occluders = []
    viewer._selection_index = object()
    viewer._preselected_key = "old"
    viewer._hover_key = "old"
    viewer._highlighted_tags = frozenset(("persistent",))
    viewer._cycle_candidates = (object(),)
    viewer._cycle_anchor = (1, 1)
    viewer._cycle_index = 2
    viewer._selection_press = (1, 1)
    viewer._selection_current = (2, 2)
    viewer._selection_points = [(1, 1)]
    viewer._selection_dragging = True
    viewer._selection_press_hit_keys = frozenset(("old",))
    viewer._selection_operation = SelectionOperation.REMOVE
    monkeypatch.setattr(viewer, "stop_animation", lambda: None)

    viewer.clear(keep_canvas=True)

    assert viewer._host.cancelled == 1
    assert viewer._host.requested == 0
    assert viewer._preselected_key is None
    assert viewer._hover_key is None
    assert viewer._cycle_candidates == ()
    assert viewer._selection_press is None
    assert viewer._highlighted_tags == frozenset(("persistent",))


def test_gpu_legacy_pick_preserves_combined_modifier_flags(monkeypatch):
    pytest.importorskip("moderngl")
    from any3dview import (
        PickOwner,
        SelectionConfig,
        SelectionHit,
        SelectionOperation,
    )
    from any3dview.gpu import Any3DView

    hit = SelectionHit(PickOwner("face:1"), primitive=0, depth=1.0, item=7)
    received = []
    viewer = Any3DView.__new__(Any3DView)
    viewer._selection_config = SelectionConfig()
    viewer._cycle_candidates = ()
    viewer._cycle_anchor = None
    viewer._cycle_time = 0.0
    viewer._cycle_index = -1
    viewer._selection_callback = None
    viewer._pick_callback = received.append
    viewer._pick_prefix = ""
    viewer._selection_modifiers = (True, True, False)
    monkeypatch.setattr(viewer, "query_point", lambda *_args, **_kwargs: (hit,))
    monkeypatch.setattr(viewer, "_display_primitive_count", lambda _limit: 1)

    viewer._emit_click((8, 9), SelectionOperation.TOGGLE)

    assert received[0].shift is True
    assert received[0].ctrl is True
    assert received[0].alt is False


def test_gpu_fit_to_scene_accounts_for_portrait_aspect_and_empty_redraw(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import Camera3D
    from any3dview.gpu import Any3DView

    viewer = Any3DView.__new__(Any3DView)
    viewer.camera = Camera3D()
    viewer._selection_index = None
    viewer._renderer = type("Renderer", (), {"pick_dirty": False})()
    viewer._host = type("Host", (), {"framebuffer_size": lambda self: (200, 800)})()
    viewer.width, viewer.height = 200, 800
    redraws = []
    monkeypatch.setattr(viewer, "redraw", lambda: redraws.append(True))
    monkeypatch.setattr(
        viewer,
        "_scene_bounds",
        lambda: (np.asarray((-5.0, -1.0, -1.0)), np.asarray((5.0, 1.0, 1.0))),
    )

    viewer.fit_to_scene()
    portrait_distance = viewer.camera.distance
    assert portrait_distance > 30.0

    monkeypatch.setattr(viewer, "_scene_bounds", lambda: None)
    viewer.fit_to_scene(redraw=True)
    assert len(redraws) == 2


def test_gpu_hover_preselection_uses_hit_primitive_without_owner_table_scan(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import MeshArrays, MeshHandle, PackedOwnerTable, PickBinding
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray([[0, 0, 0]], np.float32),
        np.empty((0, 3), np.uint32),
        point_indices=np.asarray([0], np.uint32),
    )
    hit_handle, other_handle = MeshHandle(mesh), MeshHandle(mesh)
    table = PackedOwnerTable.from_owners(points=(PickBinding.one("node:1"),))
    calls = []

    class Renderer:
        def set_semantic_masks(self, handle, **masks):
            calls.append((handle, tuple(masks["preselected_points"])))

    viewer = Any3DView.__new__(Any3DView)
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(hit_handle): {
            "handle": hit_handle,
            "owners": table,
            "owner_resolver": None,
            "tags": frozenset(),
            "hit_points": {"node:1": np.asarray([0], np.uint32)},
        },
        id(other_handle): {
            "handle": other_handle,
            "owners": table,
            "owner_resolver": None,
            "tags": frozenset(),
        },
    }
    viewer._highlighted_tags = frozenset()
    viewer._preselected_key = "node:1"
    viewer._preselection_from_hit = True
    monkeypatch.setattr(viewer, "redraw", lambda: None)
    monkeypatch.setattr(
        viewer,
        "_semantic_primitives_for_key",
        lambda *_args: (_ for _ in ()).throw(AssertionError("hover owner scan")),
    )

    viewer._apply_highlight_masks()

    assert calls == [(hit_handle, (0,)), (other_handle, ())]


def test_gpu_chunk_local_owners_drive_fast_pick_and_distinct_masks(monkeypatch):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import (
        Camera3D,
        MeshArrays,
        MeshHandle,
        PackedOwnerTable,
        PickBinding,
        SelectionFilter,
    )
    from any3dview.gpu import Any3DView

    empty = MeshArrays(
        np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    )
    chunk = MeshArrays(
        np.asarray(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]],
            np.float32,
        ),
        np.asarray([[0, 1, 2]], np.uint32),
        lines=np.asarray([[3, 4]], np.uint32),
        point_indices=np.asarray([2], np.uint32),
    )
    owners = PackedOwnerTable.from_owners(
        triangles=(PickBinding.one("chunk:face", "face"),),
        lines=(PickBinding.one("chunk:line", "line"),),
        points=(PickBinding.one("chunk:point", "point"),),
    )
    handle = MeshHandle(empty)
    handle.add_chunk("local", chunk, owners=owners)
    calls = []

    class Host:
        @staticmethod
        def make_current():
            return None

        @staticmethod
        def framebuffer_size():
            return 64, 64

    class Renderer:
        pick_dirty = False

        @staticmethod
        def pick(*_args, **_kwargs):
            return handle, "triangle", 0

        @staticmethod
        def pick_detail(*_args, **_kwargs):
            return handle, "triangle", 0, "local"

        @staticmethod
        def set_semantic_masks(_handle, **_masks):
            return None

        @staticmethod
        def set_chunk_pickable(_handle, chunk_id, pickable=True):
            calls.append(("pickable", chunk_id, pickable))

        @staticmethod
        def set_chunk_semantic_masks(_handle, chunk_id, **masks):
            calls.append(
                (
                    "masks",
                    chunk_id,
                    {key: tuple(value) for key, value in masks.items()},
                )
            )

    viewer = Any3DView.__new__(Any3DView)
    viewer.camera = Camera3D()
    viewer._host = Host()
    viewer._renderer = Renderer()
    viewer._animation_frame_active = False
    viewer._entries = {
        id(handle): {
            "handle": handle,
            "owners": None,
            "owner_resolver": None,
            "appearance": {"pickable": True},
            "tags": frozenset(),
            "item": 41,
            "chunk_semantics": {},
        }
    }
    viewer._section_plane = None
    viewer.show_mesh_lines = True
    viewer._occlude_lines = True
    viewer._highlighted_tags = frozenset(("chunk:line",))
    viewer._preselected_key = "chunk:point"
    viewer._preselection_from_hit = False
    monkeypatch.setattr(viewer, "redraw", lambda: None)

    hits = viewer._gpu_point_hits(32, 32, SelectionFilter())
    assert [hit.key for hit in hits] == ["chunk:face"]
    assert hits[0].item == 41
    assert tuple(
        viewer._entries[id(handle)]["chunk_semantics"]["local"]
        ["hit_elements"]["chunk:face"]
    ) == (0,)

    viewer._apply_highlight_masks()
    assert ("pickable", "local", True) in calls
    chunk_masks = next(value[2] for value in calls if value[:2] == ("masks", "local"))
    assert chunk_masks["selected_lines"] == (0,)
    assert chunk_masks["preselected_points"] == (0,)
    assert chunk_masks["selected_elements"] == ()


def test_gpu_unowned_chunk_uses_stable_handle_tag_without_synthetic_owner(
    monkeypatch,
):
    pytest.importorskip("moderngl")
    import numpy as np
    from any3dview import MeshArrays, MeshHandle
    from any3dview.gpu import Any3DView

    mesh = MeshArrays(
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.asarray([[0, 1, 2]], np.uint32),
    )
    handle = MeshHandle(mesh)
    handle.add_chunk("tagged", mesh)
    viewer = Any3DView.__new__(Any3DView)
    entry = {"handle": handle, "owners": None, "tags": frozenset(("part:7",))}

    binding = viewer._selection_binding(
        handle, entry, "triangle", 0, chunk_id="tagged"
    )
    assert binding is not None
    assert tuple(owner.key for owner in binding.owners) == ("part:7",)

    entry["tags"] = frozenset()
    assert viewer._selection_binding(
        handle, entry, "triangle", 0, chunk_id="tagged"
    ) is None
