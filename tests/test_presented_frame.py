"""Frame notifications belong to successful host swaps on the viewer thread."""

from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import SimpleNamespace

import pytest

pytest.importorskip("moderngl")
from any3dview.gpu.widget import Any3DView


@pytest.fixture
def viewer():
    result = Any3DView.__new__(Any3DView)
    result._closed = False
    result._presentation_owner = get_ident()
    result._presented_frame_sequence = 0
    result._frame_presented_listeners = []
    events = []
    result._host = SimpleNamespace(make_current=lambda: events.append("current"), framebuffer_size=lambda: (320, 240), swap_buffers=lambda: events.append("swap"))
    result._renderer = SimpleNamespace(render=lambda *args, **kwargs: events.append("render"))
    result._sync_pixel_ratio = lambda: None
    result._render_hud = lambda _: events.append("hud")
    result.camera = object()
    result.bg = "white"
    result._section_plane = None
    result._light = None
    result._shading_enabled = True
    result._occlude_lines = True
    result.show_mesh_lines = True
    result._highlight_fill = "orange"
    errors = []
    result._root = lambda: SimpleNamespace(report_callback_exception=lambda *error: errors.append(error))
    return result, events, errors


def test_notification_after_swap_and_independent_unsubscription(viewer):
    widget, events, _ = viewer
    frames = []

    def callback(frame):
        assert events[-1] == "swap"
        assert get_ident() == widget._presentation_owner
        frames.append(frame)

    unsubscribe = widget.add_frame_presented_listener(callback)
    widget.add_frame_presented_listener(callback)
    widget._draw_now()
    assert [frame.sequence for frame in frames] == [1, 1]
    assert frames[0].presented_ns > 0
    unsubscribe()
    unsubscribe()
    widget._draw_now()
    assert [frame.sequence for frame in frames] == [1, 1, 2]
    assert widget.presented_frame_sequence == 2
    assert (widget.width, widget.height) == (320, 240)


@pytest.mark.parametrize("failure_point", ["render", "swap"])
def test_failed_frame_never_publishes(viewer, failure_point):
    widget, _, _ = viewer
    frames = []
    widget.add_frame_presented_listener(frames.append)

    def fail(*args, **kwargs):
        raise RuntimeError("GPU failure")

    if failure_point == "render":
        widget._renderer.render = fail
    else:
        widget._host.swap_buffers = fail
    with pytest.raises(RuntimeError, match="GPU failure"):
        widget._draw_now()
    assert frames == []
    assert widget.presented_frame_sequence == 0


def test_listener_failure_reported_without_losing_other_listeners(viewer):
    widget, _, errors = viewer
    frames = []

    def fail(_):
        raise ValueError("listener failure")

    widget.add_frame_presented_listener(fail)
    widget.add_frame_presented_listener(frames.append)
    widget._draw_now()
    assert frames[0].sequence == 1
    assert errors[0][0] is ValueError


def test_remove_during_notification_and_closed_frame(viewer):
    widget, _, _ = viewer
    frames = []
    widget.add_frame_presented_listener(lambda _: unsubscribe())
    unsubscribe = widget.add_frame_presented_listener(frames.append)
    widget._draw_now()
    assert frames == []
    widget._closed = True
    widget._draw_now()
    assert widget.presented_frame_sequence == 1
    with pytest.raises(RuntimeError, match="closed"):
        widget.add_frame_presented_listener(frames.append)


def test_presentation_registration_and_draw_require_owner(viewer):
    widget, _, _ = viewer
    unsubscribe = widget.add_frame_presented_listener(lambda _: None)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for operation in (widget._draw_now, lambda: widget.add_frame_presented_listener(lambda _: None), unsubscribe):
            with pytest.raises(RuntimeError, match="owner thread"):
                pool.submit(operation).result()
    assert widget.presented_frame_sequence == 0
