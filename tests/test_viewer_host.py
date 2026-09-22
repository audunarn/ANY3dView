from __future__ import annotations

import sys

import pytest

from any3dview import ViewerHostAdapter, create_viewer


class _HeadlessHost:
    toolkit_name = "headless-test"

    def __init__(self) -> None:
        self.calls = []

    def create_viewer(self, parent, *, backend, **options):
        result = object()
        self.calls.append((parent, backend, options, result))
        return result


def test_explicit_host_creates_viewer_without_importing_tk() -> None:
    before = set(sys.modules)
    host = _HeadlessHost()

    viewer = create_viewer("parent", backend="AUTO", host=host, width=640)

    assert isinstance(host, ViewerHostAdapter)
    assert host.calls == [("parent", "auto", {"width": 640}, viewer)]
    assert "tkinter" not in set(sys.modules).difference(before)


def test_invalid_host_fails_before_widget_creation() -> None:
    with pytest.raises(TypeError, match="ViewerHostAdapter"):
        create_viewer(None, host=object())


def test_unknown_backend_fails_before_calling_host() -> None:
    host = _HeadlessHost()
    with pytest.raises(ValueError, match="backend"):
        create_viewer(None, backend="vulkan", host=host)
    assert host.calls == []
