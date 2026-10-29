"""Native canvases must retire from rendercanvas before Qt deletes them."""
import os
import pytest

if os.environ.get('ANY3DVIEW_TEST_WEBGPU')!='1':
    pytest.skip('enable WebGPU checks explicitly',allow_module_level=True)

from PySide6.QtCore import QCoreApplication,QEvent
from PySide6.QtWidgets import QApplication
from any3dview.qt_webgpu import QtWebGPUViewer


@pytest.mark.parametrize('route',['replace','destroy'])
def test_native_canvas_closed_before_qt_deletion(route):
    from rendercanvas.qt import loop
    from shiboken6 import isValid
    app=QApplication.instance() or QApplication([])
    viewer=QtWebGPUViewer(width=160,height=120)
    try:
        # Keep the widget unexposed: this checks lifecycle without native swaps.
        viewer.set_present_method('native')
        surface=viewer.surface
        assert not surface.get_closed()
        assert surface in loop.get_canvases()
        if route=='replace':viewer.set_present_method('bitmap')
        else:viewer.destroy()
        assert surface.get_closed()
        assert surface not in loop.get_canvases(close_closed=True)
        surface.release()  # Cleanup is idempotent.
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
        assert not isValid(surface)
        assert surface not in loop.get_canvases(close_closed=True)
        surface.close()  # Same operation rendercanvas performs during shutdown.
    finally:
        viewer.destroy()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
        app.processEvents()


def test_failed_native_initialization_closes_registered_canvas(monkeypatch):
    from rendercanvas.qt import loop
    from rendercanvas.contexts.wgpucontext import WgpuContextToScreen
    from any3dview.errors import GPUUnavailableError
    app=QApplication.instance() or QApplication([])
    viewer=QtWebGPUViewer(width=160,height=120)
    original=viewer.surface
    before=set(loop.get_canvases(close_closed=True))
    def fail(*args,**kwargs):raise RuntimeError('injected configure failure')
    monkeypatch.setattr(WgpuContextToScreen,'configure',fail)
    try:
        with pytest.raises(GPUUnavailableError,match='injected configure failure'):
            viewer.set_present_method('native')
        assert viewer.surface is original
        assert set(loop.get_canvases(close_closed=True))==before
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
        assert set(loop.get_canvases(close_closed=True))==before
    finally:
        viewer.destroy()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
        app.processEvents()
