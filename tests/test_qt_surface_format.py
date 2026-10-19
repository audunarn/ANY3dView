"""Application presentation policy must reach the explicit widget format."""
import pytest
pytest.importorskip('PySide6')
from PySide6.QtGui import QSurfaceFormat
from any3dview.qt import _gl_surface_format

@pytest.mark.parametrize('interval',[0,1,2])
def test_gl_format_honors_application_swap_policy_without_mutating_default(interval,monkeypatch):
    monkeypatch.setenv('ANY3DVIEW_MSAA_SAMPLES','0')
    original=QSurfaceFormat.defaultFormat()
    requested=QSurfaceFormat(original);requested.setSwapInterval(interval)
    try:
        QSurfaceFormat.setDefaultFormat(requested)
        actual=_gl_surface_format()
        assert actual.swapInterval()==interval
        assert actual.version()==(3,3)
        assert actual.profile()==QSurfaceFormat.CoreProfile
        assert actual.depthBufferSize()==24
        assert QSurfaceFormat.defaultFormat()==requested
    finally:
        QSurfaceFormat.setDefaultFormat(original)
