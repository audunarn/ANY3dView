"""Independent passive-diagnostics oracle: no live GL or widget activity."""
import json
from types import SimpleNamespace
import pytest
pytest.importorskip('PySide6')
from any3dview import qt

class Forbidden:
    def __getattr__(self,name):
        raise AssertionError('Diagnostic touched live resource: '+name)

@pytest.mark.parametrize('valid',[True,False,'deleted'])
def test_snapshot_reads_cached_identity_and_qt_validity_only(monkeypatch,valid):
    observed=[]
    def is_valid():
        observed.append('isValid')
        if valid=='deleted':raise RuntimeError('deleted Qt object')
        return valid
    owner=SimpleNamespace(_backend='gpu',_closed=False,
        _gl_context=SimpleNamespace(isValid=is_valid),
        _gl_info={'vendor':'cached vendor','renderer':'cached renderer','version':'cached version'},
        _lifecycle_counters={'initialize_gl_attempts':3},_lifecycle_errors=['retained error'],
        _renderer=Forbidden(),surface=Forbidden(),camera=Forbidden())
    monkeypatch.setattr(qt,'QOpenGLContext',Forbidden())
    before=dict(vars(owner))
    result=qt.QtViewer.diagnostic_snapshot(owner)
    assert observed==['isValid'] and vars(owner)==before
    assert result['context_valid'] is (valid is True)
    assert result['gl_vendor']=='cached vendor'
    assert result['gl_renderer']=='cached renderer'
    assert result['gl_version']=='cached version'
    json.dumps(result)
    result['lifecycle_counters']['initialize_gl_attempts']=100
    result['lifecycle_errors'].append('mutated copy')
    assert owner._lifecycle_counters=={'initialize_gl_attempts':3}
    assert owner._lifecycle_errors==['retained error']


def test_closed_software_snapshot_requires_no_context_or_gl():
    owner=SimpleNamespace(_backend='software',_closed=True,_renderer=Forbidden(),surface=Forbidden())
    result=qt.QtViewer.diagnostic_snapshot(owner)
    assert result['closed'] and result['active_backend']=='software'
    assert result['context_valid'] is False
    json.dumps(result)
