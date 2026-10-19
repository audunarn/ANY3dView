"""Deferred Qt failures must belong to the surface/error that queued them."""
from types import SimpleNamespace
import pytest
pytest.importorskip('PySide6')
from any3dview.qt import QtViewer


def owner(surface, backend='gpu', closed=False):
    events=[]
    return SimpleNamespace(surface=surface, _backend=backend, _closed=closed,
        _backend_diagnostics=(), _allow_fallback=False,
        setToolTip=lambda message:events.append(('tooltip',message)),
        render_error=SimpleNamespace(emit=lambda message:events.append(('error',message))),
        _release_gl=lambda:events.append(('release',None))), events


@pytest.mark.parametrize('case',['recovered','replaced','new_error','closed','software'])
def test_obsolete_failure_does_not_mutate_current_surface(case):
    old_error=RuntimeError('old context')
    source=SimpleNamespace(error=old_error,hide=lambda:pytest.fail('hidden'))
    current=source
    if case=='recovered':source.error=None
    if case=='replaced':current=SimpleNamespace(error=RuntimeError('other'))
    if case=='new_error':source.error=RuntimeError('new context')
    if case=='software':current=SimpleNamespace()  # no GL error attribute
    viewer,events=owner(current,backend='software' if case=='software' else 'gpu',closed=case=='closed')
    QtViewer._render_failed(viewer,source,old_error)
    assert events==[] and viewer._backend_diagnostics==()


def test_current_failure_still_reports_releases_and_hides():
    failure=RuntimeError('lost context'); hidden=[]
    surface=SimpleNamespace(error=failure,hide=lambda:hidden.append(True))
    viewer,events=owner(surface)
    QtViewer._render_failed(viewer,surface,failure)
    assert 'lost context' in viewer._backend_diagnostics[0]
    assert [e[0] for e in events]==['tooltip','error','release']
    assert hidden==[True]


def test_direct_failure_handler_ignores_recovered_surface():
    surface=SimpleNamespace(error=None,hide=lambda:pytest.fail('recovered surface hidden'))
    viewer,events=owner(surface)
    QtViewer._render_failed(viewer)
    assert events==[]

@pytest.mark.parametrize('recover',[False,True])
def test_deferred_callback_captures_original_failure(monkeypatch,recover):
    import any3dview.qt as qt
    queued=[]
    monkeypatch.setattr(qt,'QTimer',SimpleNamespace(singleShot=lambda delay,callback:queued.append(callback)))
    error=RuntimeError('first');surface=SimpleNamespace(error=error)
    calls=[];surface.owner=SimpleNamespace(_render_failed=lambda *args:calls.append(args))
    qt._GLSurface._defer_render_failure(surface)
    assert len(queued)==1 and calls==[]
    if recover:surface.error=None
    queued[0]()
    assert calls==[(surface,error)]
