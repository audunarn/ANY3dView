"""Real Qt timer and shared scheduler checks, with no QWidget or GPU context."""
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
import pytest
pytest.importorskip('PySide6')
from PySide6.QtCore import QObject,QCoreApplication,QEventLoop,QTimer
from any3dview.qt import QtViewer
from any3dview.scheduler import ViewerScheduler

class Pump(QObject):
    _start_update_pump=QtViewer._start_update_pump
    _poll_updates=QtViewer._poll_updates
    after_cancel=QtViewer.after_cancel
    destroy=QtViewer.destroy
    _cancel_hover=QtViewer._cancel_hover
    def __init__(self):
        super().__init__();self._closed=False;self._update_poll_id=None;self._timers=set()
        self._update_scheduler=ViewerScheduler();self.redraws=0
        self._hover_timer=None;self._pending_hover=None
        self._renderer=SimpleNamespace(release=lambda:None)
    def redraw(self):
        if not self._closed:self.redraws+=1
    def stop_animation(self):pass
    def _release_gl(self):pass
    def hide(self):pass

@pytest.fixture
def pump():
    app=QCoreApplication.instance() or QCoreApplication([])
    p=Pump();yield p
    if not p._closed:p.destroy()


def test_one_notifier_and_no_idle_timer(pump):
    pump._start_update_pump();notifier=pump._update_poll_id
    for _ in range(100):
        pump._start_update_pump();pump._poll_updates()
    assert pump._update_poll_id is notifier
    assert not pump._timers
    assert not pump.findChildren(QTimer)
    assert pump.redraws==0


def test_worker_submission_runs_fifo_on_owner_through_real_timer(pump):
    seen=[];owner=get_ident();pump._start_update_pump()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(lambda:[pump._update_scheduler.submit(lambda i=i:seen.append((i,get_ident()))) for i in range(4)]).result()
    assert seen==[]
    loop=QEventLoop();QTimer.singleShot(20,loop.quit);loop.exec()
    assert seen==[(i,owner) for i in range(4)]
    assert pump.redraws==1


def test_drain_limit_retained_and_next_tick_finishes(pump):
    seen=[]
    pump._start_update_pump()
    for i in range(1025):pump._update_scheduler.submit(seen.append,i)
    pump._poll_updates()
    assert seen==list(range(1024))
    pump._poll_updates();assert seen==list(range(1025))
    assert pump.redraws==2


def test_destroy_rejects_submissions_and_stale_notification(pump):
    pump._start_update_pump();seen=[]
    pump._update_scheduler.submit(seen.append,'stale')
    pump.destroy()
    assert not pump._timers
    pump._poll_updates();pump._start_update_pump();assert seen==[]
    with pytest.raises(RuntimeError,match='closed'):pump._update_scheduler.submit(lambda:None)


def test_destroy_during_drain_discards_remaining_callbacks(pump):
    seen=[];pump._update_scheduler.submit(pump.destroy)
    pump._update_scheduler.submit(seen.append,'stale')
    pump._start_update_pump()
    assert pump._closed and seen==[] and pump.redraws==0
    assert not pump._timers


def test_worker_cannot_drain_or_execute_callback(pump):
    seen=[];pump._update_scheduler.submit(seen.append,'owner')
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError,match='owning thread'):pool.submit(pump._poll_updates).result()
    assert seen==[]
    pump._poll_updates();assert seen==['owner']


def test_initial_callback_failure_does_not_leave_running_timer(pump):
    def fail():raise ValueError('bad update')
    pump._update_scheduler.submit(fail)
    with pytest.raises(ValueError,match='bad update'):pump._start_update_pump()
    assert pump._update_poll_id is None and not pump._timers
    pump._start_update_pump();assert pump._update_poll_id is not None


def test_many_submissions_coalesce_and_bounded_event_reschedules(pump,monkeypatch):
    from any3dview import qt
    events=[]
    monkeypatch.setattr(qt.QCoreApplication,'postEvent',lambda receiver,event:events.append((receiver,event)))
    pump._start_update_pump();seen=[]
    for i in range(2049):pump._update_scheduler.submit(seen.append,i)
    assert len(events)==1
    receiver,event=events.pop(0);receiver.customEvent(event)
    assert len(seen)==1024 and len(events)==1
    receiver,event=events.pop(0);receiver.customEvent(event)
    assert len(seen)==2048 and len(events)==1
    receiver,event=events.pop(0);receiver.customEvent(event)
    assert seen==list(range(2049)) and not events


def test_failed_post_can_retry(pump,monkeypatch):
    from any3dview import qt
    pump._start_update_pump();notifier=pump._update_poll_id
    def fail(*args):raise RuntimeError('post failed')
    monkeypatch.setattr(qt.QCoreApplication,'postEvent',fail)
    with pytest.raises(RuntimeError,match='post failed'):notifier.wake()
    assert not notifier._pending
    events=[]
    monkeypatch.setattr(qt.QCoreApplication,'postEvent',lambda *args:events.append(args))
    notifier.wake();assert len(events)==1
