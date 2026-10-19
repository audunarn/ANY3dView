"""Qt GL context ownership and cleanup lifecycle, fake Qt objects only.

No widgets, native contexts, GUI event loops or GPU runs: every test drives
unbound QtViewer/_GLSurface methods with SimpleNamespace fakes, matching the
established pattern in tests/test_qt_deferred_failure.py.
"""
import json
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

import any3dview.qt as qt
from any3dview.qt import QtViewer

DELETED = RuntimeError("Internal C++ object (QOpenGLContext) already deleted.")


class FakeQOpenGLContext:
    """Stand-in for qt.QOpenGLContext; tests set the thread-current context."""
    current = None

    @staticmethod
    def currentContext():
        return FakeQOpenGLContext.current


@pytest.fixture(autouse=True)
def qt_current_context(monkeypatch):
    FakeQOpenGLContext.current = None
    monkeypatch.setattr(qt, "QOpenGLContext", FakeQOpenGLContext)
    return FakeQOpenGLContext


def make_counters():
    return {"initialize_gl_attempts": 0, "context_recreations": 0,
            "initialize_gl_repeats": 0,
            "gl_release_attempts": 0, "gl_release_failures": 0,
            "destroy_calls": 0, "destroy_failures": 0}


def make_resource(name, failure=None):
    calls = []

    def release():
        calls.append(name)
        if failure is not None:
            raise failure

    return SimpleNamespace(release=release), calls


def make_surface(valid=True, make_current_failure=None):
    calls = []

    def is_valid():
        calls.append("isValid")
        if valid is DELETED:
            raise DELETED
        return valid

    def make_current():
        calls.append("makeCurrent")
        if make_current_failure is not None:
            raise make_current_failure

    def done_current():
        calls.append("doneCurrent")

    return SimpleNamespace(isValid=is_valid, makeCurrent=make_current,
                           doneCurrent=done_current, calls=calls)


def make_context(valid=True, with_signal=False):
    calls = []

    def is_valid():
        calls.append("isValid")
        if valid is DELETED:
            raise DELETED
        return valid

    context = SimpleNamespace(isValid=is_valid, calls=calls)
    if with_signal:
        connected = []

        def connect(slot):
            calls.append("connect")
            connected.append(slot)

        def disconnect(slot):
            calls.append("disconnect")
            if slot not in connected:
                raise RuntimeError("slot not connected")
            connected.remove(slot)

        context.aboutToBeDestroyed = SimpleNamespace(connect=connect,
                                                     disconnect=disconnect)
        context.connected_slots = connected
    return context


def make_owner(backend="gpu", hud=None, renderer=None, gl_context="absent",
               surface=None, closed=False):
    owner = SimpleNamespace(_backend=backend, _hud=hud, _renderer=renderer,
                            _gl_context=gl_context, surface=surface,
                            _closed=closed,
                            _lifecycle_counters=make_counters(),
                            _lifecycle_errors=[])
    owner._note_lifecycle = QtViewer._note_lifecycle.__get__(owner)
    owner._release_gl = QtViewer._release_gl.__get__(owner)
    if surface is not None and gl_context not in (None, "absent"):
        # The surface reports the owned context, and it is thread-current:
        # the default happy path for identity verification.
        surface.context = lambda: gl_context
        FakeQOpenGLContext.current = gl_context
    return owner


def make_destroyable(owner):
    """Add the collaborators QtViewer.destroy touches, as recording fakes."""
    owner.stop_animation = lambda: None
    owner._cancel_hover = lambda: None
    owner._update_scheduler = SimpleNamespace(close=lambda: None)
    owner._timers = set()
    owner.after_cancel = QtViewer.after_cancel.__get__(owner)
    owner.hide = lambda: owner.__dict__.setdefault("hidden", []).append(True)
    owner.deleteLater = lambda: owner.__dict__.setdefault(
        "deleted_later", []).append(True)
    owner.destroy = QtViewer.destroy.__get__(owner)
    return owner


class TestReleaseGuardedByOwnershipToken:
    def test_stale_callback_cannot_release_replacement_resources(self):
        old_context, new_context = make_context(), make_context()
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=new_context,
                           surface=surface)
        QtViewer._release_gl(owner, old_context)
        assert hud_calls == [] and renderer_calls == []
        assert owner._gl_context is new_context
        assert owner._lifecycle_counters["gl_release_attempts"] == 0
        QtViewer._release_gl(owner, new_context)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert owner._gl_context is None and owner._hud is None
        assert owner._renderer is None
        assert owner._lifecycle_counters["gl_release_attempts"] == 1

    def test_release_without_token_uses_tracked_context(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=make_context(),
                           surface=surface)
        QtViewer._release_gl(owner)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert surface.calls == ["isValid", "makeCurrent", "doneCurrent"]

    def test_no_token_means_nothing_owned_so_release_is_noop(self):
        surface = make_surface(valid=True)
        owner = make_owner(hud=None,
                           renderer=SimpleNamespace(release=lambda: None),
                           gl_context=None, surface=surface)
        QtViewer._release_gl(owner)
        assert surface.calls == []
        assert owner._renderer is not None  # untouched: not GL-owned
        assert owner._lifecycle_counters["gl_release_attempts"] == 0

    def test_software_backend_never_takes_the_gl_path(self):
        surface = make_surface(valid=True)
        owner = make_owner(backend="software",
                           hud=SimpleNamespace(release=lambda: None),
                           renderer=SimpleNamespace(release=lambda: None),
                           gl_context=make_context(), surface=surface)
        QtViewer._release_gl(owner)
        assert surface.calls == []
        assert owner._hud is not None and owner._renderer is not None


class TestCleanupDisconnectsContextCallback:
    def test_release_disconnects_the_exact_stored_slot(self):
        context = make_context(with_signal=True)
        slot_calls = []
        slot = lambda: slot_calls.append("fired")
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=context,
                           surface=surface)
        context.aboutToBeDestroyed.connect(slot)
        owner._gl_destroy_slot = (context, slot)
        QtViewer._release_gl(owner)
        assert "disconnect" in context.calls
        assert context.connected_slots == []
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert owner._gl_destroy_slot is None
        # The disconnected callback is inert even if it is ever invoked.
        slot()
        assert owner._lifecycle_counters["gl_release_attempts"] == 1

    def test_stale_slot_entry_cannot_disconnect_the_replacement(self):
        old_context = make_context(with_signal=True)
        old_slot = lambda: None
        new_context = make_context(with_signal=True)
        new_slot = lambda: None
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=new_context,
                           surface=surface)
        old_context.aboutToBeDestroyed.connect(old_slot)
        new_context.aboutToBeDestroyed.connect(new_slot)
        owner._gl_destroy_slot = (old_context, old_slot)
        QtViewer._release_gl(owner)
        assert old_context.connected_slots == [old_slot]  # stale entry untouched
        assert new_context.connected_slots == [new_slot]
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]


class TestContextIdentityGuards:
    def test_wrong_surface_context_never_receives_old_release_calls(self):
        owned = make_context()
        replacement = make_context()
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=owned,
                           surface=surface)
        surface.context = lambda: replacement  # Qt replaced the surface context
        QtViewer._release_gl(owner)
        assert hud_calls == [] and renderer_calls == []
        assert surface.calls == ["isValid"]
        assert owner._hud is None and owner._renderer is None
        assert any("surface context" in error
                   for error in owner._lifecycle_errors)

    def test_wrong_current_context_skips_release_but_unbinds(self):
        owned = make_context()
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=owned,
                           surface=surface)
        FakeQOpenGLContext.current = make_context()  # not the owned token
        QtViewer._release_gl(owner)
        assert hud_calls == [] and renderer_calls == []
        assert "doneCurrent" in surface.calls  # binding succeeded: unbind
        assert any("current context" in error
                   for error in owner._lifecycle_errors)

    def test_missing_current_context_skips_release(self):
        owned = make_context()
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer, gl_context=owned,
                           surface=surface)
        FakeQOpenGLContext.current = None  # deleted/absent thread-current
        QtViewer._release_gl(owner)
        assert hud_calls == [] and renderer_calls == []
        assert any("current context" in error
                   for error in owner._lifecycle_errors)


class TestReleaseContinuesAfterFailures:
    def test_hud_release_failure_still_releases_renderer_and_done_current(self):
        hud, hud_calls = make_resource("hud", failure=RuntimeError("hud boom"))
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(), surface=surface)
        QtViewer._release_gl(owner)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert surface.calls[-1] == "doneCurrent"
        assert owner._lifecycle_counters["gl_release_failures"] == 1
        assert any("hud boom" in error for error in owner._lifecycle_errors)
        assert owner._hud is None and owner._renderer is None

    def test_renderer_release_failure_still_releases_hud(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource(
            "renderer", failure=RuntimeError("renderer boom"))
        surface = make_surface(valid=True)
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(), surface=surface)
        QtViewer._release_gl(owner)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert owner._lifecycle_counters["gl_release_failures"] == 1

    def test_make_current_failure_skips_gl_releases_but_clears_ownership(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True,
                               make_current_failure=RuntimeError("no current"))
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(), surface=surface)
        QtViewer._release_gl(owner)
        assert hud_calls == [] and renderer_calls == []
        assert "doneCurrent" not in surface.calls
        assert owner._hud is None and owner._renderer is None
        assert owner._lifecycle_counters["gl_release_failures"] == 1

    @pytest.mark.parametrize("surface_valid", [False, DELETED])
    def test_invalid_or_deleted_context_never_receives_gl_calls(self, surface_valid):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=surface_valid)
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(valid=surface_valid),
                           surface=surface)
        QtViewer._release_gl(owner)
        assert hud_calls == [] and renderer_calls == []
        assert "makeCurrent" not in surface.calls
        assert owner._hud is None and owner._renderer is None
        assert any("skipped GL release" in error
                   for error in owner._lifecycle_errors)

    def test_repeated_release_is_safe_and_counted_once(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(), surface=make_surface())
        QtViewer._release_gl(owner)
        QtViewer._release_gl(owner)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert owner._lifecycle_counters["gl_release_attempts"] == 1


class TestDestroy:
    def test_gpu_destroy_releases_gl_owners_exactly_once(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        owner = make_destroyable(make_owner(
            hud=hud, renderer=renderer, gl_context=make_context(),
            surface=make_surface(valid=True)))
        owner.destroy()
        owner.destroy()
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert owner._lifecycle_counters["destroy_calls"] == 1
        assert owner._closed and owner.hidden == [True]
        assert owner.deleted_later == [True]

    def test_software_destroy_releases_software_resources_once(self):
        software, software_calls = make_resource("software")
        owner = make_destroyable(make_owner(
            backend="software", hud=SimpleNamespace(release=lambda: None),
            renderer=software, gl_context=None, surface=make_surface()))
        owner.destroy()
        owner.destroy()
        assert software_calls == ["software"]
        assert owner._lifecycle_counters["destroy_calls"] == 1

    def test_gpu_destroy_before_initialization_releases_placeholder_once(self):
        placeholder_calls = []
        placeholder = qt._SoftwareResources()
        placeholder.release = lambda: placeholder_calls.append("placeholder")
        owner = make_destroyable(make_owner(
            hud=None, renderer=placeholder, gl_context=None,
            surface=make_surface()))
        owner.destroy()
        owner.destroy()
        assert placeholder_calls == ["placeholder"]


class TestDestroyResilience:
    class FakeTimer:
        def __init__(self, cancelled):
            self.stop = lambda: cancelled.append("stop")
            self.deleteLater = lambda: cancelled.append("delete")

    def test_stop_animation_failure_still_cancels_and_hides(self):
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        owner = make_destroyable(make_owner(
            hud=hud, renderer=renderer, gl_context=make_context(),
            surface=make_surface(valid=True)))
        def broken_animation():
            raise RuntimeError("animation boom")
        owner.stop_animation = broken_animation
        cancelled = []
        owner._timers = {self.FakeTimer(cancelled)}
        owner.destroy()
        assert cancelled == ["stop", "delete"]
        assert owner._closed and owner.hidden == [True]
        assert owner.deleted_later == [True]
        assert hud_calls == ["hud"]  # GL release still happened
        assert renderer_calls == ["renderer"]  # exactly one release, no duplicate
        assert any("stop_animation" in error and "animation boom" in error
                   for error in owner._lifecycle_errors)
        assert owner._lifecycle_counters["destroy_failures"] == 1

    def test_release_gl_failure_still_releases_software_and_hides(self):
        software, software_calls = make_resource("software")
        owner = make_destroyable(make_owner(
            backend="software", hud=None, renderer=software, gl_context=None,
            surface=make_surface()))
        def broken_release():
            raise RuntimeError("release boom")
        owner._release_gl = broken_release
        owner.destroy()
        assert software_calls == ["software"]  # legitimate software cleanup kept
        assert owner.hidden == [True] and owner.deleted_later == [True]
        assert any("_release_gl" in error and "release boom" in error
                   for error in owner._lifecycle_errors)

    def test_renderer_release_failure_still_hides_and_deletes(self):
        renderer, renderer_calls = make_resource(
            "renderer", failure=RuntimeError("renderer boom"))
        owner = make_destroyable(make_owner(
            backend="software", hud=None, renderer=renderer, gl_context=None,
            surface=make_surface()))
        owner.destroy()
        assert renderer_calls == ["renderer"]
        assert owner.hidden == [True] and owner.deleted_later == [True]
        assert any("renderer.release" in error and "renderer boom" in error
                   for error in owner._lifecycle_errors)


class TestInitializeGL:
    @pytest.fixture
    def fake_gl(self, monkeypatch):
        connections = []
        state = SimpleNamespace(released=0, uploads=[], instances=[],
                                hud_released=0, hud_instances=0,
                                ctx=SimpleNamespace(info={
                                    "GL_VENDOR": "vendor inc",
                                    "GL_RENDERER": "fake gpu",
                                    "GL_VERSION": "3.3 (fake)"}),
                                context_argument=None)
        state.hud_failure = None
        state.release_failure = None

        class FakeRenderer:
            def __init__(self, context):
                state.context_argument = context
                state.instances.append(self)
                self.ctx = state.ctx

            def release(self):
                state.released += 1
                if state.release_failure is not None:
                    raise state.release_failure

            def add_mesh(self, handle, color=None, opacity=None, **extra):
                state.uploads.append((handle, color, opacity))

        class FakeHud:
            def __init__(self, context):
                if state.hud_failure is not None:
                    raise state.hud_failure
                state.hud_instances += 1

            def release(self):
                state.hud_released += 1

        def build_surface(context, owner):
            surface = make_surface(valid=True)
            surface.owner = owner
            surface.context = lambda: context
            surface.aboutToBeDestroyed_connections = connections
            context.aboutToBeDestroyed = SimpleNamespace(
                connect=lambda callback: connections.append((context, callback)))
            return surface

        monkeypatch.setitem(sys.modules, "moderngl", SimpleNamespace(
            create_context=lambda **kwargs: SimpleNamespace(loader_kwargs=kwargs)))
        monkeypatch.setitem(sys.modules, "any3dview.gpu.renderer", SimpleNamespace(
            ModernGLRenderer=FakeRenderer))
        monkeypatch.setitem(sys.modules, "any3dview.gpu.hud", SimpleNamespace(
            GPUHudRenderer=FakeHud))
        return SimpleNamespace(state=state, connections=connections,
                               build_surface=build_surface)

    def make_gl_owner(self, surface):
        owner = SimpleNamespace(
            _backend="gpu", _closed=False, _initializing=True,
            _renderer="old renderer", _hud="old hud", _gl_context=None,
            _gl_destroy_slot=None, _gl_info={},
            _lifecycle_counters=make_counters(),
            _lifecycle_errors=[], surface=surface,
            _display_entries=lambda: {"entry": {
                "handle": "handle", "appearance": {"color": "red", "bogus": 1}}},
            _apply_highlight_masks=lambda: None)
        owner._note_lifecycle = QtViewer._note_lifecycle.__get__(owner)
        owner._release_gl = QtViewer._release_gl.__get__(owner)
        return owner

    def test_hud_failure_releases_renderer_and_keeps_prior_ownership(self, fake_gl):
        fake_gl.state.hud_failure = RuntimeError("hud construction failed")
        surface = fake_gl.build_surface(SimpleNamespace(), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        qt._GLSurface.initializeGL(surface)
        assert fake_gl.state.released == 1
        assert owner._renderer == "old renderer" and owner._hud == "old hud"
        assert owner._gl_context is None and fake_gl.connections == []
        assert "hud construction failed" in str(surface.error)
        assert owner._lifecycle_counters["initialize_gl_attempts"] == 1

    def test_hud_cleanup_failure_preserves_constructor_error(self, fake_gl):
        fake_gl.state.hud_failure = RuntimeError("hud construction failed")
        fake_gl.state.release_failure = RuntimeError("release boom")
        surface = fake_gl.build_surface(make_context(valid=True), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        qt._GLSurface.initializeGL(surface)
        assert "hud construction failed" in str(surface.error)
        assert "release boom" not in str(surface.error)
        assert fake_gl.state.released == 1  # cleanup was attempted
        assert any("release boom" in error for error in owner._lifecycle_errors)
        assert owner._lifecycle_counters["gl_release_failures"] == 1
        assert owner._renderer == "old renderer" and owner._hud == "old hud"
        assert owner._gl_context is None and fake_gl.connections == []

    def test_success_assigns_ownership_token_info_and_replay(self, fake_gl):
        surface = fake_gl.build_surface(SimpleNamespace(), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        context = surface.context()
        qt._GLSurface.initializeGL(surface)
        assert surface.error is None
        assert owner._renderer is fake_gl.state.instances[-1]
        assert owner._hud is not None and not isinstance(owner._hud, str)
        assert owner._gl_context is context
        assert owner._gl_destroy_slot[0] is context
        assert owner._gl_info == {"vendor": "vendor inc", "renderer": "fake gpu",
                                  "version": "3.3 (fake)"}
        assert len(fake_gl.connections) == 1
        assert fake_gl.connections[0][0] is context
        assert fake_gl.state.uploads == [("handle", "red", None)]
        assert owner._lifecycle_counters["initialize_gl_attempts"] == 1
        assert owner._lifecycle_counters["context_recreations"] == 0

    def test_repeated_init_same_context_keeps_installed_resources(self, fake_gl):
        surface = fake_gl.build_surface(make_context(valid=True), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        qt._GLSurface.initializeGL(surface)
        first_renderer = owner._renderer
        first_hud = owner._hud
        qt._GLSurface.initializeGL(surface)
        assert owner._renderer is first_renderer and owner._hud is first_hud
        assert len(fake_gl.state.instances) == 1
        assert len(fake_gl.connections) == 1
        assert owner._lifecycle_counters["initialize_gl_attempts"] == 2
        assert owner._lifecycle_counters["initialize_gl_repeats"] == 1
        assert owner._lifecycle_counters["context_recreations"] == 0
        assert surface.error is None

    def test_reinit_after_normal_cleanup_installs_fresh_resources(
            self, fake_gl, qt_current_context):
        surface = fake_gl.build_surface(make_context(valid=True), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        first = surface.context()
        qt._GLSurface.initializeGL(surface)
        # Normal cleanup through the stored callback: release + disconnect.
        qt_current_context.current = first
        owner._gl_destroy_slot[1]()
        assert fake_gl.state.released == 1 and owner._gl_context is None
        assert owner._gl_destroy_slot is None
        second = make_context(valid=True)
        surface.context = lambda: second
        second.aboutToBeDestroyed = SimpleNamespace(
            connect=lambda callback: fake_gl.connections.append((second, callback)))
        qt_current_context.current = second
        qt._GLSurface.initializeGL(surface)
        assert owner._gl_context is second
        assert owner._gl_destroy_slot[0] is second
        assert owner._renderer is fake_gl.state.instances[-1]
        assert owner._renderer is not fake_gl.state.instances[0]
        assert owner._lifecycle_counters["initialize_gl_attempts"] == 2
        assert owner._lifecycle_counters["context_recreations"] == 0
        assert fake_gl.state.released == 1  # only the normal cleanup release

    def test_recreation_counts_and_stale_callback_spares_new_resources(
            self, fake_gl, qt_current_context):
        surface = fake_gl.build_surface(make_context(valid=True), None)
        owner = self.make_gl_owner(surface)
        surface.owner = owner
        first = surface.context()
        qt._GLSurface.initializeGL(surface)
        second = make_context(valid=True)
        surface.context = lambda: second
        second.aboutToBeDestroyed = SimpleNamespace(
            connect=lambda callback: fake_gl.connections.append((second, callback)))
        qt._GLSurface.initializeGL(surface)
        assert owner._lifecycle_counters["context_recreations"] == 1
        assert owner._gl_context is second
        # The old context's resources are abandoned, not released on the
        # replacement context; the abandonment is recorded.
        assert fake_gl.state.released == 0 and fake_gl.state.hud_released == 0
        assert any("surface context" in error for error in owner._lifecycle_errors)
        current_hud, current_renderer = owner._hud, owner._renderer
        # The old context's aboutToBeDestroyed fires after the replacement is
        # live: it must not release the new owner's resources.
        for token, callback in fake_gl.connections:
            if token is first:
                callback()
        assert owner._hud is current_hud
        assert owner._renderer is current_renderer
        assert owner._gl_context is second
        assert fake_gl.state.hud_released == 0 and fake_gl.state.released == 0
        # The replacement context's own callback still releases its resources.
        qt_current_context.current = second
        for token, callback in fake_gl.connections:
            if token is second:
                callback()
        assert fake_gl.state.hud_released == 1 and fake_gl.state.released == 1
        assert owner._gl_context is None


    def test_replacement_cleanup_does_not_unbind_new_context(self, fake_gl, qt_current_context):
        first=make_context();surface=fake_gl.build_surface(first,None)
        owner=self.make_gl_owner(surface);surface.owner=owner
        qt._GLSurface.initializeGL(surface)
        second=make_context(with_signal=True)
        surface.context=lambda:second
        surface.makeCurrent=lambda:pytest.fail("must not bind replacement to clean old resources")
        surface.doneCurrent=lambda:pytest.fail("must not unbind replacement during initialization")
        qt_current_context.current=second
        qt._GLSurface.initializeGL(surface)
        assert surface.error is None and owner._gl_context is second
        assert qt_current_context.current is second and owner._gl_ready

    def test_failed_replay_reinitialize_rebuilds_instead_of_false_success(self, fake_gl, qt_current_context):
        context=make_context();surface=fake_gl.build_surface(context,None)
        owner=self.make_gl_owner(surface);surface.owner=owner
        def fail():raise RuntimeError("replay failed")
        owner._apply_highlight_masks=fail
        qt._GLSurface.initializeGL(surface)
        assert isinstance(surface.error,RuntimeError) and not getattr(owner,'_gl_ready',False)
        owner._apply_highlight_masks=lambda:None
        qt_current_context.current=context
        qt._GLSurface.initializeGL(surface)
        assert surface.error is None and owner._gl_ready
        assert len(fake_gl.state.instances)==2 and fake_gl.state.released==1



class TestRenderFailureStillReleases:
    def test_render_failure_releases_owned_gl_and_refuses_to_swap(self):
        failure = RuntimeError("context lost during paint")
        hud, hud_calls = make_resource("hud")
        renderer, renderer_calls = make_resource("renderer")
        surface = make_surface(valid=True)
        surface.error = failure
        surface.hide = lambda: surface.calls.append("hide")
        owner = make_owner(hud=hud, renderer=renderer,
                           gl_context=make_context(), surface=surface)
        owner._backend_diagnostics = ()
        owner._allow_fallback = False
        owner.setToolTip = lambda message: owner.__dict__.setdefault(
            "tooltips", []).append(message)
        owner.render_error = SimpleNamespace(
            emit=lambda message: owner.__dict__.setdefault(
                "errors", []).append(message))
        QtViewer._render_failed(owner, surface, failure)
        assert hud_calls == ["hud"] and renderer_calls == ["renderer"]
        assert "hide" in surface.calls
        assert owner._backend == "gpu"  # fallback refusal preserved
        assert owner.errors and "context lost" in owner.errors[0]


class TestDiagnosticSnapshot:
    def make_snapshot_owner(self, context=None):
        return SimpleNamespace(_backend="gpu", _closed=False,
                               _gl_context=context,
                               _gl_info={"vendor": "v", "renderer": "r",
                                         "version": "3.3"},
                               _lifecycle_counters=make_counters(),
                               _lifecycle_errors=["RuntimeError: earlier"])

    def test_snapshot_is_json_compatible_and_copies_are_independent(self):
        context = make_context(valid=True)
        owner = self.make_snapshot_owner(context)
        first = QtViewer.diagnostic_snapshot(owner)
        assert first["schema"] == "any3dview.qt.lifecycle"
        assert first["version"] == 1
        assert first["active_backend"] == "gpu"
        assert first["closed"] is False
        assert first["context_valid"] is True
        assert first["gl_vendor"] == "v" and first["gl_renderer"] == "r"
        assert first["gl_version"] == "3.3"
        assert first["lifecycle_errors"] == ["RuntimeError: earlier"]
        json.dumps(first)
        first["lifecycle_counters"]["destroy_calls"] = 99
        first["lifecycle_errors"].append("mutated")
        second = QtViewer.diagnostic_snapshot(owner)
        assert second["lifecycle_counters"]["destroy_calls"] == 0
        assert second["lifecycle_errors"] == ["RuntimeError: earlier"]
        assert context.calls == ["isValid", "isValid"]

    def test_snapshot_tolerates_deleted_context_and_missing_token(self):
        deleted = QtViewer.diagnostic_snapshot(
            self.make_snapshot_owner(make_context(valid=DELETED)))
        assert deleted["context_valid"] is False
        absent = QtViewer.diagnostic_snapshot(self.make_snapshot_owner(None))
        assert absent["context_valid"] is False
        assert absent["gl_vendor"] == "v"

    def test_snapshot_is_passive_no_surface_or_live_state_touched(self):
        context = make_context(valid=True)
        owner = self.make_snapshot_owner(context)
        # No surface/redraw/make_current/capture attributes exist: any
        # rendering, context or project access would raise AttributeError.
        snapshot = QtViewer.diagnostic_snapshot(owner)
        assert snapshot["context_valid"] is True
        assert context.calls == ["isValid"]  # Qt isValid only, nothing else


class TestSnapshotPoisonedLiveState:
    def test_snapshot_ignores_poisoned_surface_and_live_gl_queries(self):
        class Poisoned:
            def __getattr__(self, name):
                raise AssertionError(f"snapshot touched live state: {name}")

        class PoisonedContext:
            def isValid(self):
                return True

            def __getattr__(self, name):
                raise AssertionError(f"snapshot touched live GL: {name}")

        context = PoisonedContext()
        owner = SimpleNamespace(_backend="gpu", _closed=False,
                               _gl_context=context,
                               _gl_info={"vendor": "v", "renderer": "r",
                                         "version": "3.3"},
                               _lifecycle_counters=make_counters(),
                               _lifecycle_errors=[],
                               surface=Poisoned(), make_current=Poisoned(),
                               redraw=Poisoned(), capture_image=Poisoned(),
                               _renderer="owned", _hud="owned")
        snapshot = QtViewer.diagnostic_snapshot(owner)
        assert snapshot["context_valid"] is True
        json.dumps(snapshot)
        # Ownership is untouched: no makeCurrent, render or mutation happened.
        assert owner._gl_context is context
        assert owner._renderer == "owned" and owner._hud == "owned"


class TestDeletedSurfaceRecovery:
    def test_closed_viewer_rejects_update_without_touching_deleted_surface(self):
        class Poison:
            def makeCurrent(self):pytest.fail("closed viewer touched native surface")
        owner=SimpleNamespace(_closed=True,_backend='gpu',surface=Poison())
        with pytest.raises(RuntimeError,match='closed Qt viewer'):
            QtViewer.make_current(owner)

    def test_destroy_deleted_timer_and_widget_still_releases_and_closes(self):
        renderer,calls=make_resource('software')
        owner=make_destroyable(make_owner(backend='software',renderer=renderer,gl_context=None))
        def deleted():raise RuntimeError('Internal C++ object already deleted')
        owner._cancel_hover=deleted;owner.hide=deleted;owner.deleteLater=deleted
        owner.destroy();owner.destroy()
        assert owner._closed and calls==['software']
        assert len(owner._lifecycle_errors)==3

    def test_cleanup_failure_never_bypasses_gl_context_guard(self):
        renderer,calls=make_resource('GPU')
        owner=make_destroyable(make_owner(renderer=renderer,gl_context=make_context()))
        def fail():
            owner._gl_context=None
            raise RuntimeError('cleanup failed after detaching token')
        owner._release_gl=fail
        owner.destroy()
        assert calls==[] and owner.hidden==[True] and owner.deleted_later==[True]

    def test_auto_fallback_can_replace_already_deleted_surface(self,monkeypatch):
        error=RuntimeError('Internal C++ object (_GLSurface) already deleted')
        def deleted():raise error
        old=SimpleNamespace(error=error,hide=deleted)
        owner=make_owner(surface=old,gl_context=None)
        owner._allow_fallback=True;owner._entries={};owner._backend_diagnostics=()
        owner.setToolTip=lambda text:None;owner.render_error=SimpleNamespace(emit=lambda text:None)
        owner._display_entries=lambda:{}
        owner._apply_highlight_masks=lambda:None;owner.redraw=lambda:None
        changes=[];owner.backend_changed=SimpleNamespace(emit=changes.append)
        added=[];owner.layout=lambda:SimpleNamespace(addWidget=added.append,
            replaceWidget=lambda *args:pytest.fail('deleted old widget used in layout'))
        new=SimpleNamespace(show=lambda:None)
        monkeypatch.setattr(qt,'_PaintSurface',lambda parent:new)
        QtViewer._render_failed(owner,old,error)
        assert owner.surface is new and owner._backend=='software'
        assert added==[new] and changes==['software']
