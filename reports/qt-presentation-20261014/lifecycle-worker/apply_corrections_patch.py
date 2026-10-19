"""Apply Astra review corrections 1-5 to src/any3dview/qt.py (CRLF-safe).

Each pattern must occur exactly once; the script refuses partial application.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "src/any3dview/qt.py"

PATCHES = [
    # Correction 2/production: QOpenGLContext.currentContext() identity check.
    (
        "from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF, QSurfaceFormat",
        "from PySide6.QtGui import QColor, QFont, QImage, QPainter, QOpenGLContext, QPen, QPolygonF, QSurfaceFormat",
    ),
    # Correction 3: reconcile prior ownership before installing replacements;
    # repeated init on the same live context must not overwrite unreleased
    # resources.
    (
        '        owner._note_lifecycle("initialize_gl_attempts")\n'
        "        try:\n"
        "            import inspect\n"
        "            import moderngl\n"
        "            from .gpu.renderer import ModernGLRenderer\n"
        "            from .gpu.hud import GPUHudRenderer\n"
        "            context = self.context()\n"
        "            renderer = ModernGLRenderer(moderngl.create_context(\n",
        '        owner._note_lifecycle("initialize_gl_attempts")\n'
        "        context = self.context()\n"
        '        prior = getattr(owner, "_gl_context", None)\n'
        "        if prior is context and getattr(owner, \"_renderer\", None) is not None:\n"
        "            # Qt re-ran initializeGL on the same live context: the\n"
        "            # installed renderer/HUD are still owned and unreleased, so\n"
        "            # re-running the constructor path would only install\n"
        "            # unreleased replacements over them.\n"
        '            owner._note_lifecycle("initialize_gl_repeats")\n'
        "            return\n"
        "        if prior is not None and prior is not context:\n"
        '            owner._note_lifecycle("context_recreations")\n'
        "            # Reconcile prior ownership before installing replacements:\n"
        "            # the old token's callback is disconnected and the abandoned\n"
        "            # resources are dropped. The old context is no longer this\n"
        "            # surface's context, so _release_gl skips unsafe GL calls\n"
        "            # and records the abandonment instead of touching the\n"
        "            # replacement context.\n"
        "            owner._release_gl(prior)\n"
        "        try:\n"
        "            import inspect\n"
        "            import moderngl\n"
        "            from .gpu.renderer import ModernGLRenderer\n"
        "            from .gpu.hud import GPUHudRenderer\n"
        "            renderer = ModernGLRenderer(moderngl.create_context(\n",
    ),
    # Correction 4: preserve the HUD constructor error if the renderer
    # cleanup release also fails; record the cleanup error.
    (
        "            try:\n"
        "                hud = GPUHudRenderer(renderer.ctx)\n"
        "            except BaseException:\n"
        "                # The context is current here; release the renderer's\n"
        "                # derived resources so a failing HUD constructor cannot\n"
        "                # leak them past the ownership token below.\n"
        "                renderer.release()\n"
        "                raise\n",
        "            try:\n"
        "                hud = GPUHudRenderer(renderer.ctx)\n"
        "            except BaseException as error:\n"
        "                # The context is current here; release the renderer's\n"
        "                # derived resources so a failing HUD constructor cannot\n"
        "                # leak them past the ownership token below. The original\n"
        "                # constructor error is preserved if this cleanup fails.\n"
        "                try:\n"
        "                    renderer.release()\n"
        "                except BaseException as cleanup_error:\n"
        '                    owner._note_lifecycle("gl_release_failures",\n'
        "                                           error=cleanup_error)\n"
        "                raise\n",
    ),
    # Corrections 1+3: store the exact slot+context, disconnect on cleanup.
    (
        "            if owner._gl_context is not None and owner._gl_context is not context:\n"
        '                owner._note_lifecycle("context_recreations")\n'
        "            owner._gl_context = context\n"
        "            owner._gl_info = _capture_gl_info(renderer.ctx)\n"
        "            # The closure carries the context token: a stale callback for a\n"
        "            # replaced context must not release the replacement's resources.\n"
        "            context.aboutToBeDestroyed.connect(lambda: owner._release_gl(context))\n",
        "            owner._gl_context = context\n"
        "            owner._gl_info = _capture_gl_info(renderer.ctx)\n"
        "            # Store the exact slot and its context so cleanup can\n"
        "            # disconnect it (Qt's aboutToBeDestroyed lifecycle contract).\n"
        "            # The closure carries the context token: a stale callback for\n"
        "            # a replaced context must not release the replacement's\n"
        "            # resources.\n"
        "            slot = lambda: owner._release_gl(context)\n"
        '            owner._gl_destroy_slot = (context, slot)\n'
        "            context.aboutToBeDestroyed.connect(slot)\n",
    ),
    # Owner state: track the stored slot alongside the ownership token.
    (
        "        self._gl_context = None\n"
        "        self._gl_info = {}\n",
        "        self._gl_context = None\n"
        "        self._gl_destroy_slot = None\n"
        "        self._gl_info = {}\n",
    ),
    # Correction 2: document the identity verification contract.
    (
        "        The optional token is the Qt context whose aboutToBeDestroyed\n"
        "        fired; a stale callback for a replaced context must not release\n"
        "        the replacement's resources. Qt owns the native context: only\n"
        "        derived resources are released, never against an invalid context.\n"
        '        """\n',
        "        The optional token is the Qt context whose aboutToBeDestroyed\n"
        "        fired; a stale callback for a replaced context must not release\n"
        "        the replacement's resources. Qt owns the native context: only\n"
        "        derived resources are released, and only after the surface's\n"
        "        context and Qt's thread-current context are verified to be the\n"
        "        owned token; a valid replacement context is never released into.\n"
        '        """\n',
    ),
    # Corrections 1+2: disconnect the exact stored slot during cleanup; verify
    # surface.context() and QOpenGLContext.currentContext() identity after
    # makeCurrent; doneCurrent is exception-safe whenever binding succeeded.
    (
        "        if context is not None and context is not owner_context:\n"
        "            return\n"
        "        self._gl_context = None\n"
        '        self._note_lifecycle("gl_release_attempts")\n'
        "        hud, renderer, surface = self._hud, self._renderer, self.surface\n"
        "        failures = []\n"
        "        try:\n"
        "            valid = bool(surface.isValid()) and bool(owner_context.isValid())\n"
        "        except RuntimeError:\n"
        "            valid = False  # Deleted Qt objects; native resources are gone.\n"
        "        if not valid:\n"
        '            failures.append("skipped GL release: Qt context unavailable")\n'
        "        if valid:\n"
        "            try:\n"
        "                surface.makeCurrent()\n"
        "            except RuntimeError as error:\n"
        '                failures.append(f"makeCurrent: {type(error).__name__}: {error}")\n'
        "                valid = False\n"
        "        if valid:\n"
        "            # Each owner releases independently so one failure cannot skip\n"
        "            # the others or the doneCurrent pairing.\n"
        '            for name, resource in (("hud", hud), ("renderer", renderer)):\n'
        "                if resource is None:\n"
        "                    continue\n"
        "                try:\n"
        "                    resource.release()\n"
        "                except BaseException as error:\n"
        '                    failures.append(f"{name}.release: {type(error).__name__}: {error}")\n'
        "            try:\n"
        "                surface.doneCurrent()\n"
        "            except RuntimeError as error:\n"
        '                failures.append(f"doneCurrent: {type(error).__name__}: {error}")\n'
        "        for failure in failures:\n"
        '            self._note_lifecycle("gl_release_failures", error=RuntimeError(failure))\n'
        "        self._hud = None\n"
        "        self._renderer = None\n",
        "        if context is not None and context is not owner_context:\n"
        "            return\n"
        "        # Qt's lifecycle contract: disconnect the exact stored slot (and\n"
        "        # its context) during cleanup so a later aboutToBeDestroyed\n"
        "        # cannot re-enter release for resources that are already gone;\n"
        "        # a stale entry for another context is left untouched.\n"
        '        slot_entry = getattr(self, "_gl_destroy_slot", None)\n'
        "        if slot_entry is not None and slot_entry[0] is owner_context:\n"
        "            try:\n"
        "                owner_context.aboutToBeDestroyed.disconnect(slot_entry[1])\n"
        "            except Exception:\n"
        "                pass  # Already disconnected, or the Qt object is gone.\n"
        "            self._gl_destroy_slot = None\n"
        "        self._gl_context = None\n"
        '        self._note_lifecycle("gl_release_attempts")\n'
        "        hud, renderer, surface = self._hud, self._renderer, self.surface\n"
        "        failures = []\n"
        "        try:\n"
        "            valid = bool(surface.isValid()) and bool(owner_context.isValid())\n"
        "        except Exception:\n"
        "            valid = False  # Deleted/broken Qt objects; native resources are gone.\n"
        "        if not valid:\n"
        '            failures.append("skipped GL release: Qt context unavailable")\n'
        "        bound = False\n"
        "        if valid:\n"
        "            try:\n"
        "                surface.makeCurrent()\n"
        "                bound = True\n"
        "            except Exception as error:\n"
        '                failures.append(f"makeCurrent: {type(error).__name__}: {error}")\n'
        "                valid = False\n"
        "        if valid:\n"
        "            # A valid replacement context must never receive old-context\n"
        "            # release calls: verify both the surface's context and Qt's\n"
        "            # thread-current context are the owned token before releasing.\n"
        "            try:\n"
        "                surface_context = surface.context()\n"
        "            except Exception:\n"
        "                surface_context = None\n"
        "            try:\n"
        "                current = QOpenGLContext.currentContext()\n"
        "            except Exception:\n"
        "                current = None\n"
        "            if surface_context is not owner_context:\n"
        '                failures.append("skipped GL release: surface context is not the owned context")\n'
        "                valid = False\n"
        "            elif current is not owner_context:\n"
        '                failures.append("skipped GL release: current context is not the owned context")\n'
        "                valid = False\n"
        "        if valid:\n"
        "            # Each owner releases independently so one failure cannot skip\n"
        "            # the others or the doneCurrent pairing.\n"
        '            for name, resource in (("hud", hud), ("renderer", renderer)):\n'
        "                if resource is None:\n"
        "                    continue\n"
        "                try:\n"
        "                    resource.release()\n"
        "                except BaseException as error:\n"
        '                    failures.append(f"{name}.release: {type(error).__name__}: {error}")\n'
        "        if bound:\n"
        "            # The binding succeeded, so unbind it even when release was\n"
        "            # skipped; a doneCurrent failure must not mask earlier ones.\n"
        "            try:\n"
        "                surface.doneCurrent()\n"
        "            except Exception as error:\n"
        '                failures.append(f"doneCurrent: {type(error).__name__}: {error}")\n'
        "        for failure in failures:\n"
        '            self._note_lifecycle("gl_release_failures", error=RuntimeError(failure))\n'
        "        self._hud = None\n"
        "        self._renderer = None\n",
    ),
    # Correction 5: destroy survives raising cleanup steps; timers/scheduler
    # cancellation and hide/deleteLater always run; no duplicate GL release.
    (
        '        counters = getattr(self, "_lifecycle_counters", None)\n'
        "        if counters is not None:\n"
        '            counters["destroy_calls"] = counters.get("destroy_calls", 0) + 1\n'
        "        self.stop_animation()\n"
        "        self._closed = True\n"
        "        self._cancel_hover()\n"
        "        self._update_scheduler.close()\n"
        "        for timer in tuple(self._timers):\n"
        "            self.after_cancel(timer)\n"
        "        self._release_gl()\n"
        "        if self._renderer is not None:\n"
        "            self._renderer.release()\n"
        "        self.hide()\n"
        "        self.deleteLater()\n",
        '        counters = getattr(self, "_lifecycle_counters", None)\n'
        "        if counters is not None:\n"
        '            counters["destroy_calls"] = counters.get("destroy_calls", 0) + 1\n'
        "        # A raising cleanup step must not prevent the remaining\n"
        "        # shutdown: timers/scheduler cancellation and hide/deleteLater\n"
        "        # always run, and each failure is recorded.\n"
        "        failures = []\n"
        "        try:\n"
        "            self.stop_animation()\n"
        "        except BaseException as error:\n"
        '            failures.append(f"stop_animation: {type(error).__name__}: {error}")\n'
        "        self._closed = True\n"
        "        self._cancel_hover()\n"
        "        self._update_scheduler.close()\n"
        "        for timer in tuple(self._timers):\n"
        "            self.after_cancel(timer)\n"
        "        try:\n"
        "            self._release_gl()\n"
        "        except BaseException as error:\n"
        '            failures.append(f"_release_gl: {type(error).__name__}: {error}")\n'
        "        if self._renderer is not None:\n"
        "            try:\n"
        "                self._renderer.release()\n"
        "            except BaseException as error:\n"
        '                failures.append(f"renderer.release: {type(error).__name__}: {error}")\n'
        '        note = getattr(self, "_note_lifecycle", None)\n'
        "        for failure in failures:\n"
        "            if note is not None:\n"
        '                note("destroy_failures", error=RuntimeError(failure))\n'
        "        self.hide()\n"
        "        self.deleteLater()\n",
    ),
]


def main():
    text = TARGET.read_bytes().decode("utf-8")  # no newline translation
    for index, (old, new) in enumerate(PATCHES, 1):
        old = old.replace("\n", "\r\n")
        new = new.replace("\n", "\r\n")
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"patch {index}: pattern occurs {count} times, expected 1")
        text = text.replace(old, new)
    TARGET.write_bytes(text.encode("utf-8"))
    print(f"applied {len(PATCHES)} patches to {TARGET}")


if __name__ == "__main__":
    main()
