"""Apply the two reviewed lifecycle deltas plus the destroy-fallback gate to qt.py.

Preserves CRLF line endings; every replacement must match exactly once.
"""
import io, sys

PATH = r"C:/Github/ANY3dView/src/any3dview/qt.py"
src = io.open(PATH, encoding="utf-8", newline="").read()

def sub(old, new):
    global src
    count = src.count(old)
    assert count == 1, f"expected 1 match, got {count}: {old[:80]!r}"
    src = src.replace(old, new)

# --- B: initializeGL ready flag + incomplete same-context recovery ---
sub(
"""        if prior is context and getattr(owner, "_renderer", None) is not None:\r
            # Qt re-ran initializeGL on the same live context: the\r
            # installed renderer/HUD are still owned and unreleased, so\r
            # re-running the constructor path would only install\r
            # unreleased replacements over them.\r
            owner._note_lifecycle("initialize_gl_repeats")\r
            return\r
""",
"""        if prior is context and getattr(owner, "_renderer", None) is not None:\r
            if getattr(owner, "_gl_ready", False):\r
                # Qt re-ran initializeGL on the same live context: the\r
                # installed renderer/HUD are still owned and unreleased, so\r
                # re-running the constructor path would only install\r
                # unreleased replacements over them.\r
                owner._note_lifecycle("initialize_gl_repeats")\r
                return\r
            # A previous attempt on this same context failed after ownership\r
            # was assigned (replay or highlight initialization): partially\r
            # initialized resources must not be treated as a successful\r
            # repeat. Release them and rebind the context before rebuilding.\r
            owner._note_lifecycle("initialize_gl_incomplete_repeats")\r
            owner._release_gl(context)\r
            self.makeCurrent()\r
""")

# --- B: ready flag set only after retained replay + highlight succeed ---
sub(
"""            owner._apply_highlight_masks()\r
        except Exception as error:\r
""",
"""            owner._apply_highlight_masks()\r
            owner._gl_ready = True\r
        except Exception as error:\r
""")

# --- B: ready flag starts clear next to the ownership token ---
sub(
"""        self._gl_context = None\r
        self._gl_destroy_slot = None\r
""",
"""        self._gl_context = None\r
        self._gl_ready = False\r
        self._gl_destroy_slot = None\r
""")

# --- A: check surface.context identity BEFORE makeCurrent ---
sub(
"""        bound = False\r
        if valid:\r
            try:\r
                surface.makeCurrent()\r
                bound = True\r
            except Exception as error:\r
                failures.append(f"makeCurrent: {type(error).__name__}: {error}")\r
                valid = False\r
        if valid:\r
            # A valid replacement context must never receive old-context\r
            # release calls: verify both the surface's context and Qt's\r
            # thread-current context are the owned token before releasing.\r
            try:\r
                surface_context = surface.context()\r
            except Exception:\r
                surface_context = None\r
            try:\r
                current = QOpenGLContext.currentContext()\r
            except Exception:\r
                current = None\r
            if surface_context is not owner_context:\r
                failures.append("skipped GL release: surface context is not the owned context")\r
                valid = False\r
            elif current is not owner_context:\r
                failures.append("skipped GL release: current context is not the owned context")\r
                valid = False\r
""",
"""        bound = False\r
        if valid:\r
            # The surface's context identity is verified BEFORE binding:\r
            # during initializeGL on a replacement context, makeCurrent\r
            # would bind the new context and doneCurrent would strip it\r
            # before moderngl.create_context runs. A replaced surface\r
            # context is recorded and skipped without any binding.\r
            try:\r
                surface_context = surface.context()\r
            except Exception:\r
                surface_context = None\r
            if surface_context is not owner_context:\r
                failures.append("skipped GL release: surface context is not the owned context")\r
                valid = False\r
        if valid:\r
            try:\r
                surface.makeCurrent()\r
                bound = True\r
            except Exception as error:\r
                failures.append(f"makeCurrent: {type(error).__name__}: {error}")\r
                valid = False\r
        if valid:\r
            # Binding succeeded; Qt's thread-current context must still be\r
            # the owned token before any release call is issued.\r
            try:\r
                current = QOpenGLContext.currentContext()\r
            except Exception:\r
                current = None\r
            if current is not owner_context:\r
                failures.append("skipped GL release: current context is not the owned context")\r
                valid = False\r
""")

# --- ready flag cleared on release, next to the ownership token ---
sub(
"""        self._gl_context = None\r
        self._note_lifecycle("gl_release_attempts")\r
""",
"""        self._gl_context = None\r
        self._gl_ready = False\r
        self._note_lifecycle("gl_release_attempts")\r
""")

# --- destroy: direct release fallback is for software resources only ---
sub(
"""        if self._renderer is not None:\r
            try:\r
                self._renderer.release()\r
            except BaseException as error:\r
                failures.append(f"renderer.release: {type(error).__name__}: {error}")\r
""",
"""        if self._renderer is not None:\r
            if getattr(self, "_gl_context", None) is not None:\r
                # _release_gl failed before detaching ownership: the leftover\r
                # renderer belongs to an unverified GL context and must not\r
                # be released directly (wrong-context risk). The direct\r
                # release fallback is for software resources only.\r
                failures.append("skipped leftover GL renderer release: "\r
                                "_release_gl failed before detaching ownership")\r
            else:\r
                try:\r
                    self._renderer.release()\r
                except BaseException as error:\r
                    failures.append(f"renderer.release: {type(error).__name__}: {error}")\r
""")

io.open(PATH, "w", encoding="utf-8", newline="").write(src)
print("qt.py patched OK")
