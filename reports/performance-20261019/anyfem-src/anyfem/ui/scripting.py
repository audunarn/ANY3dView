"""Compact trusted-Python editor and console for the Details workspace."""

from __future__ import annotations

import tkinter as tk
import traceback
from tkinter import ttk

from ..scripting import (
    ScriptConflictError,
    ScriptError,
    ScriptRunner,
)
from ..command_recording import command_event_to_python

__all__ = ["ScriptingPanel"]


class ScriptingPanel(ttk.Frame):
    """Run trusted local Python on an isolated working copy.

    The panel never commits from a worker callback.  It polls the future using
    ``after`` and performs the single transaction on Tk's owning thread.
    """

    title = "Scripting"
    POLL_MS = 50

    def __init__(self, master: tk.Misc, app) -> None:
        super().__init__(master, padding=8)
        self.app = app
        self._runner: ScriptRunner | None = None
        self._task = None
        self._poll_after = None
        self._observed_stack = None
        self._observed_session = None
        self._expected_revision_label = None

        ttk.Label(
            self,
            text="Trusted Python",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            self,
            text=(
                "Runs on a working copy. Successful changes are validated and "
                "applied as one undo item. This is trusted local code, not an "
                "OS sandbox.\n"
                "Globals: project, selection, commands, meshes, analyses, "
                "jobs, np and checkpoint."
            ),
            foreground="#555555",
            wraplength=330,
            justify="left",
        ).pack(fill="x", pady=(2, 6))

        editor_frame = ttk.Frame(self)
        editor_frame.pack(fill="both", expand=True)
        self.source = tk.Text(
            editor_frame,
            height=16,
            width=46,
            wrap="none",
            undo=True,
            font="TkFixedFont",
        )
        source_y = ttk.Scrollbar(
            editor_frame, orient="vertical", command=self.source.yview
        )
        source_x = ttk.Scrollbar(
            editor_frame, orient="horizontal", command=self.source.xview
        )
        self.source.configure(yscrollcommand=source_y.set, xscrollcommand=source_x.set)
        self.source.grid(row=0, column=0, sticky="nsew")
        source_y.grid(row=0, column=1, sticky="ns")
        source_x.grid(row=1, column=0, sticky="ew")
        editor_frame.rowconfigure(0, weight=1)
        editor_frame.columnconfigure(0, weight=1)
        self.source.insert(
            "1.0",
            "# Trusted local Python; Ctrl+Enter runs atomically.\n"
            "print(project.name)\n"
            "# point = commands.run(commands.AddPoint(0, 0, 0))\n",
        )
        self.source.bind("<Control-Return>", self._run_shortcut)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=(6, 4))
        self._run_button = ttk.Button(buttons, text="Run", command=self._run)
        self._run_button.pack(side="left")
        self._cancel_button = ttk.Button(
            buttons, text="Cancel", command=self._cancel, state="disabled"
        )
        self._cancel_button.pack(side="left", padx=4)
        self._recording = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            buttons,
            text="Record GUI commands",
            variable=self._recording,
        ).pack(side="right")
        ttk.Button(
            buttons,
            text="Copy diagnosis",
            command=self.copy_diagnosis,
        ).pack(side="right", padx=(4, 8))

        transcript = ttk.Notebook(self)
        transcript.pack(fill="both", expand=True, pady=(4, 0))
        command_page = ttk.Frame(transcript)
        output_page = ttk.Frame(transcript)
        transcript.add(command_page, text="GUI commands")
        transcript.add(output_page, text="Script output")

        command_buttons = ttk.Frame(command_page)
        command_buttons.pack(fill="x")
        ttk.Label(
            command_buttons,
            text="Live, replay-oriented model command stream",
            foreground="#555555",
        ).pack(side="left")
        ttk.Button(
            command_buttons,
            text="Copy to editor",
            command=self.copy_commands_to_editor,
        ).pack(side="right")
        ttk.Button(
            command_buttons, text="Clear", command=self.clear_commands
        ).pack(side="right", padx=4)
        self.command_stream = tk.Text(
            command_page,
            height=9,
            width=46,
            wrap="none",
            state="disabled",
            font="TkFixedFont",
        )
        self.command_stream.pack(fill="both", expand=True, pady=(2, 0))
        self._write_command(
            "# GUI model commands appear here after they succeed.\n"
            "# Paste/copy them into the editor to adapt or replay them.\n"
        )

        output_buttons = ttk.Frame(output_page)
        output_buttons.pack(fill="x")
        ttk.Label(output_buttons, text="Console").pack(side="left")
        ttk.Button(
            output_buttons, text="Clear", command=self.clear_output
        ).pack(side="right")
        self.output = tk.Text(
            output_page,
            height=9,
            width=46,
            wrap="word",
            state="disabled",
            font="TkFixedFont",
        )
        self.output.pack(fill="both", expand=True, pady=(2, 0))
        self._bind_recording_sources()

    def refresh(self) -> None:
        """Follow document replacement without ever committing to an old one."""

        if (
            self._runner is not None
            and self._task is None
            and self._runner.session is not self.app.session
        ):
            self._runner.shutdown(wait=False)
            self._runner = None
        self._bind_recording_sources()

    def _bind_recording_sources(self) -> None:
        stack = getattr(self.app, "commands", None)
        if stack is not self._observed_stack:
            if self._observed_stack is not None:
                self._observed_stack.remove_action_listener(self._on_command_event)
            self._observed_stack = stack
            if stack is not None:
                stack.add_action_listener(self._on_command_event)
        session = getattr(self.app, "session", None)
        if session is not self._observed_session:
            if self._observed_session is not None:
                self._observed_session.remove_listener(self._on_revision_event)
            self._observed_session = session
            self._expected_revision_label = None
            if session is not None:
                session.add_listener(self._on_revision_event)

    def _on_command_event(self, event) -> None:
        self._expected_revision_label = {
            "run": event.command.label,
            "undo": f"undo {event.command.label}",
            "redo": f"redo {event.command.label}",
        }.get(event.action)
        if self._recording.get():
            self._write_command(command_event_to_python(event) + "\n")

    def _on_revision_event(self, revision) -> None:
        """Expose committed GUI transactions which are not command objects."""

        label = str(getattr(revision, "label", "edit"))
        if label == self._expected_revision_label:
            self._expected_revision_label = None
            return
        self._expected_revision_label = None
        if self._recording.get():
            self._write_command(f"# GUI transaction: {label}\n")

    def _write_command(self, text: str) -> None:
        self.command_stream.configure(state="normal")
        self.command_stream.insert("end", text)
        self.command_stream.see("end")
        self.command_stream.configure(state="disabled")

    def clear_commands(self) -> None:
        self.command_stream.configure(state="normal")
        self.command_stream.delete("1.0", "end")
        self.command_stream.configure(state="disabled")

    def copy_commands_to_editor(self) -> None:
        source = self.command_stream.get("1.0", "end-1c")
        if not source.strip():
            self.app.set_status("the GUI command stream is empty")
            return
        existing = self.source.get("1.0", "end-1c")
        separator = "\n" if not existing or existing.endswith("\n") else "\n\n"
        self.source.insert("end", separator + source + "\n")
        self.source.see("end")
        self.app.set_status("GUI commands copied to the scripting editor")

    def copy_diagnosis(self) -> None:
        commands = self.command_stream.get("1.0", "end-1c").splitlines()
        report = self.app.diagnostic_report(recent_commands=commands)
        try:
            self.app.clipboard.copy_text(report)
        except tk.TclError as error:
            self.app.set_status(
                f"could not copy diagnosis: {error}",
                error=True,
                diagnostic={"type": type(error).__name__, "message": str(error)},
            )
            return
        self.app.set_status(
            f"diagnosis copied ({len(report):,} characters; "
            f"{len(getattr(self.app, '_error_diagnostics', ()))} error(s))"
        )

    def clear_output(self) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _write(self, text: str) -> None:
        if not text:
            return
        self.output.configure(state="normal")
        self.output.insert("end", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _run_shortcut(self, _event=None) -> str:
        self._run()
        return "break"

    def _run(self) -> None:
        if self._task is not None:
            self.app.set_status("a script is already running")
            return
        source = self.source.get("1.0", "end-1c")
        if not source.strip():
            self.app.set_status("enter Python source before running")
            return
        if self._runner is None or self._runner.session is not self.app.session:
            if self._runner is not None:
                self._runner.shutdown(wait=False)
            self._runner = ScriptRunner(self.app.session)
        self._write("\n>>> Run working copy\n")
        try:
            self._task = self._runner.submit(source)
        except (ValueError, TypeError, PermissionError, RuntimeError) as error:
            self._write(f"{type(error).__name__}: {error}\n")
            self.app.set_status(
                f"{type(error).__name__}: {error}",
                error=True,
                diagnostic={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            self._task = None
            return
        self._run_button.configure(state="disabled")
        self._cancel_button.configure(state="normal")
        self.app.set_status("running script on a working copy")
        self._poll_after = self.after(self.POLL_MS, self._poll)

    def _cancel(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        self._cancel_button.configure(state="disabled")
        self.app.set_status("cancelling script")

    def _poll(self) -> None:
        self._poll_after = None
        task = self._task
        if task is None:
            return
        if not task.done():
            self._poll_after = self.after(self.POLL_MS, self._poll)
            return
        self._task = None
        self._run_button.configure(state="normal")
        self._cancel_button.configure(state="disabled")
        try:
            result = task.result()
            if self._runner is None or self._runner.session is not self.app.session:
                raise ScriptConflictError(
                    "the document was replaced while the script was running; "
                    "the proposal was discarded",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            outcome = self._runner.commit(result)
        except ScriptError as error:
            self._write(error.stdout)
            self._write(error.stderr)
            if error.traceback_text:
                self._write(error.traceback_text)
            self._write(f"Rejected: {error}\n")
            self.app.set_status(
                str(error),
                error=True,
                diagnostic={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": error.traceback_text,
                    "stdout": error.stdout,
                    "stderr": error.stderr,
                },
            )
            return
        except (ValueError, TypeError, PermissionError, RuntimeError) as error:
            self._write(f"Rejected: {type(error).__name__}: {error}\n")
            self.app.set_status(
                f"{type(error).__name__}: {error}",
                error=True,
                diagnostic={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            return

        self._write(outcome.stdout)
        self._write(outcome.stderr)
        if outcome.return_value is not None:
            self._write(f"result = {outcome.return_value!r}\n")
        if outcome.committed:
            self._write("Applied as one undo item.\n")
            # Mesh-only scripts can intentionally replace the active cached
            # mesh. Model edits have already invalidated that cache.
            if outcome.meshes_changed:
                self.app.mesh = self.app.session.mesh_cache.get("active")
            self.app.set_status("script applied atomically")
        else:
            self._write("Completed; no document changes.\n")
            self.app.set_status("script completed")

    def destroy(self) -> None:
        if self._observed_stack is not None:
            self._observed_stack.remove_action_listener(self._on_command_event)
            self._observed_stack = None
        if self._observed_session is not None:
            self._observed_session.remove_listener(self._on_revision_event)
            self._observed_session = None
        if self._poll_after is not None:
            try:
                self.after_cancel(self._poll_after)
            except tk.TclError:
                pass
            self._poll_after = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
        if self._runner is not None:
            self._runner.shutdown(wait=False)
            self._runner = None
        super().destroy()
