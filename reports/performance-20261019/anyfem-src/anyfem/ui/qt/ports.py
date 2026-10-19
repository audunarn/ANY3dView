"""Qt implementations of the toolkit-neutral application ports."""
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox


class QtSchedulerPort:
    def __init__(self, parent):
        self.parent = parent
        self.pending = set()

    def call_later(self, delay_ms, callback):
        timer = QTimer(self.parent)
        timer.setSingleShot(True)
        self.pending.add(timer)
        def invoke():
            self.pending.discard(timer)
            timer.deleteLater()
            callback()
        timer.timeout.connect(invoke)
        timer.start(delay_ms)
        return timer

    def cancel_call(self, identifier):
        if identifier in self.pending:
            identifier.stop()
            self.pending.discard(identifier)
            identifier.deleteLater()

    def close(self):
        for timer in tuple(self.pending):
            self.cancel_call(timer)


class QtDialogPort:
    def __init__(self, parent):
        self.parent = parent

    def show_error(self, title, message):
        QMessageBox.critical(self.parent, title, message)

    def confirm(self, title, message):
        return QMessageBox.question(self.parent, title, message) == QMessageBox.Yes

    def confirm_save(self, title, message):
        answer = QMessageBox.question(self.parent, title, message,
                                     QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        return None if answer == QMessageBox.Cancel else answer == QMessageBox.Save

    def ask_text(self, title, prompt, *, initial=""):
        text, accepted = QInputDialog.getText(self.parent, title, prompt, text=initial)
        return text if accepted else None

    @staticmethod
    def _options(options):
        filters = ";;".join(f"{name} ({pattern})" for name,pattern in options.get("filetypes",[("All files","*")]))
        return options.get("title", "Choose file"), options.get("initialfile", ""), filters

    def open_file(self, **options):
        return QFileDialog.getOpenFileName(self.parent, *self._options(options))[0]

    def save_file(self, **options):
        path = QFileDialog.getSaveFileName(self.parent, *self._options(options))[0]
        suffix = options.get("defaultextension", "")
        from pathlib import Path
        return path+suffix if path and suffix and not Path(path).suffix else path


class QtClipboardPort:
    def copy_text(self, text):
        QApplication.clipboard().setText(text)


class QtStatusPort:
    def __init__(self, callback):
        self.callback = callback

    def publish(self, message):
        self.callback(message.text, error=message.error, diagnostic=message.diagnostic)
