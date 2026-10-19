"""Qt file inspector: async reads, source records, diagnostics and canonical FEM."""
import json
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog,QVBoxLayout,QPushButton,QPlainTextEdit,QLabel
from ...presentation.file_inspection import inspect_file
from .table import ResultTable


class FileInspector(QDialog):
    def __init__(self,app,path=None):
        super().__init__(app);self.app=app;self.result=None;self._epoch=0;self._closed=False
        self._future=None;self._poll_call=None;self.export_buttons=[]
        self.setAttribute(Qt.WA_DeleteOnClose);self.setWindowTitle("File inspector");self.resize(900,700)
        layout=QVBoxLayout(self);self.status=QLabel("Open a structural file to inspect it");layout.addWidget(self.status)
        self.summary=QPlainTextEdit();self.summary.setReadOnly(True);layout.addWidget(self.summary)
        self.records=ResultTable();layout.addWidget(self.records);self.diagnostics=ResultTable();layout.addWidget(self.diagnostics)
        for label,callback in [("Open file",self.open),("Save JSON report",self.save_report),("Write canonical FEM",self.canonicalize)]:
            button=QPushButton(label);button.clicked.connect(app.guarded(callback));layout.addWidget(button)
            if callback != self.open:
                button.setEnabled(False);self.export_buttons.append(button)
        if path:self.load(path)

    def open(self):
        path=self.app.dialogs.open_file(filetypes=[("Structural files","*.fem *.FEM *.sif *.SIF *.inp *.frd *.dat")])
        if path:self.load(path)

    def load(self,path):
        if self._closed or self.app._closing:return
        self._cancel_read()
        self._epoch+=1;epoch=self._epoch;self.result=None;self.status.setText(f"Reading {path}")
        self.summary.clear()
        self.records.show_rows(["Record","Source lines","First values"],[])
        self.diagnostics.show_rows(["Severity","Code","Line","Message"],[])
        for button in self.export_buttons:button.setEnabled(False)
        future=self.app._artifact_executor.submit(inspect_file,path)
        self._future=future
        def poll():
            self._poll_call=None
            if self._closed or self.app._closing or epoch!=self._epoch:return
            if not future.done():self._poll_call=self.app.scheduler.call_later(50,poll);return
            self._future=None
            def apply():
                try:self.result=future.result()
                except Exception as error:
                    self.status.setText(f"Could not read {path}: {error}")
                    self.diagnostics.show_rows(["Severity","Code","Line","Message"],[("error","READ","",str(error))])
                    raise
                self.summary.setPlainText(json.dumps(self.result.summary,indent=2,default=str))
                self.records.show_rows(["Record","Source lines","First values"],self.result.records)
                self.diagnostics.show_rows(["Severity","Code","Line","Message"],[(item.severity,item.code,item.line_start or "",item.message) for item in self.result.diagnostics])
                total=self.result.summary.get("records",len(self.result.records))
                self.status.setText(f"{self.result.kind}: {len(self.result.diagnostics)} diagnostics; {len(self.result.records)} of {total} records previewed")
                self.export_buttons[0].setEnabled(True)
                self.export_buttons[1].setEnabled(self.result.document is not None)
            self.app.guarded(apply)()
        self._poll_call=self.app.scheduler.call_later(50,poll)

    def _cancel_read(self):
        if self._poll_call is not None:
            self.app.scheduler.cancel_call(self._poll_call);self._poll_call=None
        if self._future is not None:
            # A running owner read cannot be interrupted. Epoch checks suppress
            # its late presentation; cancel only work that is still queued.
            self._future.cancel();self._future=None

    def save_report(self):
        if self.result is None:raise ValueError("Open a file first")
        path=self.app.dialogs.save_file(filetypes=[("JSON report","*.json")],defaultextension=".json")
        if path:Path(path).write_text(json.dumps(self.result.report(),indent=2,default=str)+"\n",encoding="utf-8")

    def canonicalize(self):
        if self.result is None or self.result.document is None:raise ValueError("Open a SESAM FEM/SIF document first")
        path=self.app.dialogs.save_file(filetypes=[("SESAM FEM","*.FEM")],defaultextension=".FEM")
        if path:
            from anyfileio.sesam.exporter import write_sesam_fem_document
            report=write_sesam_fem_document(self.result.document,path,overwrite=True)
            self.status.setText(f"Wrote {report.records_written} records ({report.bytes_written} bytes)")

    def closeEvent(self,event):
        self._closed=True;self._epoch+=1;self._cancel_read();super().closeEvent(event)
