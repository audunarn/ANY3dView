"""UI-thread run dashboard; workers retain computation and cancellation."""
import numpy as np
from PySide6.QtCore import QTimer, QSignalBlocker, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QProgressBar, QPlainTextEdit, QPushButton, QCheckBox, QSplitter, QTabWidget)
from ...presentation.run_progress import RunProgressCatalog, PROGRESS_GRAPH, duration
from .plot import HistoryPlot


class QtRunMonitor(QWidget):
    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.document_id = app.project.document_id
        self.session = app.session
        self.catalog = RunProgressCatalog()
        self._selection = None
        self._log_sequence = 0
        self._plot_revision = None
        self._manual_graph = False
        self._dirty = False
        self._pending_follow = None
        self._cancel_requested = set()
        layout = QVBoxLayout(self)
        header = QHBoxLayout(); layout.addLayout(header)
        self.jobs = QComboBox(); self.jobs.setMinimumContentsLength(20)
        self.jobs.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        header.addWidget(self.jobs,1)
        self.follow = QCheckBox('Follow active run'); self.follow.setChecked(True); header.addWidget(self.follow)
        self.cancel = QPushButton('Cancel run'); self.cancel.clicked.connect(self.cancel_selected); header.addWidget(self.cancel)
        self.jobs.currentIndexChanged.connect(self.select_run)
        self.jobs.activated.connect(lambda *_: self.follow.setChecked(False))
        splitter = QSplitter(Qt.Horizontal); layout.addWidget(splitter)
        summary = QWidget(); details = QVBoxLayout(summary)
        self.state = QLabel('No runs yet'); self.state.setStyleSheet('font-weight: 600; font-size: 14px;')
        self.stage = QLabel('Start a mesh or analysis to see live progress.'); self.stage.setWordWrap(True)
        self.message = QLabel(); self.message.setWordWrap(True)
        self.message.setMaximumHeight(70)
        self.bar = QProgressBar(); self.bar.setRange(0,1000); self.bar.setValue(0)
        self.bar.setFormat('No run')
        self.timing = QLabel(); self.timing.setWordWrap(True)
        self.metrics = QLabel(); self.metrics.setWordWrap(True)
        self.identity = QLabel(); self.identity.setTextInteractionFlags(Qt.TextSelectableByMouse); self.identity.setWordWrap(True)
        for widget in (self.state,self.stage,self.bar,self.message,self.timing,self.metrics,self.identity):details.addWidget(widget)
        details.addStretch(); splitter.addWidget(summary)
        self.tabs = QTabWidget(); splitter.addWidget(self.tabs)
        chart = QWidget(); chart_layout = QVBoxLayout(chart)
        controls = QHBoxLayout(); chart_layout.addLayout(controls)
        self.graph = QComboBox(); controls.addWidget(self.graph,1)
        self.pause = QCheckBox('Pause graph'); controls.addWidget(self.pause)
        self.plot = HistoryPlot(); self.plot.empty_text = 'Waiting for numeric progress from the worker'
        chart_layout.addWidget(self.plot)
        self.latest = QLabel(); self.latest.setWordWrap(True); chart_layout.addWidget(self.latest)
        for widget in (self.state,self.stage,self.message,self.timing,self.metrics,self.identity,self.latest):widget.setTextFormat(Qt.PlainText)
        self.tabs.addTab(chart,'Live graph')
        log = QWidget(); log_layout = QVBoxLayout(log)
        self.transcript = QPlainTextEdit(); self.transcript.setReadOnly(True)
        self.transcript.document().setMaximumBlockCount(1200); log_layout.addWidget(self.transcript)
        copy = QPushButton('Copy displayed log'); copy.clicked.connect(lambda:self.app.clipboard.copy_text(self.transcript.toPlainText()))
        log_layout.addWidget(copy); self.tabs.addTab(log,'Run log')
        splitter.setSizes([340,660]); splitter.setStretchFactor(1,1)
        self.graph.activated.connect(self.choose_graph)
        self.graph.currentTextChanged.connect(lambda *_:self.invalidate_plot())
        self.pause.toggled.connect(lambda *_:self.invalidate_plot())
        self.timer = QTimer(self); self.timer.setInterval(150); self.timer.timeout.connect(self.refresh); self.timer.start()
        self.cancel.setEnabled(False)

    def bind_document(self):
        if self.document_id == self.app.project.document_id and self.session is self.app.session:
            return
        self.document_id = self.app.project.document_id
        self.session = self.app.session
        self.catalog.runs.clear(); self.jobs.clear(); self._selection = None
        self.graph.clear();self._manual_graph=False
        self.transcript.clear(); self.plot.show_series([]); self.latest.clear()
        self.state.setText('No runs yet'); self.stage.setText('Start a mesh or analysis to see live progress.')
        self.state.setStyleSheet('font-weight: 600; font-size: 14px; color: #64748b;')
        self.message.clear(); self.timing.clear(); self.metrics.clear(); self.identity.clear()
        self.bar.setRange(0,1000); self.bar.setValue(0); self.cancel.setEnabled(False)
        self.bar.setFormat('No run')
        self._pending_follow=None;self._plot_revision=None;self._log_sequence=0
        self._cancel_requested.clear()

    def observe(self, kind, event, name, *, stale=False):
        self.bind_document()
        key = (kind,event.job_id)
        self.catalog.ingest(kind,event,name,stale=stale)
        if event.kind in {'queued','started'}:self._pending_follow=key
        self._dirty = True

    def sync_jobs(self):
        selected = self.current_key()
        keys=tuple(self.catalog.runs)
        with QSignalBlocker(self.jobs):
            existing=tuple(tuple(self.jobs.itemData(i)) for i in range(self.jobs.count()))
            if existing!=keys:
                self.jobs.clear()
                for key in keys:self.jobs.addItem('',key)
            for index,((source,identifier),run) in enumerate(self.catalog.runs.items()):
                text=f'{source.capitalize()} · {run.name} · {run.state}'
                if self.jobs.itemText(index)!=text:self.jobs.setItemText(index,text)
            wanted=self._pending_follow if self.follow.isChecked() and self._pending_follow is not None else selected
            if wanted is None and keys:wanted=keys[-1]
            for index in range(self.jobs.count()):
                if tuple(self.jobs.itemData(index)) == wanted:self.jobs.setCurrentIndex(index);break
        self._pending_follow=None

    def current_key(self):
        key = self.jobs.currentData()
        return None if key is None else tuple(key)

    def select_run(self,*_):
        self._selection = None; self._dirty = True
        self.refresh()

    def choose_graph(self,*_):
        self._manual_graph = True; self.invalidate_plot()

    def invalidate_plot(self):
        self._plot_revision = None; self._dirty = True

    def refresh(self):
        if self.app._closing:return
        self.sync_jobs()
        key = self.current_key();run = self.catalog.runs.get(key)
        if run is None:return
        if key != self._selection:
            self._selection = key; self._log_sequence = 0; self.transcript.clear()
            self._plot_revision = None; self._manual_graph = False
            self.plot.show_series([]);self.latest.setText('Graph updates paused.' if self.pause.isChecked() else '')
        now = self.catalog.clock()
        self.state.setText(f'{run.state.capitalize()}' + (' · inputs changed' if run.stale else ''))
        color={'completed':'#15803d','failed':'#b91c1c','partial':'#a16207','cancelled':'#64748b'}.get(run.state,'#2563eb')
        self.state.setStyleSheet(f'font-weight: 600; font-size: 14px; color: {color};')
        self.stage.setText(run.stage); self.message.setText(run.message[:400]+('…' if len(run.message)>400 else ''))
        self.identity.setText(f'{run.kind.capitalize()} · {run.name}\nJob {run.job_id}')
        elapsed = duration(run.elapsed(now))
        label = 'Queued for' if run.started_at is None else 'Elapsed'
        age = '' if run.terminal else f' · Last update {duration(now-run.updated_at)} ago'
        self.timing.setText(f'{label} {elapsed}{age}')
        self.metrics.setText(run.metric_text())
        if run.terminal:self._cancel_requested.discard(key)
        self.cancel.setEnabled(not run.terminal and run.state!='cancelling' and key not in self._cancel_requested)
        if not run.terminal and (run.fraction is None or run.fraction >= 1.):
            self.bar.setRange(0,0); self.bar.setFormat('Run still active')
        else:
            self.bar.setRange(0,1000)
            fraction = 1. if run.state=='completed' else run.fraction or 0.
            self.bar.setValue(round(1000*fraction))
            self.bar.setFormat(run.state.capitalize() if run.terminal else f'{100*fraction:.0f}% reported · {run.work}'.rstrip(' ·'))
        lines = [f'[{duration(elapsed)}] {text}' for sequence,elapsed,text in run.events if sequence>self._log_sequence]
        if lines:self.transcript.appendPlainText('\n'.join(lines));self._log_sequence=run.sequence
        revision = (key,run.graph_revision,self.graph.currentText())
        if not self.pause.isChecked() and self._plot_revision!=revision:
            choice = self.graph.currentText()
            choices = run.graph_choices
            if tuple(self.graph.itemText(i) for i in range(self.graph.count()))!=choices:
                with QSignalBlocker(self.graph):self.graph.clear();self.graph.addItems(list(choices));self.graph.setCurrentText(choice)
            available = [choice for choice in choices if (series:=run.series(choice)) is not None and np.isfinite(series.y).any()]
            for index in range(self.graph.count()):self.graph.model().item(index).setEnabled(self.graph.itemText(index) in available)
            if not self._manual_graph and available:
                preferred = PROGRESS_GRAPH if run.kind=='mesh' and PROGRESS_GRAPH in available else available[0]
                with QSignalBlocker(self.graph):self.graph.setCurrentText(preferred)
            series = run.series(self.graph.currentText())
            self.plot.y_limits=(0.,100.) if self.graph.currentText()==PROGRESS_GRAPH else None
            self.plot.show_series([] if series is None else [series])
            finite = [] if series is None else np.flatnonzero(np.isfinite(series.x)&np.isfinite(series.y))
            if len(finite):
                index=finite[-1];unit=f' {series.y_unit}' if series.y_unit else ''
                self.latest.setText(f'Latest: {series.y[index]:.5g}{unit} · {series.x_label} {series.x[index]:.5g} {series.x_unit}'.rstrip())
            else:self.latest.setText('This run has not reported this metric yet.')
            self._plot_revision=(key,run.graph_revision,self.graph.currentText())
        self._dirty=False

    def cancel_selected(self):
        key = self.current_key()
        if key is None:return
        kind,identifier = key
        manager = self.app.mesh_task_manager if kind=='mesh' else self.app.job_manager
        if manager.cancel(identifier):
            self._cancel_requested.add(key)
            self.cancel.setEnabled(False)
            self.app.set_status(f'Cancellation requested for {kind} {identifier[:8]}.')
        else:
            self.app.set_status('The run is already finished or cancellation is already pending.')

    def stop(self):
        self.timer.stop()
