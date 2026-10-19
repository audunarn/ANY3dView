"""Bounded run-monitor data from existing mesh/solver notifications."""
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import time
import numpy as np
from .live_progress import LiveProgressData
from ..post.history import Series

TERMINAL_STATES = frozenset({'completed', 'partial', 'cancelled', 'failed'})
PROGRESS_GRAPH = 'Reported work progress'


def value(payload, name, default=None):
    attribute = getattr(payload, name, None)
    if attribute is not None:
        return attribute
    getter = getattr(payload, 'get', None)
    return getter(name, default) if callable(getter) else default


def number(item):
    if item is None or isinstance(item, bool):
        return None
    try:
        result = float(item)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def duration(seconds):
    seconds = max(0., seconds)
    if seconds < 60:
        return f'{seconds:.1f} s'
    minutes, remainder = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes}m {remainder:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m {remainder:02d}s'


@dataclass
class RunProgress:
    kind: str
    job_id: str
    name: str
    created_at: float
    state: str = 'queued'
    stage: str = 'Queued'
    message: str = ''
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float = 0.
    fraction: float | None = None
    work: str = ''
    stale: bool = False
    sequence: int = 0
    graph_revision: int = 0
    max_events: int = 1200
    events: list[tuple[int, float, str]] = field(default_factory=list)
    progress_times: list[float] = field(default_factory=list)
    progress_values: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    data: LiveProgressData = field(default_factory=LiveProgressData)

    @property
    def terminal(self):
        return self.state in TERMINAL_STATES

    def elapsed(self, now):
        end = self.finished_at if self.finished_at is not None else now
        start = self.started_at if self.started_at is not None else self.created_at
        return max(0., end - start)

    def ingest(self, event, now, *, stale=False):
        if self.terminal:
            return
        self.stale = stale
        kind = event.kind
        payload = event.payload
        if kind == 'heartbeat':
            # Keep the last true solver update time, stage, metrics and graph.
            self.message = str(event.message)
            self.sequence += 1
            self.events.append((self.sequence,self.elapsed(now),f'heartbeat: {self.message}'[:4000]))
            del self.events[:-self.max_events]
            return
        text = str(event.message or value(payload, 'message', '')).strip()
        if kind == 'started':
            if self.started_at is None:
                self.started_at = now
            self.state = 'running'
        elif kind == 'cancelling' or kind == 'status' and text == 'cancelling':
            self.state = 'cancelling'
        elif kind in TERMINAL_STATES:
            self.state = kind
            self.finished_at = now
        elif kind == 'progress' and self.state == 'queued':
            self.state = 'running'
            self.started_at = now
        stage = str(value(payload, 'stage', '') or '').replace('_', ' ').strip()
        if not stage and text:
            stage = 'Newton iterations' if 'Newton iteration' in text else text.split('|')[0].split(':')[0][:70]
        if stage and stage != self.stage and not self.terminal:
            self.stage = stage
            self.fraction = None
            self.work = ''
            if self.kind=='solve':
                # Work units in different solver stages are not one total.
                self.progress_times.clear();self.progress_values.clear();self.graph_revision+=1
        fraction = number(value(payload, 'fraction'))
        completed, total = number(value(payload, 'completed')), number(value(payload, 'total'))
        if fraction is None and completed is not None and total is not None and total > 0:
            fraction = completed / total
        if fraction is not None and 0 <= fraction <= 1:
            self.fraction = fraction
            self.progress_times.append(self.elapsed(now))
            self.progress_values.append(100 * fraction)
            del self.progress_times[:-self.max_events]
            del self.progress_values[:-self.max_events]
            self.graph_revision += 1
        if completed is not None and total is not None and total > 0:
            self.work = f'{completed:g} / {total:g} work units'
        for key in ('iteration', 'load_factor', 'time_s', 'residual_norm', 'step_index'):
            item = number(value(payload, key))
            if item is not None:
                self.metrics[key] = item
        try:
            changed = self.data.ingest(text, payload)
        except (TypeError, ValueError, OverflowError):
            changed = False
            text += ' [live chart data unavailable for this notification]'
        if changed:
            self.graph_revision += 1
        # Current trial history is a display cache, separate from full job logs.
        del self.data.trial_iterations[:-self.max_events]
        del self.data.trial_residuals[:-self.max_events]
        self.message = text or kind.replace('_', ' ')
        self.updated_at = now
        self.sequence += 1
        self.events.append((self.sequence, self.elapsed(now), f'{kind}: {self.message}'[:4000]))
        del self.events[:-self.max_events]

    def series(self, choice):
        if choice == PROGRESS_GRAPH:
            if not self.progress_times:
                return None
            name=f'{choice}: {self.stage}' if self.kind=='solve' else choice
            return Series(name, np.asarray(self.progress_times), np.asarray(self.progress_values),
                          'Elapsed', 'Reported work progress', 's', '%')
        return self.data.series(choice)

    @property
    def graph_choices(self):
        return (*self.data.graph_choices, PROGRESS_GRAPH)

    def metric_text(self):
        pieces = []
        if self.data.trial_iterations:
            pieces += [f'Trial {self.data.trial_key[0]}',
                       f'Iteration {self.data.trial_iterations[-1]}',
                       f'Residual {self.data.trial_residuals[-1]:.4g}']
        else:
            for key, label in [('iteration', 'Iteration'), ('residual_norm', 'Residual')]:
                if key in self.metrics:
                    pieces.append(f'{label} {self.metrics[key]:.4g}')
        if self.data.load_factors:
            pieces.append(f'Accepted load factor {self.data.load_factors[-1]:.5g}')
            if math.isfinite(self.data.max_displacements[-1]):
                pieces.append(f'Max |u| {self.data.max_displacements[-1]:.5g} m')
            if math.isfinite(self.data.max_peeq[-1]):
                pieces.append(f'Max PEEQ {self.data.max_peeq[-1]:.5g}')
        elif 'load_factor' in self.metrics:
            pieces.append(f'Load factor {self.metrics["load_factor"]:.5g}')
        if 'time_s' in self.metrics:
            pieces.append(f'Analysis time {self.metrics["time_s"]:.5g} s')
        return '  ·  '.join(pieces)


class RunProgressCatalog:
    def __init__(self, *, clock=time.monotonic, max_finished=20):
        self.clock = clock
        self.max_finished = max_finished
        self.runs = OrderedDict()

    def ingest(self, kind, event, name, *, stale=False):
        key = (kind, event.job_id)
        now = self.clock()
        if key not in self.runs:
            self.runs[key] = RunProgress(kind, event.job_id, name, now, updated_at=now)
        run = self.runs[key]
        run.name = name
        run.ingest(event, now, stale=stale)
        finished = [key for key, run in self.runs.items() if run.terminal]
        for key in finished[:-self.max_finished] if self.max_finished else finished:
            del self.runs[key]
        return run
