"""One serial worker; owner settlement and nonblocking retirement."""
from concurrent.futures import ThreadPoolExecutor,wait
from dataclasses import dataclass
from threading import Event,Lock
import logging

@dataclass(frozen=True)
class CommittedResult:
    value: object
    source: str = ''

@dataclass(frozen=True)
class Completion:
    value: object = None
    error: BaseException | None = None
    cancelled: bool = False
    committed: bool = False
    source: str = ''
    pending: bool = False

class ResultRequests:
    def __init__(self,dispatch):
        self._dispatch=dispatch;self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='anyfem-results')
        self._lock=Lock();self._tasks={};self._records={};self._serial=0;self._latest={}
        self._source_generation=0;self._display_generation=0;self._read_generation=0;self._closed=False
        self._owner_gate=Event();self._owner_gate.set();self.outcomes=[];self.failures=[];self.committed_count=0
        from queue import SimpleQueue
        self.retirement_reports=SimpleQueue()
        self._report_pending=False
    def request(self,load,completed,*,channel='display',ordered=False):
        with self._lock:
            if self._closed:raise RuntimeError('result coordinator is closed')
            if not ordered:
                for task in self._tasks.values():
                    if task['channel']==channel and not task['ordered']:task['event'].set()
            self._serial+=1;serial=self._serial
            if not ordered:self._latest[channel]=serial
            previous=self._owner_gate;gate=self._owner_gate=Event();event=Event()
            record=dict(channel=channel,ordered=ordered,event=event,gate=gate,completed=completed,settled=False,result=None)
            self._tasks[serial]=self._records[serial]=record
            source_generation=self._source_generation;display_generation=self._display_generation;read_generation=self._read_generation
            def run():
                try:
                    while not previous.wait(.01):
                        if event.is_set() or self._closed:return Completion(cancelled=True)
                    if event.is_set():return Completion(cancelled=True)
                    value=load(event)
                    if isinstance(value,CommittedResult):return Completion(value=value.value,committed=True,source=value.source)
                    return Completion(value=value,cancelled=event.is_set())
                except BaseException as error:
                    from anypost.errors import RequestCancelled
                    return Completion(error=error,cancelled=isinstance(error,RequestCancelled) if ordered else event.is_set())
            future=self._executor.submit(run)
        def done(future):
            result=future.result()
            with self._lock:
                self._tasks.pop(serial,None)
                if not record['settled']:record['result']=result
                closed=self._closed
                if result.committed:
                    self.committed_count+=1;self.outcomes.append(result);del self.outcomes[:-10]
                if ordered and result.error is not None and not result.cancelled:
                    self.failures.append(dict(type=type(result.error).__name__,message=str(result.error)));del self.failures[:-10]
                if closed:self._records.pop(serial,None)
            def deliver():
                with self._lock:
                    if record['settled']:gate.set();return
                    record['settled']=True;self._records.pop(serial,None)
                    delivered=record.pop('result');callback=record.pop('completed')
                    stale=(self._closed or event.is_set() or source_generation!=self._source_generation or
                        (not ordered and read_generation!=self._read_generation) or
                        (channel=='display' and display_generation!=self._display_generation) or
                        (not ordered and self._latest.get(channel)!=serial))
                keep_error=ordered and delivered.error is not None and not delivered.cancelled
                try:callback(Completion(cancelled=True) if stale and not delivered.committed and not keep_error else delivered)
                finally:gate.set()
            if closed:
                if result.committed:logging.getLogger(__name__).info('Export completed from %s: %s',result.source,result.value)
                elif ordered and result.error is not None and not result.cancelled:logging.getLogger(__name__).error('Export failed: %s',result.error)
                with self._lock:
                    if ordered and not record['settled'] and self._report_pending:self.retirement_reports.put(deliver)
                gate.set();return
            try:self._dispatch(deliver)
            except BaseException:gate.set();raise
        future.add_done_callback(done);return serial
    def invalidate(self,*,source=False,reads=False):
        with self._lock:
            self._display_generation+=1
            if source:self._source_generation+=1
            if reads:self._read_generation+=1
            for task in self._tasks.values():
                if source or (reads and not task['ordered']) or task['channel']=='display':task['event'].set()
    def retire(self,finalizer=lambda:None,*,report_pending=False):
        with self._lock:
            self._report_pending=report_pending
            self._closed=True;self._source_generation+=1;self._owner_gate.set()
            records=tuple(record for record in self._records.values() if not record.get('retiring'))
            for record in records:
                record['event'].set();record['gate'].set();record['retiring']=True
                record['settled']=not (record['ordered'] and record['result'] is None)
            self._records={key:record for key,record in self._records.items() if not record['settled']}
        # Called by owner; settle callbacks NOW, never from the retiring worker.
        for record in records:
            result=record['result']
            keep_result=result is not None and (result.committed or (record['ordered'] and result.error is not None and not result.cancelled))
            try:record['completed'](result if keep_result else Completion(cancelled=True,pending=not record['settled']))
            except Exception:logging.getLogger(__name__).exception('Result retirement cleanup failed')
            finally:
                if record['settled']:
                    record['result']=None;record['completed']=None
        if not report_pending:
            from queue import Empty
            while True:
                try:self.retirement_reports.get_nowait()
                except Empty:break
        future=self._executor.submit(finalizer)
        return future
    def shutdown(self):
        """Release the retired executor after the last cleanup submission."""
        self._executor.shutdown(wait=False)
    def close(self,timeout=5.0):
        future=self.retire()
        self.shutdown()
        if not wait([future],timeout=timeout).done:raise TimeoutError('result worker has not retired')
