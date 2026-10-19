"""Toolkit-neutral compatibility facade over the persistent job manager."""
class JobWorkerFacade:
    """Compatibility surface while the GUI uses the persistent JobManager."""

    POLL_MS = 60

    def __init__(self, manager) -> None:
        self.manager = manager

    @property
    def running(self) -> bool:
        return self.manager.running or bool(self.manager.queued)

    def cancel(self) -> None:
        job_id = self.manager.active_job_id
        if job_id is not None:
            self.manager.cancel(job_id)

    def stop(self) -> None:
        self.cancel()
