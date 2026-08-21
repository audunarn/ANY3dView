from threading import Thread

import pytest

from any3dview import ViewerScheduler


def test_scheduler_hands_cross_thread_payload_to_owner_thread():
    scheduler = ViewerScheduler()
    values = []
    producer = Thread(target=scheduler.submit, args=(values.append, ("frame", 7)))
    producer.start()
    producer.join()

    assert values == []
    assert scheduler.drain() == 1
    assert values == [("frame", 7)]


def test_closed_scheduler_rejects_new_work():
    scheduler = ViewerScheduler()
    scheduler.close()
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.submit(lambda: None)
