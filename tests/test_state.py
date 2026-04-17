import threading

import numpy as np

from state import TrackState


def test_set_and_get_embedding():
    ts = TrackState()
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    ts.set_embedding(1, emb)
    result = ts.get_embedding(1)
    np.testing.assert_array_equal(result, emb)


def test_get_nonexistent_returns_none():
    ts = TrackState()
    assert ts.get_embedding(999) is None


def test_remove_embedding():
    ts = TrackState()
    ts.set_embedding(1, np.zeros(3, dtype=np.float32))
    ts.remove_embedding(1)
    assert ts.get_embedding(1) is None


def test_concurrent_writes_no_error():
    ts = TrackState()
    errors: list[Exception] = []

    def writer(tid: int) -> None:
        try:
            for _ in range(200):
                ts.set_embedding(tid, np.random.rand(512).astype(np.float32))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_snapshot_is_independent_copy():
    ts = TrackState()
    ts.set_embedding(1, np.ones(3, dtype=np.float32))
    emb_snap, _ = ts.snapshot()
    ts.set_embedding(1, np.zeros(3, dtype=np.float32))
    # snapshot 不受后续写入影响
    np.testing.assert_array_equal(emb_snap[1], np.ones(3, dtype=np.float32))


def test_set_and_get_name():
    ts = TrackState()
    ts.set_name(1, "Alice")
    assert ts.get_name(1) == "Alice"


def test_get_name_nonexistent_returns_none():
    ts = TrackState()
    assert ts.get_name(999) is None


def test_remove_embedding_also_removes_name():
    ts = TrackState()
    ts.set_name(1, "Alice")
    ts.set_embedding(1, np.zeros(3, dtype=np.float32))
    ts.remove_embedding(1)
    assert ts.get_name(1) is None
    assert ts.get_embedding(1) is None


def test_snapshot_returns_both_stores():
    ts = TrackState()
    ts.set_embedding(1, np.ones(3, dtype=np.float32))
    ts.set_name(1, "Alice")
    emb_snap, name_snap = ts.snapshot()
    assert 1 in emb_snap
    assert name_snap[1] == "Alice"
