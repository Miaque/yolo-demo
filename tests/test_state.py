import threading
import numpy as np
import state


def test_set_and_get_embedding():
    state.clear()
    emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    state.set_embedding(1, emb)
    result = state.get_embedding(1)
    np.testing.assert_array_equal(result, emb)


def test_get_nonexistent_returns_none():
    state.clear()
    assert state.get_embedding(999) is None


def test_remove_embedding():
    state.clear()
    state.set_embedding(1, np.zeros(3, dtype=np.float32))
    state.remove_embedding(1)
    assert state.get_embedding(1) is None


def test_concurrent_writes_no_error():
    state.clear()
    errors: list[Exception] = []

    def writer(tid: int) -> None:
        try:
            for _ in range(200):
                state.set_embedding(tid, np.random.rand(512).astype(np.float32))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_snapshot_is_independent_copy():
    state.clear()
    state.set_embedding(1, np.ones(3, dtype=np.float32))
    emb_snap, _ = state.snapshot()
    state.set_embedding(1, np.zeros(3, dtype=np.float32))
    # snapshot 不受后续写入影响
    np.testing.assert_array_equal(emb_snap[1], np.ones(3, dtype=np.float32))


def test_set_and_get_name():
    state.clear()
    state.set_name(1, "Alice")
    assert state.get_name(1) == "Alice"


def test_get_name_nonexistent_returns_none():
    state.clear()
    assert state.get_name(999) is None


def test_remove_name():
    state.clear()
    state.set_name(1, "Alice")
    state.set_embedding(1, np.zeros(3, dtype=np.float32))
    state.remove_embedding(1)
    assert state.get_name(1) is None
    assert state.get_embedding(1) is None


def test_snapshot_returns_both_stores():
    state.clear()
    state.set_embedding(1, np.ones(3, dtype=np.float32))
    state.set_name(1, "Alice")
    emb_snap, name_snap = state.snapshot()
    assert 1 in emb_snap
    assert name_snap[1] == "Alice"
