import queue
import threading
import time
import numpy as np
from unittest.mock import MagicMock, patch
import state


@patch("pipeline.embedder.FaceAnalysis")
def test_embedding_stored_on_success(mock_fa_cls):
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_app = MagicMock()
    mock_face = MagicMock()
    mock_face.embedding = np.ones(512, dtype=np.float32)
    mock_app.get.return_value = [mock_face]
    mock_fa_cls.return_value = mock_app

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop)

    crop_q.put((42, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    result = state.get_embedding(42)
    assert result is not None
    np.testing.assert_array_equal(result, np.ones(512, dtype=np.float32))


@patch("pipeline.embedder.FaceAnalysis")
def test_empty_result_not_stored(mock_fa_cls):
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_app = MagicMock()
    mock_app.get.return_value = []
    mock_fa_cls.return_value = mock_app

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop)

    crop_q.put((99, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    assert state.get_embedding(99) is None


@patch("pipeline.embedder.FaceAnalysis")
def test_exception_in_get_does_not_crash_thread(mock_fa_cls):
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_app = MagicMock()
    mock_app.get.side_effect = RuntimeError("model error")
    mock_fa_cls.return_value = mock_app

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop)

    crop_q.put((7, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    assert not thread.is_alive()  # 线程正常退出，未崩溃
