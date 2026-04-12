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


def test_cosine_match_identifies_known_face():
    """当 embedding 与已知人脸匹配时，存储对应名字。"""
    from pipeline.embedder import match_name

    known_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    known_faces = {"Alice": known_emb}

    # 相同方向 → 匹配
    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    result = match_name(query, known_faces, threshold=0.5)
    assert result == "Alice"


def test_cosine_no_match_returns_unknown():
    """当相似度低于阈值时，返回 Unknown。"""
    from pipeline.embedder import match_name

    known_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    known_faces = {"Alice": known_emb}

    # 几乎正交 → 不匹配
    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    result = match_name(query, known_faces, threshold=0.5)
    assert result == "Unknown"


def test_match_name_empty_known():
    """known_faces 为空时返回 Unknown。"""
    from pipeline.embedder import match_name

    query = np.ones(3, dtype=np.float32)
    assert match_name(query, {}, threshold=0.5) == "Unknown"


def test_embedder_stores_name_on_match():
    """EmbedderThread 处理后应同时存储 embedding 和 name。"""
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_app = MagicMock()
    mock_face = MagicMock()
    known_emb = np.ones(512, dtype=np.float32)
    mock_face.embedding = known_emb.copy()
    mock_app.get.return_value = [mock_face]

    known_faces = {"TestPerson": known_emb.copy()}

    with patch("pipeline.embedder.FaceAnalysis", return_value=mock_app):
        crop_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        thread = EmbedderThread(crop_q, stop, known_faces=known_faces)

        crop_q.put((42, np.zeros((100, 100, 3), dtype=np.uint8)))
        thread.start()
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=2)

    assert state.get_name(42) == "TestPerson"
