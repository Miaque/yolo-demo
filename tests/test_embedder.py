import queue
import threading
import time
import numpy as np
from unittest.mock import MagicMock, patch
from state import TrackState


def _make_mock_aligner(embedding: np.ndarray | None = None, aligned_face: np.ndarray | None = None):
    """创建模拟 FaceAligner，返回给定的 embedding。"""
    from pipeline.aligner import AlignmentResult, ARCFACE_REF_POINTS

    aligner = MagicMock()
    if embedding is not None:
        result = AlignmentResult(
            aligned_face=aligned_face if aligned_face is not None else np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=embedding,
            landmarks=ARCFACE_REF_POINTS.copy(),
        )
        aligner.align.return_value = result
    else:
        aligner.align.return_value = None
    return aligner


@patch("pipeline.embedder.FaceAligner")
def test_embedding_stored_on_success(mock_aligner_cls):
    from pipeline.embedder import EmbedderThread

    ts = TrackState()
    mock_aligner = _make_mock_aligner(embedding=np.ones(512, dtype=np.float32))
    mock_aligner_cls.return_value = mock_aligner

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop, ts)

    crop_q.put((42, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    result = ts.get_embedding(42)
    assert result is not None
    np.testing.assert_array_equal(result, np.ones(512, dtype=np.float32))


@patch("pipeline.embedder.FaceAligner")
def test_empty_result_not_stored(mock_aligner_cls):
    from pipeline.embedder import EmbedderThread

    ts = TrackState()
    mock_aligner = _make_mock_aligner(embedding=None)
    mock_aligner_cls.return_value = mock_aligner

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop, ts)

    crop_q.put((99, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    assert ts.get_embedding(99) is None


@patch("pipeline.embedder.FaceAligner")
def test_exception_in_align_does_not_crash_thread(mock_aligner_cls):
    from pipeline.embedder import EmbedderThread

    ts = TrackState()
    mock_aligner = MagicMock()
    mock_aligner.align.side_effect = RuntimeError("align error")
    mock_aligner_cls.return_value = mock_aligner

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop, ts)

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

    query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    result = match_name(query, known_faces, threshold=0.5)
    assert result == "Alice"


def test_cosine_no_match_returns_unknown():
    """当相似度低于阈值时，返回 Unknown。"""
    from pipeline.embedder import match_name

    known_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    known_faces = {"Alice": known_emb}

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

    ts = TrackState()
    known_emb = np.ones(512, dtype=np.float32)
    known_faces = {"TestPerson": known_emb.copy()}

    mock_aligner = _make_mock_aligner(embedding=known_emb.copy())

    with patch("pipeline.embedder.FaceAligner", return_value=mock_aligner):
        crop_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        thread = EmbedderThread(crop_q, stop, ts, known_faces=known_faces)

        crop_q.put((42, np.zeros((100, 100, 3), dtype=np.uint8)))
        thread.start()
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=2)

    assert ts.get_name(42) == "TestPerson"
