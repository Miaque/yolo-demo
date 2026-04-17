# Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor main.py and state.py to make the codebase maintainable and testable — replace global singleton with injectable class, extract Pipeline class from flat function, remove debug disk writes.

**Architecture:** Convert `state.py` from module-level functions/globals to an instantiable `TrackState` class injected via constructors. Replace `main.py:run()` with a `Pipeline` class that owns lifecycle, queues, and main loop. Remove unconditional debug disk writes from `embedder.py`.

**Tech Stack:** Python 3.12, pytest, threading, numpy, loguru

**Spec:** `docs/superpowers/specs/2026-04-17-pipeline-refactor-design.md`

---

## Chunk 1: TrackState class + tests

### Task 1: Convert state.py to TrackState class

**Files:**
- Modify: `state.py` (full rewrite)
- Modify: `tests/test_state.py` (full rewrite)

- [ ] **Step 1: Rewrite state.py as TrackState class**

Replace entire file content:

```python
# state.py
import threading
from typing import Optional

import numpy as np


class TrackState:
    """线程安全的跟踪状态，存储 track 的 embedding 和匹配名字。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._embeddings.get(track_id)

    def set_embedding(self, track_id: int, embedding: np.ndarray) -> None:
        with self._lock:
            self._embeddings[track_id] = embedding

    def remove_embedding(self, track_id: int) -> None:
        """移除 embedding 及其关联的名字。"""
        with self._lock:
            self._embeddings.pop(track_id, None)
            self._names.pop(track_id, None)

    def get_name(self, track_id: int) -> Optional[str]:
        with self._lock:
            return self._names.get(track_id)

    def set_name(self, track_id: int, name: str) -> None:
        with self._lock:
            self._names[track_id] = name

    def snapshot(self) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """返回两个 store 的浅拷贝，用于主线程传递给 overlay。"""
        with self._lock:
            return dict(self._embeddings), dict(self._names)
```

- [ ] **Step 2: Rewrite tests/test_state.py to use TrackState instances**

Replace entire file content:

```python
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
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: All 9 tests PASS

- [ ] **Step 4: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "refactor: convert state.py to injectable TrackState class"
```

---

## Chunk 2: EmbedderThread injection + debug removal

### Task 2: Update EmbedderThread to accept TrackState via constructor

**Files:**
- Modify: `pipeline/embedder.py` (constructor injection, remove debug code, remove unused imports)
- Modify: `tests/test_embedder.py` (adapt to TrackState injection)

- [ ] **Step 1: Update tests/test_embedder.py first (TDD red)**

Replace entire file content:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail (red)**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: FAIL — `EmbedderThread.__init__` does not accept `track_state` positional arg yet.

- [ ] **Step 3: Update pipeline/embedder.py**

Replace entire file content:

```python
# pipeline/embedder.py
import queue
import threading

import numpy as np
from loguru import logger

from config import settings
from state import TrackState
from pipeline.aligner import FaceAligner


def match_name(
    embedding: np.ndarray,
    known_faces: dict[str, np.ndarray],
    threshold: float = settings.RECOGNITION_THRESHOLD,
) -> str:
    """将 embedding 与已知人脸比较，返回最匹配的名字或 'Unknown'。"""
    if not known_faces:
        return "Unknown"

    best_name = "Unknown"
    best_sim = threshold
    scores: dict[str, float] = {}
    for name, known_emb in known_faces.items():
        sim = float(
            np.dot(embedding, known_emb)
            / (np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-10)
        )
        scores[name] = sim
        if sim > best_sim:
            best_sim = sim
            best_name = name
    logger.info("match_name scores={} best='{}' threshold={}", scores, best_name, threshold)
    return best_name


class EmbedderThread(threading.Thread):
    """后台线程：从 face_crop_queue 取人脸裁剪图，通过 FaceAligner 对齐并提取 Embedding。"""

    def __init__(
        self,
        face_crop_queue: queue.Queue,
        stop_event: threading.Event,
        track_state: TrackState,
        known_faces: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
        self.track_state = track_state
        self.known_faces = known_faces or {}
        self._aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    track_id, face_crop = self.face_crop_queue.get(timeout=0.5)
                    self._process(track_id, face_crop)
                except queue.Empty:
                    continue
        finally:
            self._aligner.close()

    def _process(self, track_id: int, face_crop: np.ndarray) -> None:
        try:
            result = self._aligner.align(face_crop)
            if result is None:
                return

            # 存储 embedding 和匹配名字
            self.track_state.set_embedding(track_id, result.embedding)
            name = match_name(result.embedding, self.known_faces)
            logger.info(
                "track_id={} matched='{}' emb_norm={:.4f}",
                track_id,
                name,
                float(np.linalg.norm(result.embedding)),
            )
            self.track_state.set_name(track_id, name)
        except Exception:
            logger.exception("FaceAligner failed for track_id={}", track_id)
```

Key changes from current:
- Removed `from pathlib import Path` import
- Removed `import cv2` import
- Removed `import state`
- Added `from state import TrackState`
- Constructor: added `track_state: TrackState` positional arg before `known_faces`
- Removed `self._align_count = 0`
- Removed debug disk write block (lines 73-81 in old file)
- `state.set_embedding(...)` → `self.track_state.set_embedding(...)`
- `state.set_name(...)` → `self.track_state.set_name(...)`

- [ ] **Step 4: Run tests to verify they pass (green)**

Run: `uv run pytest tests/test_embedder.py tests/test_state.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/embedder.py tests/test_embedder.py
git commit -m "refactor: inject TrackState into EmbedderThread, remove debug disk writes"
```

---

## Chunk 3: Update test_integration.py

### Task 3: Adapt integration tests to TrackState

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Update test_integration.py**

Replace entire file content:

```python
"""
集成冒烟测试：合成帧 → 主循环逻辑 → overlay 输出
不依赖真实 RTSP 流、YOLO 模型或 InsightFace。
"""
import queue
import threading
import numpy as np
from unittest.mock import MagicMock, patch
from state import TrackState
import overlay
from config import settings


def _make_frame() -> np.ndarray:
    return np.zeros((settings.INPUT_HEIGHT, settings.INPUT_WIDTH, 3), dtype=np.uint8)


@patch("pipeline.tracker.ByteTrack")
@patch("pipeline.detector.YOLO")
def test_pipeline_loop_processes_frames(mock_yolo_cls, mock_bt_cls):
    """主循环处理 10 帧不抛出异常，output_queue 收到标注帧。"""
    from pipeline.detector import FaceDetector
    from pipeline.tracker import FaceTracker

    # Mock YOLO: 无检测结果
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    # Mock ByteTrack: 无跟踪结果
    mock_bt = MagicMock()
    mock_bt.update.return_value = np.empty((0, 8), dtype=np.float32)
    mock_bt_cls.return_value = mock_bt

    ts = TrackState()
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
    face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    detector = FaceDetector()
    tracker = FaceTracker()

    submitted_ids: set[int] = set()
    track_results: list = []
    frame_count = 0

    # 注入 10 帧合成帧
    for _ in range(10):
        frame_queue.put(_make_frame())

    # 模拟主循环逻辑
    while not frame_queue.empty():
        frame = frame_queue.get_nowait()
        frame_count += 1

        if frame_count % settings.DETECT_INTERVAL == 0:
            detections = detector.detect(frame)
            track_results = tracker.update(detections, frame)
        else:
            track_results = tracker.predict()

        emb_snapshot, name_snapshot = ts.snapshot()
        annotated = overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)
        output_queue.put_nowait(annotated)

    assert output_queue.qsize() == 10
    # 所有输出帧形状正确
    while not output_queue.empty():
        f = output_queue.get_nowait()
        assert f.shape == (settings.INPUT_HEIGHT, settings.INPUT_WIDTH, 3)


@patch("pipeline.tracker.ByteTrack")
@patch("pipeline.detector.YOLO")
def test_new_track_id_queued_for_embedding(mock_yolo_cls, mock_bt_cls):
    """新 track_id 出现时，人脸裁剪应进入 face_crop_queue。"""
    from pipeline.detector import FaceDetector
    from pipeline.tracker import FaceTracker

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    mock_bt = MagicMock()
    # 返回一个 track: id=1, bbox=(50,50,150,150)
    mock_bt.update.return_value = np.array(
        [[50.0, 50.0, 150.0, 150.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    ts = TrackState()
    face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()
    submitted_ids: set[int] = set()

    frame = _make_frame()
    # 触发检测帧
    detections = detector.detect(frame)
    track_results = tracker.update(detections, frame)

    for track_id, bbox in track_results:
        if track_id not in submitted_ids:
            submitted_ids.add(track_id)
            x1, y1, x2, y2 = bbox
            face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                              max(0, x1):min(frame.shape[1], x2)].copy()
            try:
                face_crop_queue.put_nowait((track_id, face_crop))
            except queue.Full:
                pass

    assert face_crop_queue.qsize() == 1
    queued_id, queued_crop = face_crop_queue.get_nowait()
    assert queued_id == 1
    assert queued_crop.ndim == 3
```

Key changes:
- `import state` → `from state import TrackState`
- `state.clear()` → removed (fresh `ts = TrackState()` per test)
- `known_ids` → `submitted_ids`
- `emb_snapshot = state.snapshot()` → `emb_snapshot, name_snapshot = ts.snapshot()` (properly unpack both values; old code passed the tuple as a single arg by mistake, losing name overlay)
- `overlay.draw_tracks(frame, track_results, emb_snapshot)` → `overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)` (bug fix: properly pass both dicts)

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "refactor: adapt integration tests to TrackState injection"
```

---

## Chunk 4: Pipeline class

### Task 4: Replace run() with Pipeline class

**Files:**
- Modify: `main.py` (full rewrite of `run()` and `__main__`)

- [ ] **Step 1: Rewrite main.py**

Replace the `run()` function (lines 70-166) and the `__main__` block (lines 169-171) with the `Pipeline` class. The module-level functions `_crop_with_margin` (lines 21-31) and `_load_known_faces` (lines 34-67) stay unchanged.

New main.py structure:

```python
# main.py
import queue
import signal
import threading
import time
from datetime import datetime
import cv2
import numpy as np
from loguru import logger
from config import settings
import overlay
from state import TrackState
from pipeline.detector import FaceDetector
from pipeline.embedder import EmbedderThread
from pipeline.reader import ReaderThread
from pipeline.tracker import FaceTracker
from pipeline.writer import WriterThread
from pipeline.aligner import FaceAligner


def _crop_with_margin(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """BBox 周围外扩 FACE_CROP_MARGIN 比例，裁剪人脸区域。"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * settings.FACE_CROP_MARGIN), int(bh * settings.FACE_CROP_MARGIN)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    return frame[cy1:cy2, cx1:cx2].copy()


def _load_known_faces(aligner: FaceAligner) -> dict[str, np.ndarray]:
    """启动时从 faces/ 目录加载已知人脸 embedding，使用与管线相同的 FaceAligner。"""
    from pathlib import Path

    faces_dir = Path(settings.FACES_DIR)
    if not faces_dir.is_dir():
        logger.warning("Faces directory '{}' not found, all faces will be Unknown", settings.FACES_DIR)
        return {}

    known_faces: dict[str, np.ndarray] = {}
    for img_path in sorted(faces_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: {}", img_path)
            continue

        result = aligner.align(img)
        if result is None:
            logger.warning("No face detected in: {}", img_path)
            continue

        name = img_path.stem
        known_faces[name] = result.embedding
        logger.info(
            "Loaded known face: {} | emb_norm={:.4f} | emb_shape={}",
            name,
            float(np.linalg.norm(result.embedding)),
            result.embedding.shape,
        )

    logger.info("Total known faces loaded: {}", len(known_faces))
    return known_faces


class Pipeline:
    """人脸检测管线：组装所有组件，管理生命周期。"""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._track_state = TrackState()
        self._submitted_ids: set[int] = set()
        self._frame_count = 0

    def run(self) -> None:
        self._build()
        self._register_signals()
        self._start_threads()
        try:
            self._loop()
        finally:
            self._shutdown()

    def _build(self) -> None:
        """初始化队列、检测器、跟踪器、工作线程。"""
        self._frame_queue: queue.Queue = queue.Queue(maxsize=settings.FRAME_QUEUE_SIZE)
        self._face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)
        self._output_queue: queue.Queue = queue.Queue(maxsize=settings.OUTPUT_QUEUE_SIZE)

        self._detector = FaceDetector()
        self._tracker = FaceTracker()

        # 创建 FaceAligner 并用于加载已知人脸，确保 embedding 空间一致
        known_face_aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)
        try:
            known_faces = _load_known_faces(known_face_aligner)
        finally:
            known_face_aligner.close()

        output_path = f"{settings.OUTPUT_DIR}/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

        self._reader = ReaderThread(self._frame_queue, self._stop_event)
        self._embedder = EmbedderThread(
            self._face_crop_queue, self._stop_event,
            self._track_state, known_faces=known_faces,
        )
        self._writer = WriterThread(self._output_queue, self._stop_event, output_path=output_path)

    def _register_signals(self) -> None:
        def _handle_signal(sig, _frame) -> None:
            logger.info("Signal {} received, shutting down…", sig)
            self._stop_event.set()

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

    def _start_threads(self) -> None:
        self._reader.start()
        self._embedder.start()
        self._writer.start()
        logger.info(
            "Pipeline started. Input: {}, Output: {}",
            settings.RTSP_INPUT,
            self._writer.output_path,
        )

    def _loop(self) -> None:
        track_results: list[tuple[int, list[int]]] = []

        while not self._stop_event.is_set():
            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                if self._frame_count > 0 and self._frame_count % 50 == 0:
                    logger.info("Main: frame_queue empty after {} frames", self._frame_count)
                continue

            self._step(frame, track_results)

    def _step(
        self,
        frame: np.ndarray,
        track_results: list[tuple[int, list[int]]],
    ) -> None:
        self._frame_count += 1
        if self._frame_count % 100 == 0:
            logger.info(
                "Main: frame {} | frame_q={} crop_q={} out_q={}",
                self._frame_count, self._frame_queue.qsize(),
                self._face_crop_queue.qsize(), self._output_queue.qsize(),
            )

        if self._frame_count % settings.DETECT_INTERVAL == 0:
            t0 = time.monotonic()
            detections = self._detector.detect(frame)
            t1 = time.monotonic()
            track_results.clear()
            track_results.extend(self._tracker.update(detections, frame))
            t2 = time.monotonic()
            if self._frame_count <= 20 or self._frame_count % 100 == 0:
                logger.info(
                    "Main: frame {} detect={:.0f}ms track={:.0f}ms dets={}",
                    self._frame_count, (t1 - t0) * 1000, (t2 - t1) * 1000,
                    len(detections),
                )

            for removed_id in self._tracker.removed_ids():
                self._submitted_ids.discard(removed_id)
                self._track_state.remove_embedding(removed_id)
        else:
            track_results.clear()
            track_results.extend(self._tracker.predict())

        for track_id, bbox in track_results:
            if track_id not in self._submitted_ids:
                self._submitted_ids.add(track_id)
                face_crop = _crop_with_margin(frame, bbox)
                try:
                    self._face_crop_queue.put_nowait((track_id, face_crop))
                except queue.Full:
                    pass  # 队列已满，跳过此帧

        emb_snapshot, name_snapshot = self._track_state.snapshot()
        annotated = overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)
        try:
            self._output_queue.put_nowait(annotated)
        except queue.Full:
            logger.warning("Main: output_queue full, dropping frame {}", self._frame_count)

    def _shutdown(self) -> None:
        self._stop_event.set()
        for t in (self._reader, self._embedder, self._writer):
            t.join(timeout=5)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    Pipeline().run()
```

Key changes from current `main.py`:
- Removed `import state` → added `from state import TrackState`
- `run()` function → `Pipeline` class with methods
- `known_ids` → `self._submitted_ids`
- `state.remove_embedding(...)` → `self._track_state.remove_embedding(...)`
- `state.snapshot()` → `self._track_state.snapshot()`
- `_loop()` passes `track_results` as local variable to `_step()` (avoids mutable instance state for per-frame data)

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS (no test directly imports `main.run()`)

- [ ] **Step 3: Lint check**

Run: `uv run ruff check main.py state.py pipeline/embedder.py`
Expected: No errors

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "refactor: extract Pipeline class from run() function"
```

---

## Chunk 5: Final verification

### Task 5: Full verification and cleanup

- [ ] **Step 1: Run complete test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

- [ ] **Step 2: Run lint on all changed files**

Run: `uv run ruff check state.py main.py pipeline/embedder.py tests/test_state.py tests/test_embedder.py tests/test_integration.py`
Expected: No errors

- [ ] **Step 3: Run format check**

Run: `uv run ruff format --check state.py main.py pipeline/embedder.py tests/test_state.py tests/test_embedder.py tests/test_integration.py`
Expected: No errors (all files already formatted)

- [ ] **Step 4: Verify no remaining `import state` or `state.` references**

Run: `grep -rn "\bimport state\b" --include="*.py" . ; grep -rn "\bstate\." --include="*.py" . | grep -v "track_state"`
Expected: No matches (all global state references eliminated)
