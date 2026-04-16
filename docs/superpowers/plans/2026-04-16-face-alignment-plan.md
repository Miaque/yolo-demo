# Face Alignment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add face landmark alignment (affine transform) to the pipeline via a `FaceAligner` class, supporting InsightFace and FaceApiClient backends.

**Architecture:** New `pipeline/aligner.py` encapsulates landmark detection, affine transformation, and embedding extraction. `EmbedderThread` delegates to `FaceAligner` instead of calling InsightFace directly. Backend is config-driven via `ALIGNMENT_BACKEND`.

**Tech Stack:** Python 3.12, OpenCV (cv2.warpAffine, cv2.estimateAffinePartial2D), InsightFace, FaceApiClient, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-16-face-alignment-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `pipeline/aligner.py` | Create | `FaceAligner` class, `AlignmentResult`, ArcFace ref points, affine transform, feature decoding |
| `config.py` | Modify | Add `ALIGNMENT_BACKEND` to `FaceSettings` |
| `pipeline/embedder.py` | Modify | Replace `FaceAnalysis` with `FaceAligner` |
| `main.py` | Modify | Update `_load_known_faces()` to use `FaceAligner` |
| `tests/test_aligner.py` | Create | Unit tests for `FaceAligner` |
| `tests/test_embedder.py` | Modify | Update mocks for new `FaceAligner`-based flow |

---

## Chunk 1: FaceAligner Core

### Task 1: Add ALIGNMENT_BACKEND config

**Files:**
- Modify: `config.py:37-46` (FaceSettings class)

- [ ] **Step 1: Add config field**

Add `ALIGNMENT_BACKEND` to `FaceSettings`:

```python
class FaceSettings(BaseSettings):
    """人脸识别相关配置"""

    FACE_MIN_HEIGHT: int = 40  # 上半身裁剪区域最小高度（像素）
    FACE_CROP_MARGIN: float = 0.2  # InsightFace 裁剪时的外扩比例
    INSIGHTFACE_MODEL: str = "buffalo_sc"
    INSIGHTFACE_DET_SIZE: tuple[int, int] = (320, 320)
    FACES_DIR: str = "faces"
    RECOGNITION_THRESHOLD: float = 0.4
    ALIGNMENT_BACKEND: str = "insightface"  # "insightface" | "api"
```

- [ ] **Step 2: Commit**

```bash
git add config.py
git commit -m "feat: add ALIGNMENT_BACKEND config to FaceSettings"
```

---

### Task 2: Create FaceAligner with InsightFace backend — tests first

**Files:**
- Create: `tests/test_aligner.py`
- Create: `pipeline/aligner.py`

- [ ] **Step 1: Write failing tests for InsightFace backend**

Create `tests/test_aligner.py`:

```python
"""FaceAligner 单元测试。"""

import base64

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pipeline.aligner import AlignmentResult, FaceAligner, ARCFACE_REF_POINTS


class TestInsightFaceBackend:
    """InsightFace 后端测试。"""

    @patch("pipeline.aligner.FaceAnalysis")
    def test_align_returns_result_on_valid_crop(self, mock_fa_cls):
        """InsightFace 返回有效 Face → 得到 AlignmentResult。"""
        mock_app = MagicMock()
        mock_face = MagicMock()
        mock_face.kps = ARCFACE_REF_POINTS.copy()  # 关键点正好是参考点
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert isinstance(result, AlignmentResult)
        assert result.aligned_face.shape == (112, 112, 3)
        assert result.embedding.shape == (512,)
        assert result.landmarks.shape == (5, 2)
        aligner.close()

    @patch("pipeline.aligner.FaceAnalysis")
    def test_align_returns_none_on_no_face(self, mock_fa_cls):
        """InsightFace 无检测结果 → 返回 None。"""
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result is None
        aligner.close()

    @patch("pipeline.aligner.FaceAnalysis")
    def test_aligned_face_size(self, mock_fa_cls):
        """对齐图尺寸为 112x112x3。"""
        mock_app = MagicMock()
        mock_face = MagicMock()
        mock_face.kps = ARCFACE_REF_POINTS.copy()
        mock_face.embedding = np.ones(512, dtype=np.float32)
        mock_app.get.return_value = [mock_face]
        mock_fa_cls.return_value = mock_app

        aligner = FaceAligner(backend="insightface")
        crop = np.zeros((300, 300, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result.aligned_face.shape == (112, 112, 3)
        aligner.close()


class TestApiBackend:
    """API 后端测试。"""

    @patch("pipeline.aligner.FaceApiClient")
    def test_align_api_backend(self, mock_client_cls):
        """API 返回关键点和 feature → 正确解析。"""
        # 构造模拟关键点 [x0..x4, y0..y4]
        ref = ARCFACE_REF_POINTS
        points = [int(ref[i, 0]) for i in range(5)] + [int(ref[i, 1]) for i in range(5)]

        # 构造模拟 feature (base64 编码的 float32 数组)
        emb = np.ones(512, dtype=np.float32)
        feature_b64 = base64.b64encode(emb.tobytes()).decode("utf-8")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_face_data = MagicMock()
        mock_face_data.face_pos.points = points
        mock_face_data.feature = feature_b64
        mock_resp.data = [[mock_face_data]]
        mock_client = MagicMock()
        mock_client.get_face_feature.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        aligner = FaceAligner(backend="api")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert isinstance(result, AlignmentResult)
        assert result.aligned_face.shape == (112, 112, 3)
        assert result.embedding.shape == (512,)
        aligner.close()

    @patch("pipeline.aligner.FaceApiClient")
    def test_align_api_backend_no_landmarks(self, mock_client_cls):
        """API 返回空关键点 → 返回 None。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_face_data = MagicMock()
        mock_face_data.face_pos.points = []
        mock_face_data.feature = ""
        mock_resp.data = [[mock_face_data]]
        mock_client = MagicMock()
        mock_client.get_face_feature.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        aligner = FaceAligner(backend="api")
        crop = np.zeros((200, 200, 3), dtype=np.uint8)
        result = aligner.align(crop)

        assert result is None
        aligner.close()


class TestGeneral:
    """通用测试。"""

    def test_invalid_backend_raises(self):
        """无效后端字符串 → ValueError。"""
        with pytest.raises(ValueError, match="不支持的 backend"):
            FaceAligner(backend="invalid")

    def test_close_releases_resources_insightface(self):
        """InsightFace 后端 close() 不报错。"""
        with patch("pipeline.aligner.FaceAnalysis"):
            aligner = FaceAligner(backend="insightface")
            aligner.close()  # 不应抛出异常

    def test_close_releases_resources_api(self):
        """API 后端 close() 调用 FaceApiClient.close()。"""
        with patch("pipeline.aligner.FaceApiClient") as mock_cls:
            aligner = FaceAligner(backend="api")
            aligner.close()
            mock_cls.return_value.close.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_aligner.py -v`
Expected: FAIL — `pipeline.aligner` module does not exist

- [ ] **Step 3: Implement FaceAligner**

Create `pipeline/aligner.py`:

```python
# pipeline/aligner.py
"""人脸对齐器 — 基于关键点的仿射变换 + embedding 提取。"""

import base64
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from loguru import logger

from config import settings

# ArcFace 标准 5 点参考坐标 (112x112)
ARCFACE_REF_POINTS = np.array(
    [
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ],
    dtype=np.float32,
)

ALIGNMENT_SIZE = (112, 112)


@dataclass(frozen=True)
class AlignmentResult:
    """对齐结果。"""

    aligned_face: np.ndarray  # 仿射对齐后的人脸图像 (112x112x3)
    embedding: np.ndarray  # 512-dim embedding 向量
    landmarks: np.ndarray  # 5 个人脸关键点 (5x2)


def _align_face(crop: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """根据关键点做仿射对齐，返回 112x112 的标准化人脸图像。"""
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_REF_POINTS)
    aligned = cv2.warpAffine(crop, M, ALIGNMENT_SIZE)
    return aligned


def _parse_api_points(points: list[int]) -> np.ndarray | None:
    """解析 API 返回的关键点为 (5, 2) 数组。

    Layout: [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4]
    顺序: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
    """
    if len(points) < 10:
        return None
    xs = np.array(points[:5], dtype=np.float32)
    ys = np.array(points[5:], dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def _decode_feature(feature_str: str) -> np.ndarray | None:
    """解码 API feature 字符串为 embedding 向量。"""
    try:
        raw = base64.b64decode(feature_str)
        embedding = np.frombuffer(raw, dtype=np.float32)
        if embedding.size == 0:
            return None
        return embedding.copy()  # frombuffer 返回只读数组，需要可写副本
    except Exception:
        logger.warning("feature 解码失败")
        return None


class FaceAligner:
    """人脸对齐器，根据配置使用 InsightFace 或 FaceApiClient。

    Backend:
    - "insightface": InsightFace FaceAnalysis 获取关键点和 embedding
    - "api": FaceApiClient 获取关键点和 embedding
    """

    def __init__(self, backend: Literal["insightface", "api"] | None = None) -> None:
        if backend is None:
            backend = settings.ALIGNMENT_BACKEND  # type: ignore[assignment]
        if backend not in ("insightface", "api"):
            raise ValueError(f"不支持的 backend: {backend!r}，可选值为 'insightface' 或 'api'")

        self._backend = backend

        if backend == "insightface":
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=settings.INSIGHTFACE_DET_SIZE)
        else:
            from pipeline.face_api import FaceApiClient

            self._client = FaceApiClient()

    def align(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """对齐人脸裁剪图并提取 embedding。检测失败返回 None。"""
        if self._backend == "insightface":
            return self._align_insightface(face_crop)
        return self._align_api(face_crop)

    def close(self) -> None:
        """释放资源。"""
        if self._backend == "api":
            self._client.close()

    def _align_insightface(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """InsightFace 后端：获取关键点和 embedding，仿射对齐用于 debug。"""
        try:
            results = self._app.get(face_crop)
        except Exception:
            logger.exception("InsightFace 检测失败")
            return None

        if not results:
            return None

        face = results[0]
        landmarks = np.array(face.kps, dtype=np.float32)
        if landmarks.shape != (5, 2):
            return None

        aligned = _align_face(face_crop, landmarks)
        return AlignmentResult(
            aligned_face=aligned,
            embedding=face.embedding,
            landmarks=landmarks,
        )

    def _align_api(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """API 后端：通过 FaceApiClient 获取关键点和 embedding。"""
        try:
            _, buf = cv2.imencode(".jpg", face_crop)
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            resp = self._client.get_face_feature(img_b64)
        except Exception:
            logger.exception("FaceApiClient 调用失败")
            return None

        if resp.status_code != 200 or not resp.data or not resp.data[0]:
            return None

        face_data = resp.data[0][0]
        landmarks = _parse_api_points(face_data.face_pos.points)
        if landmarks is None:
            return None

        embedding = _decode_feature(face_data.feature)
        if embedding is None:
            return None

        aligned = _align_face(face_crop, landmarks)
        return AlignmentResult(
            aligned_face=aligned,
            embedding=embedding,
            landmarks=landmarks,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aligner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/aligner.py tests/test_aligner.py
git commit -m "feat: add FaceAligner with InsightFace and API backends"
```

---

## Chunk 2: Integrate FaceAligner into Pipeline

### Task 3: Refactor EmbedderThread to use FaceAligner — tests first

**Files:**
- Modify: `tests/test_embedder.py`
- Modify: `pipeline/embedder.py`

- [ ] **Step 1: Write failing tests for refactored EmbedderThread**

Update `tests/test_embedder.py`. The tests currently patch `pipeline.embedder.FaceAnalysis`. After refactoring, they need to patch `pipeline.embedder.FaceAligner` instead:

```python
import queue
import threading
import time
import numpy as np
from unittest.mock import MagicMock, patch
import state


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

    state.clear()

    mock_aligner = _make_mock_aligner(embedding=np.ones(512, dtype=np.float32))
    mock_aligner_cls.return_value = mock_aligner

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


@patch("pipeline.embedder.FaceAligner")
def test_empty_result_not_stored(mock_aligner_cls):
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_aligner = _make_mock_aligner(embedding=None)
    mock_aligner_cls.return_value = mock_aligner

    crop_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    thread = EmbedderThread(crop_q, stop)

    crop_q.put((99, np.zeros((100, 100, 3), dtype=np.uint8)))

    thread.start()
    time.sleep(0.3)
    stop.set()
    thread.join(timeout=2)

    assert state.get_embedding(99) is None


@patch("pipeline.embedder.FaceAligner")
def test_exception_in_align_does_not_crash_thread(mock_aligner_cls):
    from pipeline.embedder import EmbedderThread

    state.clear()

    mock_aligner = MagicMock()
    mock_aligner.align.side_effect = RuntimeError("align error")
    mock_aligner_cls.return_value = mock_aligner

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

    known_emb = np.ones(512, dtype=np.float32)
    known_faces = {"TestPerson": known_emb.copy()}

    mock_aligner = _make_mock_aligner(embedding=known_emb.copy())

    with patch("pipeline.embedder.FaceAligner", return_value=mock_aligner):
        crop_q: queue.Queue = queue.Queue()
        stop = threading.Event()
        thread = EmbedderThread(crop_q, stop, known_faces=known_faces)

        crop_q.put((42, np.zeros((100, 100, 3), dtype=np.uint8)))
        thread.start()
        time.sleep(0.3)
        stop.set()
        thread.join(timeout=2)

    assert state.get_name(42) == "TestPerson"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embedder.py -v`
Expected: FAIL — EmbedderThread still uses FaceAnalysis, not FaceAligner

- [ ] **Step 3: Refactor EmbedderThread**

Replace `pipeline/embedder.py` with:

```python
# pipeline/embedder.py
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from config import settings
from pipeline.aligner import FaceAligner
import state


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
        known_faces: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
        self.known_faces = known_faces or {}
        self._aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)
        self._align_count = 0

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

            # DEBUG: 保存对齐后的标准化人脸图
            debug_dir = Path("output/debug_aligned")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"align_{track_id}_{self._align_count}.jpg"
            cv2.imwrite(str(debug_path), result.aligned_face)
            logger.info(
                "DEBUG saved aligned: {} shape={}", debug_path, result.aligned_face.shape
            )
            self._align_count += 1

            # 存储 embedding 和匹配名字
            state.set_embedding(track_id, result.embedding)
            name = match_name(result.embedding, self.known_faces)
            logger.info(
                "track_id={} matched='{}' emb_norm={:.4f}",
                track_id,
                name,
                float(np.linalg.norm(result.embedding)),
            )
            state.set_name(track_id, name)
        except Exception:
            logger.exception("FaceAligner failed for track_id={}", track_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embedder.py tests/test_aligner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add pipeline/embedder.py tests/test_embedder.py
git commit -m "refactor: replace InsightFace with FaceAligner in EmbedderThread"
```

---

## Chunk 3: Update Known Faces Loading and Main Entry Point

### Task 4: Update _load_known_faces to use FaceAligner

**Files:**
- Modify: `main.py:33-65` (`_load_known_faces` function)
- Modify: `main.py:68-98` (`run` function, where embedder is created)

- [ ] **Step 1: Refactor `_load_known_faces` to accept FaceAligner**

Replace `_load_known_faces` in `main.py`:

```python
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
```

- [ ] **Step 2: Update `run()` to create FaceAligner before loading known faces**

Update the top of `run()` in `main.py`. Add `FaceAligner` import and restructure initialization:

```python
# main.py imports — add FaceAligner
from pipeline.aligner import FaceAligner

def run() -> None:
    stop_event = threading.Event()

    frame_queue: queue.Queue = queue.Queue(maxsize=settings.FRAME_QUEUE_SIZE)
    face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=settings.OUTPUT_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()

    # 创建 FaceAligner 并用于加载已知人脸，确保 embedding 空间一致
    known_face_aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)
    known_faces = _load_known_faces(known_face_aligner)
    known_face_aligner.close()

    output_path = f"{settings.OUTPUT_DIR}/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    reader = ReaderThread(frame_queue, stop_event)
    embedder = EmbedderThread(face_crop_queue, stop_event, known_faces=known_faces)
    writer = WriterThread(output_queue, stop_event, output_path=output_path)

    # ... rest of run() unchanged ...
```

Note: `known_face_aligner` is closed after loading known faces. The `EmbedderThread` creates its own `FaceAligner` internally. This is intentional — the aligner is not shared between threads.

- [ ] **Step 3: Remove unused InsightFace import from `_load_known_faces`**

The old `_load_known_faces` imported `FaceAnalysis` locally. After refactoring, that import is no longer needed in `main.py`. Verify no other references to `FaceAnalysis` exist in `main.py` — there shouldn't be.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 5: Run linter**

Run: `uv run ruff check .`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "refactor: update _load_known_faces to use FaceAligner"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 2: Run linter**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No errors

- [ ] **Step 3: Verify no stale imports**

Ensure `main.py` no longer imports `FaceAnalysis` directly.
Ensure `pipeline/embedder.py` no longer imports `FaceAnalysis` directly (only `FaceAligner`).

```bash
grep -n "FaceAnalysis" main.py pipeline/embedder.py
```

Expected: No matches (or only in comments)
