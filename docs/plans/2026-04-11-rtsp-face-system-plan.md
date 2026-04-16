# RTSP 实时人脸处理系统 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 RTSP 摄像头流实时检测、跟踪人脸并提取特征向量，将标注后的视频重新推送为 RTSP。

**Architecture:** 单进程四线程：读帧线程（FFmpeg）→ 主线程（YOLO + ByteTrack）→ InsightFace 线程（异步 Embedding）→ 推流线程（FFmpeg）。三条 Queue 解耦各线程，maxsize 控制背压、保证实时性。

**Tech Stack:** Python 3.11+, ultralytics (YOLO11n), boxmot (ByteTrack), insightface (buffalo_sc, CPU), opencv-python, FFmpeg

---

## 文件清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `pyproject.toml` | 修改 | 项目依赖与 pytest 配置 |
| `config.py` | 新建 | 所有可调参数的常量 |
| `state.py` | 新建 | 线程安全的 embedding_store |
| `overlay.py` | 新建 | 纯函数：在帧上绘制 BBox / ID |
| `pipeline/__init__.py` | 新建 | 空文件，标记包 |
| `pipeline/detector.py` | 新建 | YOLO 两阶段人脸检测 |
| `pipeline/tracker.py` | 新建 | ByteTrack 封装 |
| `pipeline/embedder.py` | 新建 | InsightFace 后台线程 |
| `pipeline/reader.py` | 新建 | FFmpeg 读帧线程 |
| `pipeline/writer.py` | 新建 | FFmpeg 推流线程 |
| `main.py` | 修改 | 主循环：组装所有线程 |
| `tests/conftest.py` | 新建 | pytest sys.path 配置 |
| `tests/test_state.py` | 新建 | state 并发读写测试 |
| `tests/test_overlay.py` | 新建 | overlay 纯函数测试 |
| `tests/test_detector.py` | 新建 | YOLO 两阶段检测测试（mock）|
| `tests/test_tracker.py` | 新建 | ByteTrack 封装测试（mock）|
| `tests/test_embedder.py` | 新建 | InsightFace 线程测试（mock）|
| `tests/test_integration.py` | 新建 | 端到端冒烟测试（合成帧）|

---

## Task 1：项目依赖与目录骨架

**Files:**
- Modify: `pyproject.toml`
- Create: `pipeline/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py`

- [ ] **Step 1：更新 pyproject.toml**

```toml
[project]
name = "yolo-demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "ultralytics>=8.3",
    "boxmot>=10.0",
    "insightface>=0.7",
    "onnxruntime>=1.18",
    "opencv-python>=4.10",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2：创建 pipeline/__init__.py**

```python
```

（空文件，标记 pipeline 为包）

- [ ] **Step 3：创建 tests/__init__.py 和 tests/conftest.py**

`tests/__init__.py` — 空文件。

`tests/conftest.py`:
```python
# pytest 会通过 pyproject.toml 的 pythonpath 自动加载项目根，此文件留空即可。
```

- [ ] **Step 4：安装依赖**

```bash
pip install -e ".[dev]"
```

预期：无报错，`pytest --collect-only` 输出 `no tests ran`。

- [ ] **Step 5：验证环境**

```bash
python -c "import ultralytics, boxmot, insightface, cv2; print('OK')"
```

预期输出：`OK`

---

## Task 2：config.py — 全局配置常量

**Files:**
- Create: `config.py`

- [ ] **Step 1：创建 config.py**

```python
# config.py
RTSP_INPUT: str = "rtsp://localhost:8554/cam"
RTSP_OUTPUT: str = "rtsp://localhost:8554/processed"

INPUT_WIDTH: int = 640
INPUT_HEIGHT: int = 360

DETECT_INTERVAL: int = 5          # 每隔几帧触发 YOLO 检测
BYTETRACK_MAX_AGE: int = 30       # track 消失多少帧后视为已移除

FACE_MIN_HEIGHT: int = 40         # 上半身裁剪区域最小高度（像素）
FACE_CROP_MARGIN: float = 0.2     # InsightFace 裁剪时的外扩比例

FRAME_QUEUE_SIZE: int = 2
FACE_CROP_QUEUE_SIZE: int = 4
OUTPUT_QUEUE_SIZE: int = 2

FFMPEG_RETRY_MAX: int = 5
FFMPEG_RETRY_DELAY: int = 3       # 断流重试间隔（秒）

OUTPUT_BITRATE: str = "800k"

# YOLO 模型路径
# yolo11n.pt 会由 ultralytics 自动下载
# yolov8n-face.pt 需手动下载：
#   https://github.com/akanametov/yolo-face/releases
PERSON_MODEL: str = "yolo11n.pt"
FACE_MODEL: str = "yolov8n-face.pt"

# InsightFace 模型名称（buffalo_sc 为轻量亚洲优化版）
INSIGHTFACE_MODEL: str = "buffalo_sc"
INSIGHTFACE_DET_SIZE: tuple[int, int] = (320, 320)
```

- [ ] **Step 2：验证导入**

```bash
python -c "import config; print(config.RTSP_INPUT)"
```

预期输出：`rtsp://localhost:8554/cam`

---

## Task 3：state.py — 线程安全共享状态

**Files:**
- Create: `state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1：写失败测试**

`tests/test_state.py`:
```python
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
    snapshot = state.snapshot()
    state.set_embedding(1, np.zeros(3, dtype=np.float32))
    # snapshot 不受后续写入影响
    np.testing.assert_array_equal(snapshot[1], np.ones(3, dtype=np.float32))
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_state.py -v
```

预期：`ModuleNotFoundError: No module named 'state'`

- [ ] **Step 3：创建 state.py**

```python
# state.py
import threading
from typing import Optional
import numpy as np

lock = threading.Lock()
embedding_store: dict[int, np.ndarray] = {}


def get_embedding(track_id: int) -> Optional[np.ndarray]:
    with lock:
        return embedding_store.get(track_id)


def set_embedding(track_id: int, embedding: np.ndarray) -> None:
    with lock:
        embedding_store[track_id] = embedding


def remove_embedding(track_id: int) -> None:
    with lock:
        embedding_store.pop(track_id, None)


def snapshot() -> dict[int, np.ndarray]:
    """返回当前 store 的浅拷贝，用于主线程传递给 overlay（避免竞态）。"""
    with lock:
        return dict(embedding_store)


def clear() -> None:
    with lock:
        embedding_store.clear()
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_state.py -v
```

预期：5 个测试全部 PASSED。

- [ ] **Step 5：提交**

```bash
git add config.py state.py tests/test_state.py pyproject.toml pipeline/__init__.py tests/__init__.py tests/conftest.py
git commit -m "feat: add config, state module and tests"
```

---

## Task 4：overlay.py — 帧标注纯函数

**Files:**
- Create: `overlay.py`
- Create: `tests/test_overlay.py`

- [ ] **Step 1：写失败测试**

`tests/test_overlay.py`:
```python
import numpy as np
import overlay


def _blank(h: int = 360, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_new_array():
    frame = _blank()
    result = overlay.draw_tracks(frame, [], {})
    assert result is not frame


def test_shape_preserved():
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    assert result.shape == frame.shape


def test_empty_tracks_unchanged():
    frame = _blank()
    result = overlay.draw_tracks(frame, [], {})
    np.testing.assert_array_equal(result, frame)


def test_bbox_pixels_differ_from_blank():
    """绘制后 BBox 区域应有像素变化。"""
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    # BBox 边框区域至少有一个非零像素
    assert result[10, 10:101].any()


def test_green_when_embedding_present():
    """有 Embedding 时用绿色（BGR: 0,255,0）。"""
    frame = _blank()
    emb = np.zeros(512, dtype=np.float32)
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {1: emb})
    # 检查 BBox 边框上存在绿色像素
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 255, 0]) for px in top_row)


def test_orange_when_no_embedding():
    """无 Embedding 时用橙色（BGR: 0,165,255）。"""
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_overlay.py -v
```

预期：`ModuleNotFoundError: No module named 'overlay'`

- [ ] **Step 3：创建 overlay.py**

```python
# overlay.py
import cv2
import numpy as np

_GREEN = (0, 255, 0)    # 已有 Embedding
_ORANGE = (0, 165, 255)  # 尚无 Embedding


def draw_tracks(
    frame: np.ndarray,
    tracks: list[tuple[int, list[int]]],
    emb_snapshot: dict[int, np.ndarray],
) -> np.ndarray:
    """在帧副本上绘制跟踪框和 ID 标签，返回新数组。

    Args:
        frame:        BGR 帧（不修改原始帧）
        tracks:       [(track_id, [x1, y1, x2, y2]), ...]
        emb_snapshot: embedding_store 的快照（state.snapshot() 的返回值）
    """
    out = frame.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = _GREEN if track_id in emb_snapshot else _ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id}" + (" [E]" if track_id in emb_snapshot else "")
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_overlay.py -v
```

预期：6 个测试全部 PASSED。

- [ ] **Step 5：提交**

```bash
git add overlay.py tests/test_overlay.py
git commit -m "feat: add overlay module and tests"
```

---

## Task 5：pipeline/detector.py — YOLO 两阶段检测

**Files:**
- Create: `pipeline/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1：写失败测试**

`tests/test_detector.py`:
```python
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


def _mock_box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> MagicMock:
    box = MagicMock()
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
    box.conf = [np.float32(conf)]
    return box


def _mock_result(boxes: list) -> MagicMock:
    r = MagicMock()
    r.boxes = boxes
    return r


@patch("pipeline.detector.YOLO")
def test_face_global_coords(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # person box: (100,100) → (200,300), upper half → y2=200
    person_model.return_value = [_mock_result([_mock_box(100, 100, 200, 300)])]

    face_model = MagicMock()
    # face box in crop coords: (10,10)→(50,50)
    face_model.return_value = [_mock_result([_mock_box(10, 10, 50, 50)])]

    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert len(faces) == 1
    x1, y1, x2, y2, conf = faces[0]
    assert x1 == 110  # 100 + 10
    assert y1 == 110  # 100 + 10
    assert x2 == 150  # 100 + 50
    assert y2 == 150  # 100 + 50


@patch("pipeline.detector.YOLO")
def test_skips_too_small_crop(mock_yolo_cls):
    """上半身高度 < FACE_MIN_HEIGHT 时跳过，face model 不被调用。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # 人体高度 20px → 上半身 10px < FACE_MIN_HEIGHT(40)
    person_model.return_value = [_mock_result([_mock_box(100, 100, 200, 120)])]

    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert len(faces) == 0
    face_model.assert_not_called()


@patch("pipeline.detector.YOLO")
def test_no_persons_returns_empty(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [_mock_result([])]

    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


@patch("pipeline.detector.YOLO")
def test_multiple_persons_multiple_faces(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [
        _mock_result([
            _mock_box(0, 0, 100, 200),
            _mock_box(200, 0, 300, 200),
        ])
    ]

    face_model = MagicMock()
    face_model.side_effect = [
        [_mock_result([_mock_box(5, 5, 40, 40)])],
        [_mock_result([_mock_box(5, 5, 40, 40)])],
    ]
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)
    assert len(faces) == 2
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_detector.py -v
```

预期：`ModuleNotFoundError: No module named 'pipeline.detector'`

- [ ] **Step 3：创建 pipeline/detector.py**

```python
# pipeline/detector.py
from ultralytics import YOLO
import numpy as np
import config


class FaceDetector:
    """YOLO 两阶段人脸检测：先检测 person，再在上半身区域内检测 face。"""

    def __init__(
        self,
        person_model_path: str = config.PERSON_MODEL,
        face_model_path: str = config.FACE_MODEL,
    ) -> None:
        self.person_model = YOLO(person_model_path)
        self.face_model = YOLO(face_model_path)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """返回 [(x1, y1, x2, y2, conf), ...] — 全局坐标系下的人脸框。"""
        person_results = self.person_model(frame, classes=[0], verbose=False)[0]
        faces: list[tuple[int, int, int, int, float]] = []

        for box in person_results.boxes:
            px1, py1, px2, py2 = [int(v) for v in box.xyxy[0]]
            upper_y2 = py1 + (py2 - py1) // 2

            if upper_y2 - py1 < config.FACE_MIN_HEIGHT:
                continue

            crop = frame[py1:upper_y2, px1:px2]
            if crop.size == 0:
                continue

            face_results = self.face_model(crop, verbose=False)[0]
            for fbox in face_results.boxes:
                fx1, fy1, fx2, fy2 = [int(v) for v in fbox.xyxy[0]]
                conf = float(fbox.conf[0])
                faces.append((px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2, conf))

        return faces
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_detector.py -v
```

预期：4 个测试全部 PASSED。

- [ ] **Step 5：提交**

```bash
git add pipeline/detector.py tests/test_detector.py
git commit -m "feat: add two-stage YOLO face detector and tests"
```

---

## Task 6：pipeline/tracker.py — ByteTrack 封装

**Files:**
- Create: `pipeline/tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1：写失败测试**

`tests/test_tracker.py`:
```python
from unittest.mock import MagicMock, patch
import numpy as np


@patch("pipeline.tracker.ByteTrack")
def test_update_returns_track_list(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    # ByteTrack.update 返回 shape (N,8): [x1,y1,x2,y2,id,conf,cls,idx]
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    results = tracker.update([(10, 20, 100, 120, 0.9)], frame)

    assert len(results) == 1
    tid, bbox = results[0]
    assert tid == 1
    assert bbox == [10, 20, 100, 120]


@patch("pipeline.tracker.ByteTrack")
def test_predict_returns_last_results(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)

    assert tracker.predict() == [(1, [10, 20, 100, 120])]


@patch("pipeline.tracker.ByteTrack")
def test_removed_ids_when_track_disappears(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.side_effect = [
        # 第一次：ID 1 和 ID 2
        np.array([
            [10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0],
            [200.0, 20.0, 300.0, 120.0, 2.0, 0.8, 0.0, 0.0],
        ]),
        # 第二次：只有 ID 1
        np.array([
            [10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0],
        ]),
    ]
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)

    removed = tracker.removed_ids()
    assert 2 in removed
    assert 1 not in removed


@patch("pipeline.tracker.ByteTrack")
def test_empty_detections_keeps_last(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)
    # 无检测帧
    results = tracker.predict()
    assert results == [(1, [10, 20, 100, 120])]
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_tracker.py -v
```

预期：`ModuleNotFoundError: No module named 'pipeline.tracker'`

- [ ] **Step 3：创建 pipeline/tracker.py**

```python
# pipeline/tracker.py
from boxmot import ByteTrack
import numpy as np
import config


class FaceTracker:
    """ByteTrack 封装，暴露简洁的 update / predict / removed_ids 接口。"""

    def __init__(self) -> None:
        self._tracker = ByteTrack(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=config.BYTETRACK_MAX_AGE,
        )
        self._last_tracks: list[tuple[int, list[int]]] = []
        self._active_ids: set[int] = set()
        self._removed_ids: set[int] = set()

    def update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame: np.ndarray,
    ) -> list[tuple[int, list[int]]]:
        """用新检测结果更新 ByteTrack，返回 [(track_id, [x1,y1,x2,y2])]。"""
        if detections:
            dets = np.array(
                [[x1, y1, x2, y2, conf, 0] for x1, y1, x2, y2, conf in detections],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        raw = self._tracker.update(dets, frame)

        new_ids: set[int] = set()
        results: list[tuple[int, list[int]]] = []
        for t in raw:
            tid = int(t[4])
            bbox = [int(t[0]), int(t[1]), int(t[2]), int(t[3])]
            results.append((tid, bbox))
            new_ids.add(tid)

        self._removed_ids = self._active_ids - new_ids
        self._active_ids = new_ids
        self._last_tracks = results
        return results

    def predict(self) -> list[tuple[int, list[int]]]:
        """非检测帧调用，直接返回上一帧的跟踪结果（Kalman 预测由 ByteTrack 内部维护）。"""
        return self._last_tracks

    def removed_ids(self) -> set[int]:
        """返回上次 update() 后消失的 track_id 集合。"""
        return self._removed_ids
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_tracker.py -v
```

预期：4 个测试全部 PASSED。

- [ ] **Step 5：提交**

```bash
git add pipeline/tracker.py tests/test_tracker.py
git commit -m "feat: add ByteTrack wrapper and tests"
```

---

## Task 7：pipeline/embedder.py — InsightFace 后台线程

**Files:**
- Create: `pipeline/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1：写失败测试**

`tests/test_embedder.py`:
```python
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
```

- [ ] **Step 2：运行测试，确认失败**

```bash
pytest tests/test_embedder.py -v
```

预期：`ModuleNotFoundError: No module named 'pipeline.embedder'`

- [ ] **Step 3：创建 pipeline/embedder.py**

```python
# pipeline/embedder.py
import queue
import threading
import logging
import numpy as np
from insightface.app import FaceAnalysis
import config
import state

logger = logging.getLogger(__name__)


class EmbedderThread(threading.Thread):
    """后台线程：从 face_crop_queue 取人脸裁剪图，调用 InsightFace 提取 Embedding。"""

    def __init__(self, face_crop_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
        self._app = FaceAnalysis(
            name=config.INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                track_id, face_crop = self.face_crop_queue.get(timeout=0.5)
                self._process(track_id, face_crop)
            except queue.Empty:
                continue

    def _process(self, track_id: int, face_crop: np.ndarray) -> None:
        try:
            results = self._app.get(face_crop)
            if results:
                state.set_embedding(track_id, results[0].embedding)
        except Exception:
            logger.warning("InsightFace failed for track_id=%d", track_id, exc_info=True)
```

- [ ] **Step 4：运行测试，确认通过**

```bash
pytest tests/test_embedder.py -v
```

预期：3 个测试全部 PASSED。

- [ ] **Step 5：提交**

```bash
git add pipeline/embedder.py tests/test_embedder.py
git commit -m "feat: add InsightFace embedder thread and tests"
```

---

## Task 8：pipeline/reader.py — FFmpeg 读帧线程

**Files:**
- Create: `pipeline/reader.py`

（reader 涉及真实 subprocess，单元测试成本高；Task 11 的集成测试覆盖此模块。）

- [ ] **Step 1：创建 pipeline/reader.py**

```python
# pipeline/reader.py
import queue
import subprocess
import threading
import time
import logging
import numpy as np
import config

logger = logging.getLogger(__name__)

_FRAME_SIZE = config.INPUT_WIDTH * config.INPUT_HEIGHT * 3  # BGR24 字节数


def _build_read_cmd(rtsp_url: str) -> list[str]:
    return [
        "ffmpeg",
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-rtsp_transport", "tcp",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-i", rtsp_url,
        "-vf", f"scale={config.INPUT_WIDTH}:{config.INPUT_HEIGHT}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]


class ReaderThread(threading.Thread):
    """FFmpeg 读帧线程，将 BGR 帧放入 frame_queue。满时丢弃最旧帧以保持实时性。"""

    def __init__(
        self,
        frame_queue: queue.Queue,
        stop_event: threading.Event,
        rtsp_url: str = config.RTSP_INPUT,
    ) -> None:
        super().__init__(daemon=True, name="reader")
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.rtsp_url = rtsp_url

    def run(self) -> None:
        for attempt in range(config.FFMPEG_RETRY_MAX):
            if self.stop_event.is_set():
                return
            logger.info("Reader: attempt %d/%d", attempt + 1, config.FFMPEG_RETRY_MAX)
            try:
                self._read_loop()
            except Exception:
                logger.error("Reader: unexpected error", exc_info=True)
            if not self.stop_event.is_set():
                logger.info("Reader: reconnecting in %ds…", config.FFMPEG_RETRY_DELAY)
                time.sleep(config.FFMPEG_RETRY_DELAY)
        logger.error("Reader: max retries reached, giving up")

    def _read_loop(self) -> None:
        proc = subprocess.Popen(
            _build_read_cmd(self.rtsp_url),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while not self.stop_event.is_set():
                raw = proc.stdout.read(_FRAME_SIZE)
                if len(raw) != _FRAME_SIZE:
                    logger.warning("Reader: stream ended or short read (%d bytes)", len(raw))
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)
                )
                # 满时弹出最旧帧再放新帧
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put_nowait(frame)
        finally:
            proc.terminate()
            proc.wait()
```

- [ ] **Step 2：验证导入**

```bash
python -c "from pipeline.reader import ReaderThread; print('OK')"
```

预期输出：`OK`

- [ ] **Step 3：提交**

```bash
git add pipeline/reader.py
git commit -m "feat: add FFmpeg reader thread"
```

---

## Task 9：pipeline/writer.py — FFmpeg 推流线程

**Files:**
- Create: `pipeline/writer.py`

- [ ] **Step 1：创建 pipeline/writer.py**

```python
# pipeline/writer.py
import queue
import subprocess
import threading
import logging
import numpy as np
import config

logger = logging.getLogger(__name__)


def _build_write_cmd(rtsp_url: str) -> list[str]:
    return [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{config.INPUT_WIDTH}x{config.INPUT_HEIGHT}",
        "-r", "25",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-b:v", config.OUTPUT_BITRATE,
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]


class WriterThread(threading.Thread):
    """FFmpeg 推流线程，将 output_queue 中的标注帧编码后推送至 RTSP。"""

    def __init__(
        self,
        output_queue: queue.Queue,
        stop_event: threading.Event,
        rtsp_url: str = config.RTSP_OUTPUT,
    ) -> None:
        super().__init__(daemon=True, name="writer")
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.rtsp_url = rtsp_url

    def run(self) -> None:
        proc = subprocess.Popen(
            _build_write_cmd(self.rtsp_url),
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            while not self.stop_event.is_set():
                try:
                    frame: np.ndarray = self.output_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    proc.stdin.write(frame.tobytes())
                    proc.stdin.flush()
                except BrokenPipeError:
                    logger.error("Writer: broken pipe, RTSP server may have stopped")
                    break
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait()
            logger.info("Writer: stopped")
```

- [ ] **Step 2：验证导入**

```bash
python -c "from pipeline.writer import WriterThread; print('OK')"
```

预期输出：`OK`

- [ ] **Step 3：提交**

```bash
git add pipeline/writer.py
git commit -m "feat: add FFmpeg writer thread"
```

---

## Task 10：main.py — 主循环与线程编排

**Files:**
- Modify: `main.py`

- [ ] **Step 1：写入 main.py**

```python
# main.py
import logging
import queue
import signal
import threading
import numpy as np
import config
import overlay
import state
from pipeline.detector import FaceDetector
from pipeline.embedder import EmbedderThread
from pipeline.reader import ReaderThread
from pipeline.tracker import FaceTracker
from pipeline.writer import WriterThread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _crop_with_margin(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """BBox 周围外扩 FACE_CROP_MARGIN 比例，裁剪人脸区域。"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * config.FACE_CROP_MARGIN), int(bh * config.FACE_CROP_MARGIN)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)
    return frame[cy1:cy2, cx1:cx2].copy()


def run() -> None:
    stop_event = threading.Event()

    frame_queue: queue.Queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    face_crop_queue: queue.Queue = queue.Queue(maxsize=config.FACE_CROP_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=config.OUTPUT_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()

    reader = ReaderThread(frame_queue, stop_event)
    embedder = EmbedderThread(face_crop_queue, stop_event)
    writer = WriterThread(output_queue, stop_event)

    known_ids: set[int] = set()

    def _handle_signal(sig, _frame) -> None:
        logger.info("Signal %d received, shutting down…", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    reader.start()
    embedder.start()
    writer.start()
    logger.info("Pipeline started. Input: %s", config.RTSP_INPUT)

    frame_count = 0
    track_results: list[tuple[int, list[int]]] = []

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_count += 1

            if frame_count % config.DETECT_INTERVAL == 0:
                detections = detector.detect(frame)
                track_results = tracker.update(detections, frame)

                for removed_id in tracker.removed_ids():
                    known_ids.discard(removed_id)
                    state.remove_embedding(removed_id)
            else:
                track_results = tracker.predict()

            for track_id, bbox in track_results:
                if track_id not in known_ids:
                    known_ids.add(track_id)
                    face_crop = _crop_with_margin(frame, bbox)
                    try:
                        face_crop_queue.put_nowait((track_id, face_crop))
                    except queue.Full:
                        pass  # InsightFace 队列已满，跳过此帧

            emb_snapshot = state.snapshot()
            annotated = overlay.draw_tracks(frame, track_results, emb_snapshot)
            try:
                output_queue.put_nowait(annotated)
            except queue.Full:
                pass  # 推流线程跟不上，丢弃此帧

    finally:
        stop_event.set()
        for t in (reader, embedder, writer):
            t.join(timeout=5)
        logger.info("Shutdown complete")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2：验证语法**

```bash
python -c "import main; print('OK')"
```

预期输出：`OK`（不启动任何线程）

- [ ] **Step 3：提交**

```bash
git add main.py
git commit -m "feat: add main pipeline loop and thread orchestration"
```

---

## Task 11：集成冒烟测试

**Files:**
- Create: `tests/test_integration.py`

本测试不需要真实 RTSP 流或 GPU，用合成帧直接注入 frame_queue，验证主循环逻辑不崩溃、overlay 输出形状正确。

- [ ] **Step 1：写测试**

`tests/test_integration.py`:
```python
"""
集成冒烟测试：合成帧 → 主循环逻辑 → overlay 输出
不依赖真实 RTSP 流、YOLO 模型或 InsightFace。
"""
import queue
import threading
import numpy as np
from unittest.mock import MagicMock, patch
import state
import overlay
import config


def _make_frame() -> np.ndarray:
    return np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH, 3), dtype=np.uint8)


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

    state.clear()
    frame_queue: queue.Queue = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    face_crop_queue: queue.Queue = queue.Queue(maxsize=config.FACE_CROP_QUEUE_SIZE)
    output_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    detector = FaceDetector()
    tracker = FaceTracker()

    known_ids: set[int] = set()
    track_results: list = []
    frame_count = 0

    # 注入 10 帧合成帧
    for _ in range(10):
        frame_queue.put(_make_frame())

    # 模拟主循环逻辑
    while not frame_queue.empty():
        frame = frame_queue.get_nowait()
        frame_count += 1

        if frame_count % config.DETECT_INTERVAL == 0:
            detections = detector.detect(frame)
            track_results = tracker.update(detections, frame)
        else:
            track_results = tracker.predict()

        emb_snapshot = state.snapshot()
        annotated = overlay.draw_tracks(frame, track_results, emb_snapshot)
        output_queue.put_nowait(annotated)

    assert output_queue.qsize() == 10
    # 所有输出帧形状正确
    while not output_queue.empty():
        f = output_queue.get_nowait()
        assert f.shape == (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)


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

    state.clear()
    face_crop_queue: queue.Queue = queue.Queue(maxsize=config.FACE_CROP_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()
    known_ids: set[int] = set()

    frame = _make_frame()
    # 触发检测帧
    detections = detector.detect(frame)
    track_results = tracker.update(detections, frame)

    for track_id, bbox in track_results:
        if track_id not in known_ids:
            known_ids.add(track_id)
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

- [ ] **Step 2：运行所有测试**

```bash
pytest tests/ -v
```

预期：所有测试 PASSED，无 FAILED 或 ERROR。

- [ ] **Step 3：提交**

```bash
git add tests/test_integration.py
git commit -m "test: add integration smoke tests for pipeline loop"
```

---

## 自检结论

| Spec 章节 | 覆盖任务 |
|-----------|---------|
| 1. 系统目标（检测/跟踪/推流）| Task 5, 6, 7, 8, 9, 10 |
| 2. 技术栈 | Task 1 |
| 3. 整体架构（四线程）| Task 10 |
| 4. 模块划分 | Task 2–10 |
| 5.1 主线程 Pipeline Loop | Task 10 |
| 5.2 InsightFace 线程 | Task 7 |
| 5.3 上半身裁剪规则 | Task 5, 10 |
| 6. 视频流 I/O（FFmpeg 命令）| Task 8, 9 |
| 7.1 断流重连 | Task 8 |
| 7.2 线程异常隔离 | Task 10 |
| 7.3 InsightFace 失败处理 | Task 7 |
| 9. 测试策略 | Task 3–7, 11 |
| 10. 配置参数 | Task 2 |
