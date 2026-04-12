# Face Recognition Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add face recognition so known faces from `faces/` directory are identified by name on the output video bounding boxes.

**Architecture:** Load known face embeddings at startup using InsightFace. At runtime, `EmbedderThread` matches new embeddings against known ones via cosine similarity. Match results are stored in `state.name_store` and rendered by `overlay.draw_tracks()`.

**Tech Stack:** InsightFace (already in use), NumPy cosine similarity, OpenCV text rendering.

---

### Task 1: Add config constants

**Files:**
- Modify: `config.py:31-32` (append after `INSIGHTFACE_DET_SIZE`)

**Step 1: Add constants**

Append to `config.py`:

```python
FACES_DIR: str = "faces"
RECOGNITION_THRESHOLD: float = 0.4
```

**Step 2: Verify**

Run: `python -c "import config; print(config.FACES_DIR, config.RECOGNITION_THRESHOLD)"`
Expected: `faces 0.4`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add FACES_DIR and RECOGNITION_THRESHOLD config"
```

---

### Task 2: Add name_store to state.py

**Files:**
- Modify: `state.py`
- Test: `tests/test_state.py`

**Step 1: Write failing tests**

Append to `tests/test_state.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_state.py -v`
Expected: FAIL — `set_name` / `get_name` not defined, `snapshot` returns wrong type

**Step 3: Implement**

Replace `state.py` with:

```python
# state.py
import threading
from typing import Optional
import numpy as np

_lock = threading.Lock()
embedding_store: dict[int, np.ndarray] = {}
name_store: dict[int, str] = {}


def get_embedding(track_id: int) -> Optional[np.ndarray]:
    with _lock:
        return embedding_store.get(track_id)


def set_embedding(track_id: int, embedding: np.ndarray) -> None:
    with _lock:
        embedding_store[track_id] = embedding


def remove_embedding(track_id: int) -> None:
    with _lock:
        embedding_store.pop(track_id, None)
        name_store.pop(track_id, None)


def get_name(track_id: int) -> Optional[str]:
    with _lock:
        return name_store.get(track_id)


def set_name(track_id: int, name: str) -> None:
    with _lock:
        name_store[track_id] = name


def remove_name(track_id: int) -> None:
    with _lock:
        name_store.pop(track_id, None)


def snapshot() -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """返回两个 store 的浅拷贝，用于主线程传递给 overlay。"""
    with _lock:
        return dict(embedding_store), dict(name_store)


def clear() -> None:
    with _lock:
        embedding_store.clear()
        name_store.clear()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_state.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS (existing tests should still work — `snapshot` now returns tuple but we'll fix callers in later tasks)

**Step 6: Commit**

```bash
git add state.py tests/test_state.py
git commit -m "feat: add name_store to state for face recognition"
```

---

### Task 3: Add matching logic to embedder

**Files:**
- Modify: `pipeline/embedder.py`
- Test: `tests/test_embedder.py`

**Step 1: Write failing test**

Append to `tests/test_embedder.py`:

```python
import numpy as np
import state


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
    from unittest.mock import MagicMock, patch
    from pipeline.embedder import EmbedderThread
    import queue
    import threading
    import time

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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embedder.py -v`
Expected: FAIL — `match_name` not found, `EmbedderThread` missing `known_faces` param

**Step 3: Implement**

Replace `pipeline/embedder.py` with:

```python
# pipeline/embedder.py
import queue
import threading
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
import config
import state


def match_name(
    embedding: np.ndarray,
    known_faces: dict[str, np.ndarray],
    threshold: float = config.RECOGNITION_THRESHOLD,
) -> str:
    """将 embedding 与已知人脸比较，返回最匹配的名字或 'Unknown'。"""
    if not known_faces:
        return "Unknown"

    best_name = "Unknown"
    best_sim = threshold
    for name, known_emb in known_faces.items():
        sim = float(np.dot(embedding, known_emb) / (np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-10))
        if sim > best_sim:
            best_sim = sim
            best_name = name
    return best_name


class EmbedderThread(threading.Thread):
    """后台线程：从 face_crop_queue 取人脸裁剪图，调用 InsightFace 提取 Embedding。"""

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
                embedding = results[0].embedding
                state.set_embedding(track_id, embedding)
                name = match_name(embedding, self.known_faces)
                state.set_name(track_id, name)
        except Exception:
            logger.exception("InsightFace failed for track_id={}", track_id)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embedder.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add pipeline/embedder.py tests/test_embedder.py
git commit -m "feat: add face matching logic to embedder"
```

---

### Task 4: Add _load_known_faces to main.py

**Files:**
- Modify: `main.py`

**Step 1: Add _load_known_faces function**

Add this function after `_crop_with_margin` in `main.py` (around line 29):

```python
def _load_known_faces() -> dict[str, np.ndarray]:
    """启动时从 faces/ 目录加载已知人脸 embedding。"""
    from pathlib import Path
    from insightface.app import FaceAnalysis

    faces_dir = Path(config.FACES_DIR)
    if not faces_dir.is_dir():
        logger.warning("Faces directory '{}' not found, all faces will be Unknown", config.FACES_DIR)
        return {}

    app = FaceAnalysis(name=config.INSIGHTFACE_MODEL, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=config.INSIGHTFACE_DET_SIZE)

    known_faces: dict[str, np.ndarray] = {}
    for img_path in sorted(faces_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: {}", img_path)
            continue
        results = app.get(img)
        if not results:
            logger.warning("No face detected in: {}", img_path)
            continue
        name = img_path.stem
        known_faces[name] = results[0].embedding
        logger.info("Loaded known face: {}", name)

    logger.info("Total known faces loaded: {}", len(known_faces))
    return known_faces
```

Also add `import cv2` to the imports at the top of `main.py` if not already present.

**Step 2: Wire into run()**

In `run()`, after creating `detector` and `tracker` (around line 39-40), add:

```python
    known_faces = _load_known_faces()
```

And change the embedder creation (around line 45) from:

```python
    embedder = EmbedderThread(face_crop_queue, stop_event)
```

to:

```python
    embedder = EmbedderThread(face_crop_queue, stop_event, known_faces=known_faces)
```

**Step 3: Update snapshot call**

In the main loop (around line 110-111), change:

```python
            emb_snapshot = state.snapshot()
            annotated = overlay.draw_tracks(frame, track_results, emb_snapshot)
```

to:

```python
            emb_snapshot, name_snapshot = state.snapshot()
            annotated = overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add _load_known_faces and wire into pipeline"
```

---

### Task 5: Update overlay to display names

**Files:**
- Modify: `overlay.py`
- Test: `tests/test_overlay.py`

**Step 1: Write failing tests**

Append to `tests/test_overlay.py`:

```python
def test_matched_name_shown_in_blue():
    """匹配到名字时用蓝色框。"""
    frame = _blank()
    name_snap = {1: "Alice"}
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {}, name_snap)
    top_row = result[10, 10:101]
    # 蓝色 BGR: (255, 0, 0)
    assert any(np.array_equal(px, [255, 0, 0]) for px in top_row)


def test_unknown_name_shown_in_orange():
    """未匹配到名字时用橙色框。"""
    frame = _blank()
    name_snap = {1: "Unknown"}
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {}, name_snap)
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_overlay.py -v`
Expected: FAIL — `draw_tracks` missing `name_snapshot` parameter

**Step 3: Implement**

Replace `overlay.py` with:

```python
# overlay.py
import cv2
import numpy as np

_BLUE = (255, 0, 0)      # 已识别（有名字）
_ORANGE = (0, 165, 255)   # 未识别（Unknown 或无名字）


def draw_tracks(
    frame: np.ndarray,
    tracks: list[tuple[int, list[int]]],
    emb_snapshot: dict[int, np.ndarray],
    name_snapshot: dict[int, str] | None = None,
) -> np.ndarray:
    """在帧副本上绘制跟踪框和标签，返回新数组。

    Args:
        frame:          BGR 帧（不修改原始帧）
        tracks:         [(track_id, [x1, y1, x2, y2]), ...]
        emb_snapshot:   embedding_store 的快照
        name_snapshot:  name_store 的快照（可选，向后兼容）
    """
    if name_snapshot is None:
        name_snapshot = {}

    out = frame.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = name_snapshot.get(track_id, "Unknown")

        if name != "Unknown":
            color = _BLUE
            label = f"{name} (ID:{track_id})"
        else:
            color = _ORANGE
            label = f"Unknown (ID:{track_id})"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_overlay.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add overlay.py tests/test_overlay.py
git commit -m "feat: overlay displays face names with color-coded bounding boxes"
```

---

### Task 6: Final integration test

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All PASS

**Step 2: Verify existing integration test still passes**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (may need minor adjustments if `snapshot()` return type changed)

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: face recognition complete — known faces identified by name"
```
