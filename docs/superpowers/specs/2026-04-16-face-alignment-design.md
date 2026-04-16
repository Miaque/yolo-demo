# Face Landmark Alignment (Affine Transform) Design

**Date**: 2026-04-16
**Status**: Draft

## Problem

The current pipeline crops faces with a bbox + margin (`_crop_with_margin()`) and feeds the raw crop to InsightFace. This produces inconsistent face images for debugging and relies entirely on InsightFace's internal alignment, which is not visible or configurable. We need explicit face alignment using facial landmarks and affine transformation to:

1. Improve recognition accuracy with standardized aligned faces
2. Produce normalized face images for debugging and visualization

## Design Decision

**Approach C: FaceAligner class** — a standalone class in `pipeline/aligner.py` that encapsulates landmark detection and affine transformation. Backend (InsightFace or FaceApiClient) is configuration-driven.

## Architecture

### Data Flow

```
Main Thread → _crop_with_margin() → face_crop_queue → EmbedderThread
                                                         ↓
                                                    FaceAligner.align(crop)
                                                    ├─ 获取关键点 (InsightFace/API)
                                                    ├─ 计算仿射矩阵
                                                    └─ cv2.warpAffine → aligned_face
                                                         ↓
                                                    embedding 提取
                                                         ↓
                                                    state.set_embedding/name
                                                    + 保存对齐图到 debug_aligned/
```

### FaceAligner Interface

```python
# pipeline/aligner.py

from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class AlignmentResult:
    aligned_face: np.ndarray   # Affine-aligned face image (112x112x3)
    embedding: np.ndarray      # 512-dim embedding vector
    landmarks: np.ndarray      # 5 facial landmarks (5x2)

class FaceAligner:
    """Face alignment using facial landmarks and affine transformation.

    Backend is determined by configuration:
    - "insightface": Uses InsightFace FaceAnalysis for landmarks and embedding
    - "api": Uses FaceApiClient for landmarks and embedding
    """

    def __init__(self, backend: Literal["insightface", "api"]) -> None:
        ...

    def align(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """Align a face crop and extract embedding. Returns None if no face detected."""
        ...

    def close(self) -> None:
        """Release resources (close FaceApiClient session, etc.)."""
        ...
```

### Backend Comparison

| Step | InsightFace Backend | API Backend |
|------|-------------------|-------------|
| Landmarks | `app.get(crop)` → `face.kps` | `client.get_face_feature(b64)` → `face_pos.points` |
| Affine transform | 5-point → ArcFace reference → `cv2.warpAffine` | Same algorithm |
| Embedding | `face.embedding` (InsightFace 内部已对齐) | Decode `feature` 字段 (见下方说明) |
| Debug save | Aligned image → `output/debug_aligned/` | Same |

**InsightFace 后端说明**: InsightFace `get()` 内部已经做了对齐和 embedding 提取。我们的仿射变换产出的对齐图像**仅用于 debug 保存和标准化可视化**。Embedding 直接使用 `face.embedding`，不做重复计算。对齐图与 InsightFace 内部使用的对齐标准一致（相同的 ArcFace 参考点）。

### Alignment Algorithm

Standard ArcFace 5-point affine alignment:

1. Detect 5 facial landmarks: left eye, right eye, nose tip, left mouth corner, right mouth corner
2. Map to standard ArcFace reference coordinates for 112x112 output
3. Compute affine matrix via `cv2.estimateAffinePartial2D(src_points, ref_points)`
4. Apply `cv2.warpAffine(crop, M, (112, 112))`

ArcFace reference points (112x112):

```python
ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)
```

### API Backend: Feature Decoding

The FaceApiClient returns `feature` as a string field (`schemas/face_feature.py`). Decoding procedure:

```python
import base64

def _decode_feature(feature_str: str) -> np.ndarray:
    """Decode API feature string to embedding vector.

    The API returns a base64-encoded float32 array.
    Implementation must verify the actual encoding by testing against the live API.
    """
    raw = base64.b64decode(feature_str)
    embedding = np.frombuffer(raw, dtype=np.float32)
    return embedding
```

**重要**: `feature` 字段的编码方式（Base64 / HEX）和向量维度需在实现时对照实际 API 响应验证。当前假设为 Base64 编码的 float32 数组，维度与 InsightFace 兼容。如不兼容，`match_name()` 的余弦相似度比较将无意义。

### API Backend: Landmarks Parsing

`FacePosition.points` 是 `list[int]`，包含 10 个元素。

**注意**: schema 注释描述为交错格式 `[lx, ly, rx, ry, ...]`，但 `pipeline/debug.py` 实际使用的是**分离格式** `xs, ys = points[:5], points[5:]`。实现时应使用分离格式（与 debug.py 一致），并在首次接入 API 时通过实际响应验证：

```python
def _parse_api_points(points: list[int]) -> np.ndarray:
    """Parse API landmarks to (5, 2) array.

    Layout: [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4]
    Points order: left eye, right eye, nose, left mouth, right mouth
    """
    xs = np.array(points[:5], dtype=np.float32)
    ys = np.array(points[5:], dtype=np.float32)
    return np.stack([xs, ys], axis=1)  # shape: (5, 2)
```

## Error Handling

`FaceAligner.align()` handles the following error cases internally, returning `None`:

- No face detected in crop (InsightFace returns empty list, or API returns no face data)
- API network timeout or HTTP error — caught internally, logged via `loguru`, returns `None`
- API returns face but `points` is empty or has fewer than 10 elements — returns `None`
- API `feature` field fails to decode — logged, returns `None`

Exceptions that are **not** caught by `align()` (propagated to caller):
- Invalid constructor arguments (e.g., unsupported backend string → `ValueError`)

## Configuration

New settings in `FaceSettings` (`config.py`):

```python
class FaceSettings(BaseSettings):
    # ... existing settings ...
    ALIGNMENT_BACKEND: str = "insightface"   # "insightface" | "api"
```

**`ALIGNMENT_BACKEND` 与 `DETECTOR_BACKEND` 的关系**:

| DETECTOR_BACKEND | ALIGNMENT_BACKEND | 行为 |
|-----------------|-------------------|------|
| "model" (默认) | "insightface" (默认) | YOLO 检测 + InsightFace 对齐/embedding |
| "model" | "api" | YOLO 检测 + API 对齐/embedding（InsightFace 不加载） |
| "api" | "insightface" | API 检测 + InsightFace 对齐/embedding |
| "api" | "api" | API 全流程，InsightFace 不加载 |

对齐后图像尺寸固定为 112x112（ArcFace 标准），不提供配置。InsightFace 的 rec_model 要求 112x112 输入，其他尺寸会降低识别准确率。

## EmbedderThread Changes

The `EmbedderThread` is refactored to use `FaceAligner` instead of directly calling `FaceAnalysis`:

```python
class EmbedderThread(threading.Thread):
    def __init__(self, face_crop_queue, stop_event, known_faces=None):
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
        self.known_faces = known_faces or {}
        self._aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)
        self._align_count = 0

    def _process(self, track_id: int, face_crop: np.ndarray) -> None:
        try:
            result = self._aligner.align(face_crop)
            if result is None:
                return

            # Save aligned face for debug
            debug_dir = Path("output/debug_aligned")
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"align_{track_id}_{self._align_count}.jpg"
            cv2.imwrite(str(debug_path), result.aligned_face)
            logger.info("DEBUG saved aligned: {} shape={}", debug_path, result.aligned_face.shape)
            self._align_count += 1

            # Store embedding and match name
            state.set_embedding(track_id, result.embedding)
            name = match_name(result.embedding, self.known_faces)
            state.set_name(track_id, name)
        except Exception:
            logger.exception("FaceAligner failed for track_id={}", track_id)
```

**Key changes**:
- `FaceAligner` replaces direct `FaceAnalysis` usage
- Debug output moves from `output/debug_crops/` (raw crops) to `output/debug_aligned/` (aligned faces)
- When using API backend, InsightFace model is not loaded at all

## _load_known_faces Changes

`main.py` 中的 `_load_known_faces()` 也需要使用 `FaceAligner` 提取已知人脸 embedding，确保 embedding 空间一致：

```python
def _load_known_faces(aligner: FaceAligner) -> dict[str, np.ndarray]:
    """启动时从 faces/ 目录加载已知人脸 embedding，使用与管线相同的 FaceAligner。"""
    # 使用 aligner.align() 提取 embedding，确保 embedding 空间与运行时一致
    ...
```

**注意**: 当使用 API 后端时，`_load_known_faces()` 会为每张已知人脸调用 API。需要确保 API 服务在启动时可用。

## New Files

| File | Purpose |
|------|---------|
| `pipeline/aligner.py` | `FaceAligner` class + `AlignmentResult` dataclass + ArcFace reference points + feature decoding |

## Modified Files

| File | Change |
|------|--------|
| `config.py` | Add `ALIGNMENT_BACKEND` to `FaceSettings` |
| `pipeline/embedder.py` | Replace `FaceAnalysis` with `FaceAligner`, update `_process()` |
| `main.py` | Update `_load_known_faces()` to use `FaceAligner` |
| `tests/test_aligner.py` | New test file for FaceAligner |
| `tests/test_embedder.py` | Update mocks to match new FaceAligner-based flow |

## Testing Strategy

### Unit Tests (`tests/test_aligner.py`)

| Test Case | Verification |
|-----------|-------------|
| `test_align_returns_result_on_valid_crop` | Mock InsightFace returns valid Face → AlignmentResult produced |
| `test_align_returns_none_on_no_face` | Mock InsightFace returns empty → None |
| `test_aligned_face_size` | Aligned image is 112x112x3 |
| `test_align_api_backend` | Mock FaceApiClient → landmarks parsed, feature decoded |
| `test_align_api_backend_no_landmarks` | Mock API with empty `points` → returns None |
| `test_affine_transform_identity` | Standard frontal landmarks → transform near identity |
| `test_invalid_backend_raises` | Invalid backend string → ValueError |
| `test_close_releases_resources` | `close()` called without error for both backends |

### Mock Strategy

- **InsightFace**: Mock `FaceAnalysis.get()` returning Face objects with `kps` and `embedding`
- **FaceApiClient**: Mock `get_face_feature()` returning response with `points` and `feature`
- No real models or network calls required

### Existing Test Impact

- `test_embedder.py`: Update mocks (FaceAligner replaces direct FaceAnalysis calls)
- `test_overlay.py`, `test_tracker.py`, `test_state.py`: No changes needed
