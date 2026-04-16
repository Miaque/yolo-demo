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

@dataclass
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
```

### Backend Comparison

| Step | InsightFace Backend | API Backend |
|------|-------------------|-------------|
| Landmarks | `app.get(crop)` → `face.kps` | `client.get_face_feature(b64)` → `face_pos.points` |
| Affine transform | 5-point → ArcFace reference → `cv2.warpAffine` | Same algorithm |
| Embedding | `face.embedding` (already aligned internally) | Decode `feature` field |
| Debug save | Aligned image → `output/debug_aligned/` | Same |

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

## Configuration

New settings in `FaceSettings` (`config.py`):

```python
class FaceSettings(BaseSettings):
    # ... existing settings ...
    ALIGNMENT_BACKEND: str = "insightface"   # "insightface" | "api"
    ALIGNMENT_SIZE: tuple[int, int] = (112, 112)  # aligned face output size
```

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

    def _process(self, track_id: int, face_crop: np.ndarray) -> None:
        result = self._aligner.align(face_crop)
        if result is None:
            return

        # Save aligned face for debug (replaces raw crop debug output)
        debug_path = Path(f"output/debug_aligned/align_{track_id}_{count}.jpg")
        cv2.imwrite(str(debug_path), result.aligned_face)

        # Store embedding and match name
        state.set_embedding(track_id, result.embedding)
        name = match_name(result.embedding, self.known_faces)
        state.set_name(track_id, name)
```

**Key changes**:
- `FaceAligner` replaces direct `FaceAnalysis` usage
- Debug output moves from `output/debug_crops/` (raw crops) to `output/debug_aligned/` (aligned faces)
- When using API backend, InsightFace model is not loaded at all

## New Files

| File | Purpose |
|------|---------|
| `pipeline/aligner.py` | `FaceAligner` class + `AlignmentResult` dataclass + ArcFace reference points |

## Modified Files

| File | Change |
|------|--------|
| `config.py` | Add `ALIGNMENT_BACKEND` and `ALIGNMENT_SIZE` to `FaceSettings` |
| `pipeline/embedder.py` | Replace `FaceAnalysis` with `FaceAligner`, update `_process()` |
| `tests/test_aligner.py` | New test file for FaceAligner |
| `tests/test_embedder.py` | Update mocks to match new FaceAligner-based flow |

## Testing Strategy

### Unit Tests (`tests/test_aligner.py`)

| Test Case | Verification |
|-----------|-------------|
| `test_align_returns_result_on_valid_crop` | Mock InsightFace returns valid Face → AlignmentResult produced |
| `test_align_returns_none_on_no_face` | Mock InsightFace returns empty → None |
| `test_aligned_face_size` | Aligned image is configured size (112x112) |
| `test_align_api_backend` | Mock FaceApiClient → landmarks parsed, feature decoded |
| `test_affine_transform_identity` | Standard frontal landmarks → transform near identity |
| `test_invalid_backend_raises` | Invalid backend string → ValueError |

### Mock Strategy

- **InsightFace**: Mock `FaceAnalysis.get()` returning Face objects with `kps` and `embedding`
- **FaceApiClient**: Mock `get_face_feature()` returning response with `points` and `feature`
- No real models or network calls required

### Existing Test Impact

- `test_embedder.py`: Update mocks (FaceAligner replaces direct FaceAnalysis calls)
- `test_overlay.py`, `test_tracker.py`, `test_state.py`: No changes needed
