# Face Recognition Design

## Overview

Add face recognition to the existing detection pipeline. Known faces are stored as images in `faces/` directory (filename = person name), loaded at startup. Recognized names are displayed on bounding boxes in the output video.

## Requirements

- Each image in `faces/` contains exactly one face
- Fixed cosine similarity threshold for matching
- Display person name when matched, "Unknown" when not
- Always show track ID alongside name

## Approach: Reuse Existing InsightFace Pipeline

Minimal changes to the current data flow. Load known face embeddings at startup, match against tracked face embeddings at runtime.

## Data Flow

### Startup
1. Create `FaceAnalysis` instance in `main.py`
2. Scan `faces/` directory, extract embedding per image
3. Store as `known_faces: dict[str, np.ndarray]` (name → embedding)

### Runtime
1. `EmbedderThread` extracts embedding for tracked face
2. Compare against `known_faces` using cosine similarity
3. Best match above threshold → store name; otherwise → "Unknown"
4. Store result in `state.name_store[track_id]`
5. `overlay.draw_tracks()` reads name and renders on frame

## Module Changes

### `config.py`
- `FACES_DIR: str = "faces"` — known face images directory
- `RECOGNITION_THRESHOLD: float = 0.4` — cosine similarity threshold

### `state.py`
- Add `name_store: dict[int, str]` with `get_name()`, `set_name()`, `remove_name()`
- `snapshot()` returns `(embedding_store_copy, name_store_copy)`
- `remove_embedding()` also calls `remove_name()`

### `main.py`
- Add `_load_known_faces()` — scan `faces/`, extract embeddings, return dict
- Pass `known_faces` to `EmbedderThread` constructor

### `pipeline/embedder.py`
- Accept `known_faces: dict[str, np.ndarray]` in `__init__`
- After extracting embedding, call `_match()` for cosine similarity comparison
- Store matched name via `state.set_name()`

### `overlay.py`
- `draw_tracks()` accepts additional `name_snapshot` parameter
- Matched: show `Name (ID:xxx)` with blue bounding box
- Unmatched: show `Unknown (ID:xxx)` with orange bounding box

### Unchanged
- `detector.py`, `tracker.py`, `reader.py`, `writer.py`

## Edge Cases

- `faces/` missing → warn and run with empty known_faces (all Unknown)
- No face detected in image → warn and skip that file
- Multiple matches above threshold → pick highest similarity
- Track removed → `remove_name()` called alongside `remove_embedding()`

## Out of Scope

- Hot-reloading faces directory
- Face database CRUD API
- Recognition history logging
