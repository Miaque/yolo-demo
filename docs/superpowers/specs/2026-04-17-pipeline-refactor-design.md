# Pipeline Refactor Design Spec

## Background

`main.py:run()` is a 166-line flat function that handles pipeline assembly, signal registration, detection/tracking dispatch, crop distribution, frame rendering, and shutdown. `state.py` is a module-level singleton that multiple modules import directly, making tests non-isolatable. `embedder.py` contains unconditional debug disk writes that will become an I/O bottleneck in production.

## Goals

Make the codebase maintainable and testable:

1. Each module has a single, clear responsibility
2. Shared state is injectable (not global), enabling isolated unit tests
3. No production I/O side effects from debug code
4. Variable names communicate intent accurately

Non-goals (deferred to future iterations):

- Coordinator class extraction
- Metrics / observability
- Structured logging
- Graceful shutdown improvements beyond current behavior

## Design

### 1. TrackState class (replaces module-level singleton)

Replace `state.py` functions and global variables with an instantiable class. The `clear()` and `remove_name()` standalone functions are removed — `clear()` is unnecessary (tests create fresh instances), and `remove_name()` has no external callers (removal is consolidated into `remove_embedding()`).

```python
# state.py
class TrackState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}

    def get_embedding(self, track_id: int) -> np.ndarray | None: ...
    def set_embedding(self, track_id: int, embedding: np.ndarray) -> None: ...
    def remove_embedding(self, track_id: int) -> None: ...  # also clears name
    def get_name(self, track_id: int) -> str | None: ...
    def set_name(self, track_id: int, name: str) -> None: ...
    def snapshot(self) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        # Returns shallow copies of both dicts.
        # numpy arrays inside embedding dict are shared references (same as before).
```

Injection path:

- `EmbedderThread.__init__(face_crop_queue, stop_event, track_state, known_faces=None)` — `track_state` is a required positional argument, placed before `known_faces`. No longer imports `state`.
- `Pipeline.__init__` creates a `TrackState()` instance and passes it to components that need it.

### 2. Pipeline class (replaces flat run() function)

```python
# main.py
class Pipeline:
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
        # Sets instance attributes:
        # self._frame_queue, self._face_crop_queue, self._output_queue (queues)
        # self._detector (FaceDetector), self._tracker (FaceTracker)
        # self._reader (ReaderThread), self._embedder (EmbedderThread), self._writer (WriterThread)
        # Also loads known faces and computes output path.

    def _register_signals(self) -> None: ...    # SIGINT/SIGTERM → stop_event.set()
    def _start_threads(self) -> None: ...       # start reader, embedder, writer

    def _loop(self) -> None:
        # ~20 lines: while not stop_event → get frame from queue → _step(frame)
        track_results: list[tuple[int, list[int]]] = []  # local, not instance attr

    def _step(self, frame: np.ndarray) -> None:
        # ~30 lines, mutates self._submitted_ids and self._frame_count:
        # 1. Increment frame_count, log queue sizes
        # 2. detect or predict → track_results
        # 3. For removed IDs: discard from submitted_ids, remove from track_state
        # 4. For new IDs: add to submitted_ids, crop face, put to face_crop_queue
        # 5. snapshot track_state, draw overlay, put to output_queue

    def _shutdown(self) -> None: ...            # stop_event.set() + join + log
```

Entry point:

```python
if __name__ == "__main__":
    Pipeline().run()
```

The old module-level `run()` function is removed entirely.

- `_load_known_faces()` and `_crop_with_margin()` remain module-level functions (no instance state dependency)
- `match_name()` remains in `embedder.py` unchanged

### 3. Remove debug disk writes

Delete `embedder.py:73-81` (unconditional save of aligned face images), the `self._align_count` field, and the now-unused imports (`from pathlib import Path`, `cv2`). `cv2` is only used for `cv2.imwrite` in the debug block and has no other usage in the file.

No replacement debug mechanism in this iteration.

### 4. Rename known_ids → submitted_ids

`main.py:93` `known_ids: set[int]` becomes `self._submitted_ids: set[int]`. Reflects actual semantics: track IDs that have already been submitted to the embedding queue.

## Files Changed

| File | Change |
|------|--------|
| `state.py` | Functions + globals → `TrackState` class. Remove `clear()`, `remove_name()`. |
| `main.py` | Remove `run()` function → `Pipeline` class. `known_ids` → `self._submitted_ids`. |
| `pipeline/embedder.py` | Delete debug writes + unused imports. `import state` → `track_state: TrackState` constructor injection (required positional arg, before `known_faces`). |
| `tests/test_state.py` | Replace `state.clear()` + `state.xxx()` with `ts = TrackState()` + `ts.xxx()`. Remove all `clear()` calls (fresh instance per test). |
| `tests/test_embedder.py` | Pass `track_state` to `EmbedderThread` constructor. Assert against instance instead of global `state`. Remove all `state.clear()` calls. |
| `tests/test_integration.py` | Replace `import state` + `state.clear()` with local `TrackState()` instance. Rename local `known_ids` → `submitted_ids`. Use `track_state.snapshot()` instead of `state.snapshot()`. |

## Files Not Changed

`config.py`, `overlay.py`, `pipeline/reader.py`, `pipeline/writer.py`, `pipeline/detector.py`, `pipeline/tracker.py`, `pipeline/aligner.py`, and all other test files.

## Test Strategy

- `test_state.py`: Each test creates a fresh `TrackState()` instance. No `clear()` needed. All assertions unchanged except `state.xxx()` → `ts.xxx()`.
- `test_embedder.py`: Pass `track_state` to `EmbedderThread` constructor. Assert against the instance instead of global `state`. Remove `state.clear()` calls.
- `test_integration.py`: Create a `TrackState()` instance, pass to functions under test. Rename `known_ids` → `submitted_ids`.
- All existing test assertions remain the same; only setup changes.
