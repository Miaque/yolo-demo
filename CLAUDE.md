# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time RTSP face detection, tracking, and recognition system for CPU-only deployment (Raspberry Pi / NUC). Single-process, multi-threaded Python pipeline that reads an RTSP stream, detects and tracks faces using YOLO + ByteTrack, recognizes known faces via InsightFace embeddings, and writes annotated output to MP4.

## Commands

```bash
uv sync                    # Install dependencies
uv run python main.py      # Run the pipeline
uv run pytest              # Run all tests
uv run pytest tests/test_detector.py::test_single_person_single_face  # Run single test
uv run ruff check .        # Lint
uv run ruff format .       # Format
uv add <package>           # Add a dependency
```

Python 3.12 required (enforced by `.python-version`). Package manager is `uv`.

## Architecture

Four threads communicating via bounded `queue.Queue` instances, coordinated by a shared `threading.Event` stop signal:

```
ReaderThread ──frame_queue──> Main Thread ──face_crop_queue──> EmbedderThread
                                │                                    │
                                └──output_queue──> WriterThread       │
                                     (FFmpeg MP4)                    │
                                      ▲                              │
                                      └── state.py (thread-safe) ────┘
```

- **Main thread** (`main.py:run()`): On every Nth frame (config.DETECT_INTERVAL=5), runs two-stage YOLO detection (person → face) then ByteTrack update. On other frames, runs tracker prediction only. New track IDs are cropped and queued for embedding. Annotated frames go to output queue.
- **ReaderThread** (`pipeline/reader.py`): Reads RTSP via OpenCV, resizes to 1280x720, pushes to frame_queue (drops old frames when full).
- **EmbedderThread** (`pipeline/embedder.py`): Extracts 512-dim InsightFace embeddings, matches against known faces by cosine similarity, stores results in `state.py`.
- **WriterThread** (`pipeline/writer.py`): Writes annotated frames to MP4 via FFmpeg pipe at 25fps.

### Key files

| File | Role |
|------|------|
| `config.py` | All tunable constants (RTSP URL, model paths, thresholds, queue sizes) |
| `state.py` | Thread-safe shared state with `threading.Lock` — stores track embeddings and matched names |
| `overlay.py` | Pure function `draw_tracks()` — blue boxes for known faces, orange for unknown |
| `pipeline/detector.py` | `FaceDetector` — two-stage YOLO: detect persons (yolo11n), crop upper body, detect faces (yolov8n-face) |
| `pipeline/tracker.py` | `FaceTracker` — ByteTrack wrapper with `update()`/`predict()`/`removed_ids()` |
| `pipeline/embedder.py` | `EmbedderThread` + `match_name()` — InsightFace embedding and cosine similarity matching |
| `pipeline/reader.py` | `ReaderThread` — RTSP reader with auto-reconnect |
| `pipeline/writer.py` | `WriterThread` — FFmpeg pipe writer |
| `main.py` | Entry point — orchestrates all threads, runs main loop, loads known faces at startup |

## Model weights

Model files are gitignored. On first run, `yolo11n.pt` auto-downloads via ultralytics. `yolov8n-face.pt` must be manually downloaded from https://github.com/akanametov/yolo-face/releases and placed in the project root.

## Known faces

Place face images in `faces/` directory. The filename stem (without extension) becomes the person's name. Loaded at startup via `_load_known_faces()` in `main.py`.

## Testing

Tests mock all external dependencies (YOLO, ByteTrack, InsightFace, OpenCV). No real models or RTSP streams needed. Tests use `unittest.mock.patch` and `MagicMock`. Test config: `testpaths=["tests"]`, `pythonpath=["."]` (set in pyproject.toml).

## Language

Comments, log messages, and documentation are in Chinese.

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
