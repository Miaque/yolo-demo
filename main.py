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
