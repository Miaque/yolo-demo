# main.py
import queue
import signal
import threading
import numpy as np
from loguru import logger
import config
import overlay
import state
from pipeline.detector import FaceDetector
from pipeline.embedder import EmbedderThread
from pipeline.reader import ReaderThread
from pipeline.tracker import FaceTracker
from pipeline.writer import WriterThread


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
        logger.info("Signal {} received, shutting down…", sig)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    reader.start()
    embedder.start()
    writer.start()
    logger.info("Pipeline started. Input: {}", config.RTSP_INPUT)

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
