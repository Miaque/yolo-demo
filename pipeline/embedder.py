# pipeline/embedder.py
import queue
import threading

import numpy as np
from loguru import logger

from config import settings
from state import TrackState
from pipeline.aligner import FaceAligner


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
        track_state: TrackState,
        known_faces: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
        self.track_state = track_state
        self.known_faces = known_faces or {}
        self._aligner = FaceAligner(backend=settings.ALIGNMENT_BACKEND)

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

            # 存储 embedding 和匹配名字
            self.track_state.set_embedding(track_id, result.embedding)
            name = match_name(result.embedding, self.known_faces)
            logger.info(
                "track_id={} matched='{}' emb_norm={:.4f}",
                track_id,
                name,
                float(np.linalg.norm(result.embedding)),
            )
            self.track_state.set_name(track_id, name)
        except Exception:
            logger.exception("FaceAligner failed for track_id={}", track_id)
