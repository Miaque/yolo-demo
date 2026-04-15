# pipeline/embedder.py
import queue
import threading
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from loguru import logger
from config import settings
import state


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
            name=settings.INSIGHTFACE_MODEL,
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=settings.INSIGHTFACE_DET_SIZE)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                track_id, face_crop = self.face_crop_queue.get(timeout=0.5)
                self._process(track_id, face_crop)
            except queue.Empty:
                continue

    _crop_count = 0

    def _process(self, track_id: int, face_crop: np.ndarray) -> None:
        try:
            # DEBUG: 保存所有裁剪图用于排查
            from pathlib import Path
            debug_dir = Path("output/debug_crops")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"crop_{track_id}_{self._crop_count}.jpg"), face_crop)
            logger.info("DEBUG saved crop: output/debug_crops/crop_{}_{}.jpg shape={}", track_id, self._crop_count, face_crop.shape)
            self._crop_count += 1

            results = self._app.get(face_crop)
            if results:
                embedding = results[0].embedding
                state.set_embedding(track_id, embedding)
                name = match_name(embedding, self.known_faces)
                logger.info("track_id={} matched='{}' emb_norm={:.4f}", track_id, name, float(np.linalg.norm(embedding)))
                state.set_name(track_id, name)
        except Exception:
            logger.exception("InsightFace failed for track_id={}", track_id)
