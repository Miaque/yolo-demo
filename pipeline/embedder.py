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
        sim = float(
            np.dot(embedding, known_emb)
            / (np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-10)
        )
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
