# pipeline/embedder.py
import queue
import threading
import numpy as np
from insightface.app import FaceAnalysis
from loguru import logger
import config
import state


class EmbedderThread(threading.Thread):
    """后台线程：从 face_crop_queue 取人脸裁剪图，调用 InsightFace 提取 Embedding。"""

    def __init__(self, face_crop_queue: queue.Queue, stop_event: threading.Event) -> None:
        super().__init__(daemon=True, name="embedder")
        self.face_crop_queue = face_crop_queue
        self.stop_event = stop_event
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
                state.set_embedding(track_id, results[0].embedding)
        except Exception:
            logger.exception("InsightFace failed for track_id={}", track_id)
