# pipeline/detector.py
import numpy as np
from ultralytics.models.yolo import YOLO

from config import settings


class FaceDetector:
    """YOLO 两阶段人脸检测：先检测 person，再在上半身区域内检测 face。"""

    def __init__(
        self,
        person_model_path: str = settings.PERSON_MODEL,
        face_model_path: str = settings.FACE_MODEL,
    ) -> None:
        self.person_model = YOLO(person_model_path)
        self.face_model = YOLO(face_model_path)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """返回 [(x1, y1, x2, y2, conf), ...] — 全局坐标系下的人脸框。"""
        person_results = self.person_model(frame, classes=[0], verbose=False)[0]
        faces: list[tuple[int, int, int, int, float]] = []

        for box in person_results.boxes:
            px1, py1, px2, py2 = [int(v) for v in box.xyxy[0]]
            upper_y2 = py1 + (py2 - py1) * 2 // 3

            if upper_y2 - py1 < settings.FACE_MIN_HEIGHT:
                continue

            crop = frame[py1:upper_y2, px1:px2]
            if crop.size == 0:
                continue

            face_results = self.face_model(crop, verbose=False)[0]
            for fbox in face_results.boxes:
                fx1, fy1, fx2, fy2 = [int(v) for v in fbox.xyxy[0]]
                conf = float(fbox.conf[0])
                faces.append((px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2, conf))

        return faces
