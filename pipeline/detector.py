# pipeline/detector.py
import base64
from typing import Literal

import cv2
import numpy as np
from ultralytics.models.yolo import YOLO

from config import settings
from pipeline.face_api import FaceApiClient


class FaceDetector:
    """人脸检测器，支持 YOLO 两阶段模型（model）和人脸特征提取接口（api）两种后端。"""

    def __init__(
        self,
        backend: Literal["model", "api"] | None = None,
        person_model_path: str = settings.PERSON_MODEL,
        face_model_path: str = settings.FACE_MODEL,
    ) -> None:
        if backend is None:
            backend = settings.DETECTOR_BACKEND  # type: ignore[assignment]
        if backend not in ("model", "api"):
            raise ValueError(f"不支持的 backend: {backend!r}，可选值为 'model' 或 'api'")

        self._backend = backend

        if backend == "model":
            self.person_model = YOLO(person_model_path)
            self.face_model = YOLO(face_model_path)
        else:
            self._api_client = FaceApiClient()

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """返回 [(x1, y1, x2, y2, conf), ...] — 全局坐标系下的人脸框。"""
        if self._backend == "model":
            return self._detect_model(frame)
        return self._detect_api(frame)

    def _detect_model(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """使用 YOLO 两阶段模型检测人脸：先检测 person，再在上半身区域检测 face。"""
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

    def _detect_api(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        """使用 Face API 检测人脸，将整帧图像发送给接口并解析返回的人脸坐标。"""
        _, buf = cv2.imencode(".jpg", frame)
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        resp = self._api_client.get_face_feature(img_b64)

        faces: list[tuple[int, int, int, int, float]] = []
        if resp.status_code == 200 and resp.data:
            for face in resp.data[0]:
                pos = face.face_pos
                faces.append((pos.x1, pos.y1, pos.x2, pos.y2, pos.probability))
        return faces

    def close(self) -> None:
        """释放资源（API 模式下关闭 HTTP Session）。"""
        if self._backend == "api":
            self._api_client.close()
