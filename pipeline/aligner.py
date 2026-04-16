# pipeline/aligner.py
"""人脸对齐器 — 基于关键点的仿射变换 + embedding 提取。"""

import base64
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from loguru import logger

from config import settings

# ArcFace 标准 5 点参考坐标 (112x112)
ARCFACE_REF_POINTS = np.array(
    [
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ],
    dtype=np.float32,
)

ALIGNMENT_SIZE = (112, 112)


@dataclass(frozen=True)
class AlignmentResult:
    """对齐结果。"""

    aligned_face: np.ndarray  # 仿射对齐后的人脸图像 (112x112x3)
    embedding: np.ndarray  # 512-dim embedding 向量
    landmarks: np.ndarray  # 5 个人脸关键点 (5x2)


def _align_face(crop: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """根据关键点做仿射对齐，返回 112x112 的标准化人脸图像。"""
    M, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_REF_POINTS)
    aligned = cv2.warpAffine(crop, M, ALIGNMENT_SIZE)
    return aligned


def _parse_api_points(points: list[int]) -> np.ndarray | None:
    """解析 API 返回的关键点为 (5, 2) 数组。

    Layout: [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4]
    顺序: 左眼, 右眼, 鼻尖, 左嘴角, 右嘴角
    """
    if len(points) < 10:
        return None
    xs = np.array(points[:5], dtype=np.float32)
    ys = np.array(points[5:], dtype=np.float32)
    return np.stack([xs, ys], axis=1)


def _decode_feature(feature_str: str) -> np.ndarray | None:
    """解码 API feature 字符串为 embedding 向量。"""
    try:
        raw = base64.b64decode(feature_str)
        embedding = np.frombuffer(raw, dtype=np.float32)
        if embedding.size == 0:
            return None
        return embedding.copy()  # frombuffer 返回只读数组，需要可写副本
    except Exception:
        logger.warning("feature 解码失败")
        return None


class FaceAligner:
    """人脸对齐器，根据配置使用 InsightFace 或 FaceApiClient。

    Backend:
    - "insightface": InsightFace FaceAnalysis 获取关键点和 embedding
    - "api": FaceApiClient 获取关键点和 embedding
    """

    def __init__(self, backend: Literal["insightface", "api"] | None = None) -> None:
        if backend is None:
            backend = settings.ALIGNMENT_BACKEND  # type: ignore[assignment]
        if backend not in ("insightface", "api"):
            raise ValueError(f"不支持的 backend: {backend!r}，可选值为 'insightface' 或 'api'")

        self._backend = backend

        if backend == "insightface":
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL,
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=settings.INSIGHTFACE_DET_SIZE)
        else:
            from pipeline.face_api import FaceApiClient

            self._client = FaceApiClient()

    def align(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """对齐人脸裁剪图并提取 embedding。检测失败返回 None。"""
        if self._backend == "insightface":
            return self._align_insightface(face_crop)
        return self._align_api(face_crop)

    def close(self) -> None:
        """释放资源。"""
        if self._backend == "api":
            self._client.close()

    def _align_insightface(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """InsightFace 后端：获取关键点和 embedding，仿射对齐用于 debug。"""
        try:
            results = self._app.get(face_crop)
        except Exception:
            logger.exception("InsightFace 检测失败")
            return None

        if not results:
            return None

        face = results[0]
        landmarks = np.array(face.kps, dtype=np.float32)
        if landmarks.shape != (5, 2):
            return None

        aligned = _align_face(face_crop, landmarks)
        return AlignmentResult(
            aligned_face=aligned,
            embedding=face.embedding,
            landmarks=landmarks,
        )

    def _align_api(self, face_crop: np.ndarray) -> AlignmentResult | None:
        """API 后端：通过 FaceApiClient 获取关键点和 embedding。"""
        try:
            _, buf = cv2.imencode(".jpg", face_crop)
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            resp = self._client.get_face_feature(img_b64)
        except Exception:
            logger.exception("FaceApiClient 调用失败")
            return None

        if resp.status_code != 200 or not resp.data or not resp.data[0]:
            return None

        face_data = resp.data[0][0]
        landmarks = _parse_api_points(face_data.face_pos.points)
        if landmarks is None:
            return None

        embedding = _decode_feature(face_data.feature)
        if embedding is None:
            return None

        aligned = _align_face(face_crop, landmarks)
        return AlignmentResult(
            aligned_face=aligned,
            embedding=embedding,
            landmarks=landmarks,
        )
