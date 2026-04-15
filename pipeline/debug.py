# pipeline/debug.py
"""调试用绘图工具 — 在图片上绘制检测/识别结果。"""

import cv2
import numpy as np

from schemas.face_feature import FaceFeatureData

# 人脸关键点名称
_LANDMARK_NAMES = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]

# 颜色常量 (BGR)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)


def draw_faces(
    frame: np.ndarray,
    faces: list[tuple[int, int, int, int, float]],
    *,
    color: tuple[int, int, int] = _BLUE,
    thickness: int = 2,
) -> np.ndarray:
    """绘制 FaceDetector.detect() 返回的人脸框。

    Args:
        frame: BGR 图像。
        faces: [(x1, y1, x2, y2, conf), ...] 列表。
        color: 框颜色 (BGR)，默认蓝色。
        thickness: 线宽。

    Returns:
        标注后的图像副本。
    """
    out = frame.copy()
    for x1, y1, x2, y2, conf in faces:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            out,
            f"{conf:.2f}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_face_features(
    frame: np.ndarray,
    faces: list[FaceFeatureData],
) -> np.ndarray:
    """绘制 FaceApiClient 返回的人脸框、关键点和属性标签。

    Args:
        frame: BGR 图像。
        faces: FaceFeatureData 列表。

    Returns:
        标注后的图像副本。
    """
    out = frame.copy()
    for face in faces:
        pos = face.face_pos

        # 人脸框（绿色）
        cv2.rectangle(out, (pos.x1, pos.y1), (pos.x2, pos.y2), _GREEN, 2)

        # 关键点（红色圆点 + 名称）
        xs, ys = pos.points[:5], pos.points[5:]
        for i, name in enumerate(_LANDMARK_NAMES):
            px, py = xs[i], ys[i]
            cv2.circle(out, (px, py), 4, _RED, -1)
            cv2.putText(
                out,
                name,
                (px + 4, py - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                _RED,
                1,
                cv2.LINE_AA,
            )

        # 属性标签
        sex_text = "M" if face.sex == 1 else "F"
        label = f"sex={sex_text} age={face.age} prob={pos.probability:.2f}"
        cv2.putText(
            out,
            label,
            (pos.x1, max(pos.y1 - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _GREEN,
            1,
            cv2.LINE_AA,
        )
    return out
