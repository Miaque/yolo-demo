# tests/test_face_api.py
"""FaceApiClient 集成测试 — 调用真实接口。"""

import asyncio
import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.face_api import FaceApiClient
from schemas.face_feature import FaceFeatureData

FACES_DIR = Path(__file__).resolve().parent.parent / "faces"
TEST_IMAGE = FACES_DIR / "zhanjh.jpg"


def _read_image_as_base64(path: Path) -> str:
    """读取图片文件并转为 Base64 字符串。"""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@pytest.fixture()
async def client() -> FaceApiClient:
    async with FaceApiClient() as c:
        yield c


@pytest.fixture()
def img_b64() -> str:
    return _read_image_as_base64(TEST_IMAGE)


async def test_get_face_feature_success(client: FaceApiClient, img_b64: str) -> None:
    """调用真实接口，验证返回结构完整且包含人脸。"""
    resp = await client.get_face_feature(img_b64)

    assert resp.status_code == 200
    assert resp.detail == "success"
    assert len(resp.data) > 0

    faces = resp.data[0]
    assert len(faces) >= 1

    face = faces[0]
    assert face.feature, "特征向量不应为空"
    assert face.age > 0
    assert face.sex in (0, 1)
    assert face.has_human_face > 0.5

    pos = face.face_pos
    assert pos.x2 > pos.x1
    assert pos.y2 > pos.y1
    assert pos.probability > 0.5
    assert len(pos.points) == 10


def _annotate_face(frame: np.ndarray, face: FaceFeatureData) -> np.ndarray:
    """在图片上绘制人脸框、关键点和属性标签。"""
    out = frame.copy()
    pos = face.face_pos

    # 人脸框（绿色）
    cv2.rectangle(out, (pos.x1, pos.y1), (pos.x2, pos.y2), (0, 255, 0), 2)

    # 5 个关键点（红色圆点）
    # points 格式：前 5 个为 X 坐标，后 5 个为 Y 坐标
    _LANDMARK_NAMES = ["L-eye", "R-eye", "Nose", "L-mouth", "R-mouth"]
    xs, ys = pos.points[:5], pos.points[5:]
    for i, name in enumerate(_LANDMARK_NAMES):
        px, py = xs[i], ys[i]
        cv2.circle(out, (px, py), 4, (0, 0, 255), -1)
        cv2.putText(
            out, name, (px + 4, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
        )

    # 属性标签
    sex_text = "M" if face.sex == 1 else "F"
    label = f"sex={sex_text} age={face.age} prob={pos.probability:.2f}"
    cv2.putText(
        out, label, (pos.x1, max(pos.y1 - 8, 14)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return out


async def test_annotate_face_image(client: FaceApiClient, img_b64: str) -> None:
    """调用真实接口，在原图上绘制人脸框和关键点，保存标注图片。"""
    resp = await client.get_face_feature(img_b64)
    assert resp.status_code == 200
    assert len(resp.data) > 0 and len(resp.data[0]) >= 1

    # 解码原图
    img_bytes = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    assert frame is not None

    # 标注所有人脸
    annotated = frame
    for face in resp.data[0]:
        annotated = _annotate_face(annotated, face)

    # 保存
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotated_face_api.jpg"
    cv2.imwrite(str(output_path), annotated)
    assert output_path.exists(), f"标注图片未生成: {output_path}"
