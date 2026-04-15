# tests/test_face_api.py
"""FaceApiClient 集成测试 — 调用真实接口。"""

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipeline.debug import draw_face_features
from pipeline.face_api import FaceApiClient
from schemas.face_feature import FaceFeatureData

FACES_DIR = Path(__file__).resolve().parent.parent / "faces"
TEST_IMAGE = FACES_DIR / "35fd8561b499fe326d720853003b2aa9.jpg"


def _read_image_as_base64(path: Path) -> str:
    """读取图片文件并转为 Base64 字符串。"""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@pytest.fixture()
def client() -> FaceApiClient:
    with FaceApiClient() as c:
        yield c


@pytest.fixture()
def img_b64() -> str:
    return _read_image_as_base64(TEST_IMAGE)


def test_get_face_feature_success(client: FaceApiClient, img_b64: str) -> None:
    """调用真实接口，验证返回结构完整且包含人脸。"""
    resp = client.get_face_feature(img_b64)

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


def test_annotate_face_image(client: FaceApiClient, img_b64: str) -> None:
    """调用真实接口，在原图上绘制人脸框和关键点，保存标注图片。"""
    resp = client.get_face_feature(img_b64)
    assert resp.status_code == 200
    assert len(resp.data) > 0 and len(resp.data[0]) >= 1

    # 解码原图
    img_bytes = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    assert frame is not None

    # 标注所有人脸
    annotated = draw_face_features(frame, resp.data[0])

    # 保存
    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "annotated_face_api.jpg"
    cv2.imwrite(str(output_path), annotated)
    assert output_path.exists(), f"标注图片未生成: {output_path}"
