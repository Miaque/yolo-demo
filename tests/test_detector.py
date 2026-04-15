from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from loguru import logger


def _mock_box(
    x1: float, y1: float, x2: float, y2: float, conf: float = 0.9
) -> MagicMock:
    box = MagicMock()
    box.xyxy = [np.array([x1, y1, x2, y2], dtype=np.float32)]
    box.conf = [np.float32(conf)]
    return box


def _mock_result(boxes: list) -> MagicMock:
    r = MagicMock()
    r.boxes = boxes
    return r


@patch("pipeline.detector.YOLO")
def test_face_global_coords(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # person box: (100,100) → (200,300), upper half → y2=200
    person_model.return_value = [_mock_result([_mock_box(100, 100, 200, 300)])]

    face_model = MagicMock()
    # face box in crop coords: (10,10)→(50,50)
    face_model.return_value = [_mock_result([_mock_box(10, 10, 50, 50)])]

    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert len(faces) == 1
    x1, y1, x2, y2, conf = faces[0]
    assert x1 == 110  # 100 + 10
    assert y1 == 110  # 100 + 10
    assert x2 == 150  # 100 + 50
    assert y2 == 150  # 100 + 50


@patch("pipeline.detector.YOLO")
def test_skips_too_small_crop(mock_yolo_cls):
    """上半身高度 < FACE_MIN_HEIGHT 时跳过，face model 不被调用。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # 人体高度 20px → 上半身 10px < FACE_MIN_HEIGHT(40)
    person_model.return_value = [_mock_result([_mock_box(100, 100, 200, 120)])]

    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert len(faces) == 0
    face_model.assert_not_called()


@patch("pipeline.detector.YOLO")
def test_no_persons_returns_empty(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [_mock_result([])]

    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


@patch("pipeline.detector.YOLO")
def test_multiple_persons_multiple_faces(mock_yolo_cls):
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [
        _mock_result(
            [
                _mock_box(0, 0, 100, 200),
                _mock_box(200, 0, 300, 200),
            ]
        )
    ]

    face_model = MagicMock()
    face_model.side_effect = [
        [_mock_result([_mock_box(5, 5, 40, 40)])],
        [_mock_result([_mock_box(5, 5, 40, 40)])],
    ]
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)
    assert len(faces) == 2


@patch("pipeline.detector.YOLO")
def test_person_detect_uses_class_zero(mock_yolo_cls):
    """person_model 应以 classes=[0] 调用，只检测 person 类。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [_mock_result([])]
    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    detector.detect(np.zeros((360, 640, 3), dtype=np.uint8))

    person_model.assert_called_once()
    assert person_model.call_args[1]["classes"] == [0]


@patch("pipeline.detector.YOLO")
def test_empty_crop_skipped(mock_yolo_cls):
    """裁剪区域超出画面边界（crop.size == 0）时跳过，不调用人脸模型。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # y2=360 与 frame 高度一致，上半身 y2=180，裁剪正常
    # 但让 y1 >= frame 高度使裁剪为空
    person_model.return_value = [_mock_result([_mock_box(0, 400, 100, 500)])]

    face_model = MagicMock()
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    # frame 高度只有 360，y1=400 已超出 → crop 为空
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert faces == []
    face_model.assert_not_called()


@patch("pipeline.detector.YOLO")
def test_single_person_multiple_faces(mock_yolo_cls):
    """同一个人裁剪区内检测到多张脸，坐标全部转换为全局坐标。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    # person box: (50, 50) → (200, 250)
    person_model.return_value = [_mock_result([_mock_box(50, 50, 200, 250)])]

    face_model = MagicMock()
    face_model.return_value = [
        _mock_result(
            [
                _mock_box(10, 10, 30, 30, conf=0.95),
                _mock_box(60, 10, 80, 30, conf=0.80),
            ]
        )
    ]
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    faces = detector.detect(frame)

    assert len(faces) == 2
    # 第一张脸: (50+10, 50+10, 50+30, 50+30, 0.95)
    assert faces[0] == (60, 60, 80, 80, pytest.approx(0.95, abs=0.01))
    # 第二张脸: (50+60, 50+10, 50+80, 50+30, 0.80)
    assert faces[1] == (110, 60, 130, 80, pytest.approx(0.80, abs=0.01))


@patch("pipeline.detector.YOLO")
def test_person_with_no_faces(mock_yolo_cls):
    """person 检测到但裁剪区内无人脸 → 返回空列表。"""
    from pipeline.detector import FaceDetector

    person_model = MagicMock()
    person_model.return_value = [_mock_result([_mock_box(0, 0, 200, 400)])]

    face_model = MagicMock()
    face_model.return_value = [_mock_result([])]
    mock_yolo_cls.side_effect = [person_model, face_model]

    detector = FaceDetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


@pytest.mark.integration
def test_detect_real_image():
    """使用真实 YOLO 模型检测 faces/ 中的人脸，生成带标注的图片。"""
    from pathlib import Path

    import cv2

    from pipeline.debug import draw_faces
    from pipeline.detector import FaceDetector

    img_path = (
        Path(__file__).resolve().parent.parent
        / "faces"
        / "35fd8561b499fe326d720853003b2aa9.jpg"
    )
    if not img_path.exists():
        pytest.skip(f"测试图片不存在: {img_path}")

    frame = cv2.imread(str(img_path))
    assert frame is not None, f"无法读取图片: {img_path}"

    try:
        detector = FaceDetector()
    except FileNotFoundError as e:
        logger.error(f"模型文件缺失: {e}")
        pytest.skip(f"模型文件缺失: {e}")

    faces = detector.detect(frame)

    annotated = draw_faces(frame, faces)

    output_dir = Path(__file__).resolve().parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "debug_detector_faces.jpg"
    cv2.imwrite(str(output_path), annotated)
    assert output_path.exists(), f"标注图片未生成: {output_path}"
