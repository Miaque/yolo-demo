from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest


def _mock_box(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> MagicMock:
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
        _mock_result([
            _mock_box(0, 0, 100, 200),
            _mock_box(200, 0, 300, 200),
        ])
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
