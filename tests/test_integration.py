"""
集成冒烟测试：合成帧 → 主循环逻辑 → overlay 输出
不依赖真实 RTSP 流、YOLO 模型或 InsightFace。
"""
import queue
import numpy as np
from unittest.mock import MagicMock, patch
from state import TrackState
import overlay
from config import settings


def _make_frame() -> np.ndarray:
    return np.zeros((settings.INPUT_HEIGHT, settings.INPUT_WIDTH, 3), dtype=np.uint8)


@patch("pipeline.tracker.ByteTrack")
@patch("pipeline.detector.YOLO")
def test_pipeline_loop_processes_frames(mock_yolo_cls, mock_bt_cls):
    """主循环处理 10 帧不抛出异常，output_queue 收到标注帧。"""
    from pipeline.detector import FaceDetector
    from pipeline.tracker import FaceTracker

    # Mock YOLO: 无检测结果
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    # Mock ByteTrack: 无跟踪结果
    mock_bt = MagicMock()
    mock_bt.update.return_value = np.empty((0, 8), dtype=np.float32)
    mock_bt_cls.return_value = mock_bt

    ts = TrackState()
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
    output_queue: queue.Queue = queue.Queue(maxsize=10)

    detector = FaceDetector()
    tracker = FaceTracker()
    track_results: list = []
    frame_count = 0

    # 注入 10 帧合成帧
    for _ in range(10):
        frame_queue.put(_make_frame())

    # 模拟主循环逻辑
    while not frame_queue.empty():
        frame = frame_queue.get_nowait()
        frame_count += 1

        if frame_count % settings.DETECT_INTERVAL == 0:
            detections = detector.detect(frame)
            track_results = tracker.update(detections, frame)
        else:
            track_results = tracker.predict()

        emb_snapshot, name_snapshot = ts.snapshot()
        annotated = overlay.draw_tracks(frame, track_results, emb_snapshot, name_snapshot)
        output_queue.put_nowait(annotated)

    assert output_queue.qsize() == 10
    # 所有输出帧形状正确
    while not output_queue.empty():
        f = output_queue.get_nowait()
        assert f.shape == (settings.INPUT_HEIGHT, settings.INPUT_WIDTH, 3)


@patch("pipeline.tracker.ByteTrack")
@patch("pipeline.detector.YOLO")
def test_new_track_id_queued_for_embedding(mock_yolo_cls, mock_bt_cls):
    """新 track_id 出现时，人脸裁剪应进入 face_crop_queue。"""
    from pipeline.detector import FaceDetector
    from pipeline.tracker import FaceTracker

    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    mock_bt = MagicMock()
    # 返回一个 track: id=1, bbox=(50,50,150,150)
    mock_bt.update.return_value = np.array(
        [[50.0, 50.0, 150.0, 150.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    face_crop_queue: queue.Queue = queue.Queue(maxsize=settings.FACE_CROP_QUEUE_SIZE)

    detector = FaceDetector()
    tracker = FaceTracker()
    submitted_ids: set[int] = set()

    frame = _make_frame()
    # 触发检测帧
    detections = detector.detect(frame)
    track_results = tracker.update(detections, frame)

    for track_id, bbox in track_results:
        if track_id not in submitted_ids:
            submitted_ids.add(track_id)
            x1, y1, x2, y2 = bbox
            face_crop = frame[max(0, y1):min(frame.shape[0], y2),
                              max(0, x1):min(frame.shape[1], x2)].copy()
            try:
                face_crop_queue.put_nowait((track_id, face_crop))
            except queue.Full:
                pass

    assert face_crop_queue.qsize() == 1
    queued_id, queued_crop = face_crop_queue.get_nowait()
    assert queued_id == 1
    assert queued_crop.ndim == 3
