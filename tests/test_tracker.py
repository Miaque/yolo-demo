from unittest.mock import MagicMock, patch
import numpy as np


@patch("pipeline.tracker.ByteTrack")
def test_update_returns_track_list(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    # ByteTrack.update 返回 shape (N,8): [x1,y1,x2,y2,id,conf,cls,idx]
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    results = tracker.update([(10, 20, 100, 120, 0.9)], frame)

    assert len(results) == 1
    tid, bbox = results[0]
    assert tid == 1
    assert bbox == [10, 20, 100, 120]


@patch("pipeline.tracker.ByteTrack")
def test_predict_returns_last_results(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)

    assert tracker.predict() == [(1, [10, 20, 100, 120])]


@patch("pipeline.tracker.ByteTrack")
def test_removed_ids_when_track_disappears(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.side_effect = [
        # 第一次：ID 1 和 ID 2
        np.array([
            [10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0],
            [200.0, 20.0, 300.0, 120.0, 2.0, 0.8, 0.0, 0.0],
        ]),
        # 第二次：只有 ID 1
        np.array([
            [10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0],
        ]),
    ]
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)

    removed = tracker.removed_ids()
    assert 2 in removed
    assert 1 not in removed


@patch("pipeline.tracker.ByteTrack")
def test_empty_detections_keeps_last(mock_bt_cls):
    from pipeline.tracker import FaceTracker

    mock_bt = MagicMock()
    mock_bt.update.return_value = np.array(
        [[10.0, 20.0, 100.0, 120.0, 1.0, 0.9, 0.0, 0.0]]
    )
    mock_bt_cls.return_value = mock_bt

    tracker = FaceTracker()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracker.update([(10, 20, 100, 120, 0.9)], frame)
    # 无检测帧
    results = tracker.predict()
    assert results == [(1, [10, 20, 100, 120])]
