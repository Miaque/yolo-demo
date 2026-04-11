# pipeline/tracker.py
from boxmot import ByteTrack
import numpy as np
import config


class FaceTracker:
    """ByteTrack 封装，暴露简洁的 update / predict / removed_ids 接口。"""

    def __init__(self) -> None:
        self._tracker = ByteTrack(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=config.BYTETRACK_MAX_AGE,
        )
        self._last_tracks: list[tuple[int, list[int]]] = []
        self._active_ids: set[int] = set()
        self._removed_ids: set[int] = set()

    def update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame: np.ndarray,
    ) -> list[tuple[int, list[int]]]:
        """用新检测结果更新 ByteTrack，返回 [(track_id, [x1,y1,x2,y2])]。"""
        if detections:
            dets = np.array(
                [[x1, y1, x2, y2, conf, 0] for x1, y1, x2, y2, conf in detections],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)

        raw = self._tracker.update(dets, frame)

        new_ids: set[int] = set()
        results: list[tuple[int, list[int]]] = []
        for t in raw:
            tid = int(t[4])
            bbox = [int(t[0]), int(t[1]), int(t[2]), int(t[3])]
            results.append((tid, bbox))
            new_ids.add(tid)

        self._removed_ids = self._active_ids - new_ids
        self._active_ids = new_ids
        self._last_tracks = results
        return results

    def predict(self) -> list[tuple[int, list[int]]]:
        """非检测帧调用，直接返回上一帧的跟踪结果（Kalman 预测由 ByteTrack 内部维护）。"""
        return self._last_tracks

    def removed_ids(self) -> set[int]:
        """返回上次 update() 后消失的 track_id 集合。"""
        return self._removed_ids
