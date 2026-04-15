# pipeline/tracker.py
import numpy as np
from boxmot import ByteTrack

from config import settings


class FaceTracker:
    """ByteTrack 封装，暴露简洁的 update / predict / removed_ids 接口。"""

    def __init__(self) -> None:
        self._tracker = ByteTrack(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=settings.BYTETRACK_MAX_AGE,
        )
        self._last_tracks: list[tuple[int, list[int]]] = []
        self._active_ids: set[int] = set()
        self._removed_ids: set[int] = set()
        self._velocity: dict[int, tuple[float, float]] = {}
        self._predict_count: int = 0

    def update(
        self,
        detections: list[tuple[int, int, int, int, float]],
        frame: np.ndarray,
    ) -> list[tuple[int, list[int]]]:
        """用新检测结果更新 ByteTrack，返回 [(track_id, [x1,y1,x2,y2])]。"""
        # 记录上一次检测位置，用于计算速度
        prev_pos: dict[int, list[int]] = {tid: bbox for tid, bbox in self._last_tracks}

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

            # 计算每帧速度（基于中心点位移）
            if tid in prev_pos:
                old = prev_pos[tid]
                cx_old = (old[0] + old[2]) / 2
                cy_old = (old[1] + old[3]) / 2
                cx_new = (bbox[0] + bbox[2]) / 2
                cy_new = (bbox[1] + bbox[3]) / 2
                interval = max(settings.DETECT_INTERVAL, 1)
                self._velocity[tid] = (
                    (cx_new - cx_old) / interval,
                    (cy_new - cy_old) / interval,
                )

        # 清除已消失 track 的速度
        gone_ids = set(self._velocity.keys()) - new_ids
        for tid in gone_ids:
            self._velocity.pop(tid, None)

        self._removed_ids = self._active_ids - new_ids
        self._active_ids = new_ids
        self._last_tracks = results
        self._predict_count = 0
        return results

    def predict(self) -> list[tuple[int, list[int]]]:
        """非检测帧调用，基于上次检测位置和速度做线性外推。"""
        self._predict_count += 1
        n = self._predict_count
        results: list[tuple[int, list[int]]] = []
        for tid, bbox in self._last_tracks:
            if tid in self._velocity:
                vx, vy = self._velocity[tid]
                new_bbox = [
                    int(bbox[0] + vx * n),
                    int(bbox[1] + vy * n),
                    int(bbox[2] + vx * n),
                    int(bbox[3] + vy * n),
                ]
                results.append((tid, new_bbox))
            else:
                results.append((tid, bbox))
        return results

    def removed_ids(self) -> set[int]:
        """返回上次 update() 后消失的 track_id 集合。"""
        return self._removed_ids
