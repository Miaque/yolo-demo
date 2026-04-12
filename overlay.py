# overlay.py
import cv2
import numpy as np

_BLUE = (255, 0, 0)      # 已识别（有名字）
_ORANGE = (0, 165, 255)   # 未识别（Unknown 或无名字）


def draw_tracks(
    frame: np.ndarray,
    tracks: list[tuple[int, list[int]]],
    emb_snapshot: dict[int, np.ndarray],
    name_snapshot: dict[int, str] | None = None,
) -> np.ndarray:
    """在帧副本上绘制跟踪框和标签，返回新数组。

    Args:
        frame:          BGR 帧（不修改原始帧）
        tracks:         [(track_id, [x1, y1, x2, y2]), ...]
        emb_snapshot:   embedding_store 的快照
        name_snapshot:  name_store 的快照（可选，向后兼容）
    """
    if name_snapshot is None:
        name_snapshot = {}

    out = frame.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        name = name_snapshot.get(track_id, "Unknown")

        if name != "Unknown":
            color = _BLUE
            label = f"{name} (ID:{track_id})"
        else:
            color = _ORANGE
            label = f"Unknown (ID:{track_id})"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
