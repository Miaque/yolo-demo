# overlay.py
import cv2
import numpy as np

_GREEN = (0, 255, 0)    # 已有 Embedding
_ORANGE = (0, 165, 255)  # 尚无 Embedding


def draw_tracks(
    frame: np.ndarray,
    tracks: list[tuple[int, list[int]]],
    emb_snapshot: dict[int, np.ndarray],
) -> np.ndarray:
    """在帧副本上绘制跟踪框和 ID 标签，返回新数组。

    Args:
        frame:        BGR 帧（不修改原始帧）
        tracks:       [(track_id, [x1, y1, x2, y2]), ...]
        emb_snapshot: embedding_store 的快照（state.snapshot() 的返回值）
    """
    out = frame.copy()
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = _GREEN if track_id in emb_snapshot else _ORANGE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id}" + (" [E]" if track_id in emb_snapshot else "")
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
