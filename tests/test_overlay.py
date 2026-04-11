import numpy as np
import overlay


def _blank(h: int = 360, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_returns_new_array():
    frame = _blank()
    result = overlay.draw_tracks(frame, [], {})
    assert result is not frame


def test_shape_preserved():
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    assert result.shape == frame.shape


def test_empty_tracks_unchanged():
    frame = _blank()
    result = overlay.draw_tracks(frame, [], {})
    np.testing.assert_array_equal(result, frame)


def test_bbox_pixels_differ_from_blank():
    """绘制后 BBox 区域应有像素变化。"""
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    # BBox 边框区域至少有一个非零像素
    assert result[10, 10:101].any()


def test_green_when_embedding_present():
    """有 Embedding 时用绿色（BGR: 0,255,0）。"""
    frame = _blank()
    emb = np.zeros(512, dtype=np.float32)
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {1: emb})
    # 检查 BBox 边框上存在绿色像素
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 255, 0]) for px in top_row)


def test_orange_when_no_embedding():
    """无 Embedding 时用橙色（BGR: 0,165,255）。"""
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)
