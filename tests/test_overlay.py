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


def test_orange_when_no_name_snapshot():
    """无 name_snapshot 时默认 Unknown，用橙色（BGR: 0,165,255）。"""
    frame = _blank()
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {})
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)


def test_orange_when_embedding_but_no_name():
    """有 Embedding 但无 name_snapshot 时仍为橙色（向后兼容）。"""
    frame = _blank()
    emb = np.zeros(512, dtype=np.float32)
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {1: emb})
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)


def test_matched_name_shown_in_blue():
    """匹配到名字时用蓝色框。"""
    frame = _blank()
    name_snap = {1: "Alice"}
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {}, name_snap)
    top_row = result[10, 10:101]
    # 蓝色 BGR: (255, 0, 0)
    assert any(np.array_equal(px, [255, 0, 0]) for px in top_row)


def test_unknown_name_shown_in_orange():
    """未匹配到名字时用橙色框。"""
    frame = _blank()
    name_snap = {1: "Unknown"}
    result = overlay.draw_tracks(frame, [(1, [10, 10, 100, 100])], {}, name_snap)
    top_row = result[10, 10:101]
    assert any(np.array_equal(px, [0, 165, 255]) for px in top_row)
