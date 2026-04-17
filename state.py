# state.py
import threading
from typing import Optional

import numpy as np


class TrackState:
    """线程安全的跟踪状态，存储 track 的 embedding 和匹配名字。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embeddings: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}

    def get_embedding(self, track_id: int) -> Optional[np.ndarray]:
        with self._lock:
            return self._embeddings.get(track_id)

    def set_embedding(self, track_id: int, embedding: np.ndarray) -> None:
        with self._lock:
            self._embeddings[track_id] = embedding

    def remove_embedding(self, track_id: int) -> None:
        """移除 embedding 及其关联的名字。"""
        with self._lock:
            self._embeddings.pop(track_id, None)
            self._names.pop(track_id, None)

    def get_name(self, track_id: int) -> Optional[str]:
        with self._lock:
            return self._names.get(track_id)

    def set_name(self, track_id: int, name: str) -> None:
        with self._lock:
            self._names[track_id] = name

    def snapshot(self) -> tuple[dict[int, np.ndarray], dict[int, str]]:
        """返回两个 store 的浅拷贝，用于主线程传递给 overlay。"""
        with self._lock:
            return dict(self._embeddings), dict(self._names)
