# state.py
import threading
from typing import Optional
import numpy as np

lock = threading.Lock()
embedding_store: dict[int, np.ndarray] = {}


def get_embedding(track_id: int) -> Optional[np.ndarray]:
    with lock:
        return embedding_store.get(track_id)


def set_embedding(track_id: int, embedding: np.ndarray) -> None:
    with lock:
        embedding_store[track_id] = embedding


def remove_embedding(track_id: int) -> None:
    with lock:
        embedding_store.pop(track_id, None)


def snapshot() -> dict[int, np.ndarray]:
    """返回当前 store 的浅拷贝，用于主线程传递给 overlay（避免竞态）。"""
    with lock:
        return dict(embedding_store)


def clear() -> None:
    with lock:
        embedding_store.clear()
