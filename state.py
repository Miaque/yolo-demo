# state.py
import threading
from typing import Optional
import numpy as np

_lock = threading.Lock()
embedding_store: dict[int, np.ndarray] = {}
name_store: dict[int, str] = {}


def get_embedding(track_id: int) -> Optional[np.ndarray]:
    with _lock:
        return embedding_store.get(track_id)


def set_embedding(track_id: int, embedding: np.ndarray) -> None:
    with _lock:
        embedding_store[track_id] = embedding


def remove_embedding(track_id: int) -> None:
    with _lock:
        embedding_store.pop(track_id, None)
        name_store.pop(track_id, None)


def get_name(track_id: int) -> Optional[str]:
    with _lock:
        return name_store.get(track_id)


def set_name(track_id: int, name: str) -> None:
    with _lock:
        name_store[track_id] = name


def remove_name(track_id: int) -> None:
    with _lock:
        name_store.pop(track_id, None)


def snapshot() -> tuple[dict[int, np.ndarray], dict[int, str]]:
    """返回两个 store 的浅拷贝，用于主线程传递给 overlay。"""
    with _lock:
        return dict(embedding_store), dict(name_store)


def clear() -> None:
    with _lock:
        embedding_store.clear()
        name_store.clear()
