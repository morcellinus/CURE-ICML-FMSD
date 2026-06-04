from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EvictionEvent:
    triggered: bool
    evicted_indices: list[int]
    evicted_timestamps: list[int]
    details: dict


class BaseMemoryPolicy(ABC):
    def __init__(self, *, budget_B: int | None, anchors_K: int, eviction_chunk_M: int) -> None:
        self.budget_B = budget_B
        self.anchors_K = anchors_K
        self.eviction_chunk_M = eviction_chunk_M

        self._X: list[np.ndarray] = []
        self._y: list[int] = []
        self._t: list[int] = []
        self._arrays_cache: tuple[np.ndarray, np.ndarray] | None = None

    def _invalidate_arrays_cache(self) -> None:
        self._arrays_cache = None

    def size(self) -> int:
        return len(self._y)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._X:
            raise ValueError("Memory is empty.")
        if self._arrays_cache is None:
            X = np.ascontiguousarray(np.stack(self._X, axis=0))
            y = np.ascontiguousarray(np.asarray(self._y, dtype=np.int64))
            self._arrays_cache = (X, y)
        return self._arrays_cache

    def labels(self) -> np.ndarray:
        return np.asarray(self._y, dtype=np.int64)

    def timestamps(self) -> np.ndarray:
        return np.asarray(self._t, dtype=np.int64)

    def anchor_indices(self) -> np.ndarray:
        if self.anchors_K <= 0 or self.size() == 0:
            return np.empty((0,), dtype=np.int64)
        k = min(self.anchors_K, self.size())
        return np.arange(self.size() - k, self.size(), dtype=np.int64)

    def candidate_indices_non_anchor(self) -> np.ndarray:
        anchors = self.anchor_indices()
        if self.size() == 0:
            return np.empty((0,), dtype=np.int64)
        if anchors.size == 0:
            return np.arange(self.size(), dtype=np.int64)
        return np.arange(0, self.size() - anchors.size, dtype=np.int64)

    def append(self, x: np.ndarray, y: int, t: int) -> None:
        self._invalidate_arrays_cache()
        self._X.append(np.ascontiguousarray(x, dtype=np.float32))
        self._y.append(int(y))
        self._t.append(int(t))

    def _drop_indices(self, drop_indices: np.ndarray) -> list[int]:
        if drop_indices.size == 0:
            return []
        keep_mask = np.ones(self.size(), dtype=bool)
        keep_mask[drop_indices] = False
        old_t = np.asarray(self._t, dtype=np.int64)
        evicted_ts = old_t[drop_indices].tolist()
        self._invalidate_arrays_cache()
        self._X = [self._X[i] for i in np.flatnonzero(keep_mask)]
        self._y = [self._y[i] for i in np.flatnonzero(keep_mask)]
        self._t = [self._t[i] for i in np.flatnonzero(keep_mask)]
        return evicted_ts

    def append_then_evict(self, x: np.ndarray, y: int, t: int) -> EvictionEvent:
        self.append(x, y, t)
        if self.budget_B is None:
            return EvictionEvent(False, [], [], {})
        # Eviction trigger: keep up to B samples after append.
        # This means prediction at the next step sees (up to) B context rows.
        if self.size() <= self.budget_B:
            return EvictionEvent(False, [], [], {})
        return self._evict_once()

    @abstractmethod
    def _evict_once(self) -> EvictionEvent:
        ...
