from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .base import BaseMemoryPolicy, EvictionEvent


@dataclass(slots=True)
class _CUREMeta:
    entropy: float
    margin_uncertainty: float
    pred: int | None


class CUREMemoryPolicy(BaseMemoryPolicy):
    """Exact main-experiment CURE policy for the TabICL-v2 bundle.

    This keeps only the single configuration used in the main paper experiments:
    - dual memory with short/long split
    - warm-fill long bank until capacity
    - normalized entropy-gated admission after warm-fill
    - majority-class nearest same-class pair redundancy detection
    - class-conditional short-bank centroid endpoint selection
    """

    def __init__(
        self,
        *,
        budget_B: int,
        short_ratio: float,
        entropy_threshold: float,
    ) -> None:
        super().__init__(budget_B=budget_B, anchors_K=0, eviction_chunk_M=1)
        if not 0.0 < short_ratio < 1.0:
            raise ValueError("short_ratio must be in (0, 1)")
        self.short_capacity = max(1, int(round(budget_B * short_ratio)))
        self.long_capacity = max(1, budget_B - self.short_capacity)
        if self.long_capacity <= 0:
            self.long_capacity = 1
            self.short_capacity = max(1, budget_B - 1)

        self.entropy_threshold = float(entropy_threshold)
        self.warm_fill_ratio = 1.0
        self.class_conditional_short_centroid = True
        self.hardness_aware_append = True
        self.redundancy_aware_eviction = True

        self._is_short: list[bool] = []
        self._meta: list[_CUREMeta] = []
        self._next_meta = _CUREMeta(entropy=0.0, margin_uncertainty=0.0, pred=None)

    def record_prediction_signal(self, *, y_true: int, y_pred: int | None, probs: np.ndarray | None = None) -> None:
        del y_true
        entropy = 0.0
        margin_uncertainty = 0.0
        if probs is not None:
            p = np.asarray(probs, dtype=np.float64).reshape(-1)
            p = np.clip(p, 1e-12, 1.0)
            p = p / max(1e-12, float(p.sum()))
            raw_entropy = float(-(p * np.log(p)).sum())
            max_entropy = float(np.log(max(int(p.size), 1)))
            entropy = float(raw_entropy / max_entropy) if max_entropy > 1e-12 else 0.0
            if p.size > 1:
                top2 = np.sort(p)[-2:]
                margin = float(top2[-1] - top2[-2])
                margin_uncertainty = float(np.clip(1.0 - margin, 0.0, 1.0))
        self._next_meta = _CUREMeta(entropy=entropy, margin_uncertainty=margin_uncertainty, pred=None if y_pred is None else int(y_pred))

    def short_size(self) -> int:
        return int(np.sum(np.asarray(self._is_short, dtype=bool)))

    def long_size(self) -> int:
        return self.size() - self.short_size()

    def short_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self._is_short, dtype=bool)).astype(np.int64)

    def long_indices(self) -> np.ndarray:
        return np.flatnonzero(~np.asarray(self._is_short, dtype=bool)).astype(np.int64)

    def _meta_score(self, meta: _CUREMeta) -> float:
        return float(meta.entropy)

    def _normalized_joint(self, indices: list[int]) -> np.ndarray:
        if not indices:
            return np.empty((0, 0), dtype=np.float32)
        X = np.stack([self._X[i] for i in indices], axis=0).astype(np.float32, copy=False)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-6
        return (X - mean) / std

    def _pairwise_nearest_pair(self, X: np.ndarray) -> tuple[int, int, float] | None:
        if X.shape[0] < 2:
            return None
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2)) / np.sqrt(max(1, X.shape[1]))
        np.fill_diagonal(dist, np.inf)
        pair_flat = int(np.argmin(dist))
        a_pos, b_pos = np.unravel_index(pair_flat, dist.shape)
        return int(a_pos), int(b_pos), float(dist[int(a_pos), int(b_pos)])

    def _pick_long_drop_index(self) -> int | None:
        long_indices = [i for i, is_short in enumerate(self._is_short) if not is_short]
        if not long_indices:
            return None

        labels = np.asarray([self._y[i] for i in long_indices], dtype=np.int64)
        counts = Counter(labels.tolist())
        majority_class = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        same = [i for i in long_indices if self._y[i] == majority_class]
        if len(same) <= 1:
            return same[0] if same else long_indices[0]

        short_indices = [i for i, is_short in enumerate(self._is_short) if is_short]
        short_same = [i for i in short_indices if self._y[i] == majority_class]
        centroid_indices = short_same if short_same else short_indices

        all_indices = same + centroid_indices
        X_all = self._normalized_joint(all_indices)
        X_same = X_all[: len(same)]
        if len(centroid_indices) > 0:
            short_centroid = X_all[len(same) :].mean(axis=0)
        else:
            short_centroid = X_same.mean(axis=0)

        pair = self._pairwise_nearest_pair(X_same)
        if pair is None:
            return same[0]
        a_pos, b_pos, _ = pair
        idx_a = same[a_pos]
        idx_b = same[b_pos]

        da = float(np.linalg.norm(X_same[a_pos] - short_centroid) / np.sqrt(max(1, X_same.shape[1])))
        db = float(np.linalg.norm(X_same[b_pos] - short_centroid) / np.sqrt(max(1, X_same.shape[1])))
        if da > db:
            return idx_a
        if db > da:
            return idx_b
        return idx_a if self._t[idx_a] <= self._t[idx_b] else idx_b

    def _drop_with_meta(self, idx: int) -> list[int]:
        old_t = np.asarray(self._t, dtype=np.int64)
        ev_ts = old_t[np.asarray([idx], dtype=np.int64)].tolist()
        keep_ix = [i for i in range(self.size()) if i != idx]
        self._invalidate_arrays_cache()
        self._X = [self._X[i] for i in keep_ix]
        self._y = [self._y[i] for i in keep_ix]
        self._t = [self._t[i] for i in keep_ix]
        self._is_short = [self._is_short[i] for i in keep_ix]
        self._meta = [self._meta[i] for i in keep_ix]
        return ev_ts

    def append_then_evict(self, x: np.ndarray, y: int, t: int) -> EvictionEvent:
        BaseMemoryPolicy.append(self, x, y, t)
        self._is_short.append(True)
        self._meta.append(self._next_meta)
        self._next_meta = _CUREMeta(entropy=0.0, margin_uncertainty=0.0, pred=None)

        evicted: list[int] = []
        evicted_ts: list[int] = []
        rank_s = 0.0
        apply_s = 0.0
        warm_fill_target = int(round(self.long_capacity * self.warm_fill_ratio))

        while self.short_size() > self.short_capacity:
            idx = next((i for i, is_short in enumerate(self._is_short) if is_short), None)
            if idx is None:
                break
            cand_score = self._meta_score(self._meta[idx])
            bypass_gate = self.long_size() < warm_fill_target
            if (not bypass_gate) and cand_score < self.entropy_threshold:
                t_apply = perf_counter()
                evicted_ts.extend(self._drop_with_meta(idx))
                apply_s += perf_counter() - t_apply
                continue
            self._is_short[idx] = False

        while self.long_size() > self.long_capacity:
            t_rank = perf_counter()
            idx = self._pick_long_drop_index()
            rank_s += perf_counter() - t_rank
            if idx is None:
                break
            evicted.append(int(idx))
            t_apply = perf_counter()
            evicted_ts.extend(self._drop_with_meta(int(idx)))
            apply_s += perf_counter() - t_apply

        triggered = bool(evicted or evicted_ts)
        return EvictionEvent(
            triggered=triggered,
            evicted_indices=evicted,
            evicted_timestamps=evicted_ts,
            details={
                "policy": "cure",
                "short_size": int(self.short_size()),
                "long_size": int(self.long_size()),
                "short_capacity": int(self.short_capacity),
                "long_capacity": int(self.long_capacity),
                "evict_rank_s": float(rank_s),
                "evict_apply_s": float(apply_s),
            },
        )

    def _evict_once(self) -> EvictionEvent:
        raise NotImplementedError("CUREMemoryPolicy uses append_then_evict directly.")
