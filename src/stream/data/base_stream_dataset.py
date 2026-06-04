from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class StreamData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_name: str


class BaseStreamDataset(ABC):
    name: str

    @abstractmethod
    def load(self) -> StreamData:
        """Load stream data in stream order."""

    def maybe_shuffle(
        self,
        data: StreamData,
        *,
        shuffle: bool,
        seed: int,
    ) -> StreamData:
        if not shuffle:
            return data
        rng = np.random.default_rng(seed)
        order = rng.permutation(data.X.shape[0])
        return StreamData(
            X=np.ascontiguousarray(data.X[order]),
            y=np.ascontiguousarray(data.y[order]),
            feature_names=data.feature_names,
            target_name=data.target_name,
        )
