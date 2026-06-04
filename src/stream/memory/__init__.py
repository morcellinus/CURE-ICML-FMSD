from __future__ import annotations

from .cure_memory import CUREMemoryPolicy


def build_cure_policy(*, budget_B: int, short_ratio: float, entropy_threshold: float, seed: int = 0):
    del seed
    return CUREMemoryPolicy(
        budget_B=budget_B,
        short_ratio=short_ratio,
        entropy_threshold=entropy_threshold,
    )


__all__ = ["CUREMemoryPolicy", "build_cure_policy"]
