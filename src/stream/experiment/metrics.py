from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


@dataclass(slots=True)
class StreamingMetrics:
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)

    def update(self, y_t: int, y_hat: int) -> dict[str, float]:
        self.y_true.append(int(y_t))
        self.y_pred.append(int(y_hat))
        yt = np.asarray(self.y_true, dtype=np.int64)
        yp = np.asarray(self.y_pred, dtype=np.int64)

        acc = float((yt == yp).mean())
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                bal_acc = float(balanced_accuracy_score(yt, yp))
        except Exception:
            bal_acc = float("nan")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        except Exception:
            macro_f1 = float("nan")

        return {
            "cum_accuracy": acc,
            "cum_balanced_accuracy": bal_acc,
            "cum_macro_f1": macro_f1,
        }
