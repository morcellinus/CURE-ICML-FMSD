from __future__ import annotations

import gc
from time import perf_counter
from types import MethodType
from typing import Any

import numpy as np
import torch


class TabICLStreamWrapper:

    def __init__(
        self,
        *,
        device: str,
        random_state: int,
        model_path: str | None = None,
        checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt",
        n_estimators: int = 1,
        kv_cache: bool | str = False,
        batch_size: int | None = 1,
        use_amp: bool | str = "auto",
        use_fa3: bool | str = "auto",
        offload_mode: str | bool = "auto",
        aggressive_cuda_cleanup: bool = True,
    ) -> None:
        if n_estimators != 1:
            raise ValueError("This streaming pipeline fixes TabICL n_estimators=1.")
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TabICL is not installed. Install it with `pip install tabicl` or add the TabICL repo to PYTHONPATH."
            ) from exc

        self._device = str(device)
        self._aggressive_cuda_cleanup = bool(aggressive_cuda_cleanup)
        self._model_loaded_once = False
        self._original_load_model = None
        self.clf = TabICLClassifier(
            n_estimators=1,
            kv_cache=kv_cache,
            batch_size=batch_size,
            model_path=None if model_path in {None, "", "auto"} else model_path,
            checkpoint_version=checkpoint_version,
            device=device,
            use_amp=use_amp,
            use_fa3=use_fa3,
            offload_mode=offload_mode,
            random_state=random_state,
            verbose=False,
        )
        self._patch_model_loader()

    def _patch_model_loader(self) -> None:
        self._original_load_model = self.clf._load_model

        def _load_model_reuse(estimator: Any) -> None:
            if getattr(estimator, "model_", None) is not None:
                self._model_loaded_once = True
                return
            assert self._original_load_model is not None
            self._original_load_model()
            self._model_loaded_once = True

        self.clf._load_model = MethodType(_load_model_reuse, self.clf)

    def _cleanup_cuda(self) -> None:
        if not self._aggressive_cuda_cleanup:
            return
        if not self._device.startswith("cuda") or not torch.cuda.is_available():
            return
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        t0 = perf_counter()
        self.clf.fit(np.ascontiguousarray(X), np.ascontiguousarray(y))
        elapsed = perf_counter() - t0
        self._cleanup_cuda()
        return elapsed

    def predict(self, x: np.ndarray) -> tuple[int, np.ndarray, float]:
        t0 = perf_counter()
        probs = np.asarray(self.clf.predict_proba(np.ascontiguousarray(x.reshape(1, -1)))[0], dtype=np.float64)
        pred_encoded = int(np.argmax(probs))
        pred = int(self.clf.classes_[pred_encoded]) if hasattr(self.clf, "classes_") else pred_encoded
        elapsed = perf_counter() - t0
        self._cleanup_cuda()
        return pred, probs, elapsed
