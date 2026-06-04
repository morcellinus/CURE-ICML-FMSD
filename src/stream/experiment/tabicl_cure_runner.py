from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..data import build_dataset
from ..memory import build_cure_policy
from ..model.tabicl_stream_wrapper import TabICLStreamWrapper
from .metrics import StreamingMetrics


@dataclass(slots=True)
class RunConfig:
    dataset: str
    data_root: str
    output_root: str
    tau: float
    budget_B: int = 1000
    short_ratio: float = 0.75
    warmup_steps: int = 100
    max_stream_length: int | None = None
    device: str = "cuda"
    seed: int = 0
    threads: int = 4
    progress_every: int = 50
    eval_after_warmup_only: bool = True
    aggressive_cuda_cleanup: bool = False
    tabicl_checkpoint_version: str = "tabicl-classifier-v2-20260212.ckpt"
    tabicl_model_path: str = "auto"



def _configure_torch_threads(threads: int) -> None:
    threads = int(threads or 0)
    if threads <= 0:
        return
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(threads, 4)))
    except Exception:
        pass


def _configure_cuda_attention_backends(device: str) -> None:
    if not torch.cuda.is_available():
        return
    if not str(device).startswith("cuda"):
        return
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:
        pass

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    import json
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_records_csv(records: list[dict[str, Any]], path: str | Path) -> None:
    pd.DataFrame(records).to_csv(path, index=False)


def run_tabicl_cure(cfg: RunConfig) -> dict[str, Any]:
    _configure_torch_threads(cfg.threads)
    _configure_cuda_attention_backends(cfg.device)
    dataset = build_dataset(cfg.dataset, cfg.data_root)
    data = dataset.load()
    n_total = data.X.shape[0]
    if cfg.max_stream_length is not None:
        n_total = min(n_total, int(cfg.max_stream_length))

    out_dir = ensure_dir(Path(cfg.output_root) / cfg.dataset)
    model = TabICLStreamWrapper(
        device=cfg.device,
        random_state=cfg.seed,
        model_path=cfg.tabicl_model_path,
        checkpoint_version=cfg.tabicl_checkpoint_version,
        n_estimators=1,
        kv_cache=False,
        batch_size=1,
        use_amp="auto",
        use_fa3="auto",
        offload_mode="auto",
        aggressive_cuda_cleanup=cfg.aggressive_cuda_cleanup,
    )
    memory = build_cure_policy(
        budget_B=cfg.budget_B,
        short_ratio=cfg.short_ratio,
        entropy_threshold=cfg.tau,
        seed=cfg.seed,
    )

    step_logs: list[dict[str, Any]] = []
    metrics = StreamingMetrics()
    n_pred_evaluated = 0
    fit_time_sum = 0.0
    pred_time_sum = 0.0
    eviction_time_sum = 0.0
    wall_start = time.perf_counter()

    iterator = range(n_total)
    use_tqdm = cfg.progress_every > 0
    if use_tqdm:
        iterator = tqdm(iterator, desc=f"{cfg.dataset}:cure", dynamic_ncols=True)

    for t in iterator:
        x_t = data.X[t]
        y_t = int(data.y[t])
        labels_now = memory.labels() if memory.size() > 0 else np.empty((0,), dtype=np.int64)
        can_predict = memory.size() >= 2 and np.unique(labels_now).size >= 2
        should_eval = (t >= cfg.warmup_steps) or (not cfg.eval_after_warmup_only)

        y_hat = None
        prob_max = None
        step_fit_s = 0.0
        step_pred_s = 0.0
        if can_predict and should_eval:
            X_mem, y_mem = memory.arrays()
            step_fit_s = float(model.fit(X_mem, y_mem))
            y_hat, probs, step_pred_s = model.predict(x_t)
            prob_max = float(np.max(probs))
            metric_values = metrics.update(y_t, y_hat)
            n_pred_evaluated += 1
            memory.record_prediction_signal(y_true=y_t, y_pred=y_hat, probs=probs)
        else:
            metric_values = {
                "cum_accuracy": float("nan"),
                "cum_balanced_accuracy": float("nan"),
                "cum_macro_f1": float("nan"),
            }
            memory.record_prediction_signal(y_true=y_t, y_pred=None, probs=None)

        fit_time_sum += step_fit_s
        pred_time_sum += step_pred_s

        ev_start = time.perf_counter()
        memory.append_then_evict(x_t, y_t, t)
        step_evict_s = time.perf_counter() - ev_start
        eviction_time_sum += step_evict_s
        step_logs.append(
            {
                "t": t,
                "pred": y_hat,
                "truth": y_t,
                "prob_max": prob_max,
                "memory_size": memory.size(),
                "short_size": int(memory.short_size()) if hasattr(memory, "short_size") else int(memory.size()),
                "long_size": int(memory.long_size()) if hasattr(memory, "long_size") else 0,
                "fit_s": step_fit_s,
                "predict_s": step_pred_s,
                "evict_s": step_evict_s,
                **metric_values,
            }
        )
        if use_tqdm:
            acc = step_logs[-1]["cum_accuracy"]
            iterator.set_postfix(cum_acc="nan" if np.isnan(acc) else f"{acc:.4f}")
        elif cfg.progress_every > 0 and (((t + 1) % cfg.progress_every == 0) or ((t + 1) == n_total)):
            acc = step_logs[-1]["cum_accuracy"]
            print(f"[progress] step={t+1}/{n_total} cum_acc={'nan' if np.isnan(acc) else f'{acc:.4f}'}", flush=True)

    total_runtime_s = time.perf_counter() - wall_start
    summary = {
        "config": asdict(cfg),
        "n_stream_steps": int(n_total),
        "n_evaluated_predictions": int(n_pred_evaluated),
        "total_runtime_s": float(total_runtime_s),
        "avg_step_latency_s": float(total_runtime_s / max(n_total, 1)),
        "fit_time_total_s": float(fit_time_sum),
        "predict_time_total_s": float(pred_time_sum),
        "eviction_time_total_s": float(eviction_time_sum),
        "final_memory_size": int(memory.size()),
        "final_short_size": int(memory.short_size()) if hasattr(memory, "short_size") else int(memory.size()),
        "final_long_size": int(memory.long_size()) if hasattr(memory, "long_size") else 0,
        "cum_accuracy": step_logs[-1]["cum_accuracy"] if step_logs else float("nan"),
        "cum_balanced_accuracy": step_logs[-1]["cum_balanced_accuracy"] if step_logs else float("nan"),
        "cum_macro_f1": step_logs[-1]["cum_macro_f1"] if step_logs else float("nan"),
        "notes": "",
    }
    save_json(summary, out_dir / "summary.json")
    save_json({"config": asdict(cfg)}, out_dir / "run_config.json")
    save_records_csv(step_logs, out_dir / "step_metrics.csv")
    return {"summary": summary, "output_dir": str(out_dir)}
