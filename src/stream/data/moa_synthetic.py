from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base_stream_dataset import BaseStreamDataset, StreamData


@dataclass(frozen=True, slots=True)
class AGRASpec:
    name: str = "agr_a_30k"
    n_samples: int = 30_000
    seed: int = 5
    drift_positions: tuple[int, ...] = (7_500, 15_000, 22_500)
    drift_width: int = 1
    perturbation: float = 0.10
    agrawal_functions: tuple[int, ...] = (0, 3, 6, 9)


AGR_A_SPEC = AGRASpec()


AGRAWAL_FEATURE_NAMES = [
    "salary",
    "commission",
    "age",
    "elevel",
    "car",
    "zipcode",
    "hvalue",
    "hyears",
    "loan",
]


def _build_concept_path(
    n_samples: int,
    concept_ids: tuple[int, ...],
    drift_positions: tuple[int, ...],
    drift_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(concept_ids) == 1:
        return np.full(n_samples, concept_ids[0], dtype=np.int64)
    if len(concept_ids) != len(drift_positions) + 1:
        raise ValueError("concept_ids length must equal len(drift_positions)+1")

    path = np.empty(n_samples, dtype=np.int64)
    start = 0
    current = concept_ids[0]
    for drift_idx, pos in enumerate(drift_positions):
        nxt = concept_ids[drift_idx + 1]
        if drift_width <= 1:
            cut = int(np.clip(pos, 0, n_samples))
            if cut > start:
                path[start:cut] = current
            start = cut
            current = nxt
            continue
        left = max(start, int(pos - drift_width // 2))
        right = min(n_samples, int(pos + drift_width // 2))
        if left > start:
            path[start:left] = current
        if right > left:
            mix = np.linspace(0.0, 1.0, right - left, endpoint=False, dtype=np.float32)
            choose_next = rng.random(right - left) < mix
            path[left:right] = np.where(choose_next, nxt, current)
        start = right
        current = nxt
    if start < n_samples:
        path[start:] = current
    return path


def _agrawal_label(
    function_id: int,
    salary: np.ndarray,
    commission: np.ndarray,
    age: np.ndarray,
    elevel: np.ndarray,
    car: np.ndarray,
    zipcode: np.ndarray,
    hvalue: np.ndarray,
    hyears: np.ndarray,
    loan: np.ndarray,
) -> np.ndarray:
    del car, zipcode
    if function_id == 0:
        return np.where((age < 40) | (age >= 60), 0, 1)
    if function_id == 1:
        return np.where(
            age < 40,
            np.where((salary >= 50000) & (salary <= 100000), 0, 1),
            np.where(
                age < 60,
                np.where((salary >= 75000) & (salary <= 125000), 0, 1),
                np.where((salary >= 25000) & (salary <= 75000), 0, 1),
            ),
        )
    if function_id == 2:
        return np.where(
            age < 40,
            np.where((elevel == 0) | (elevel == 1), 0, 1),
            np.where(
                age < 60,
                np.where((elevel == 1) | (elevel == 2) | (elevel == 3), 0, 1),
                np.where((elevel == 2) | (elevel == 3) | (elevel == 4), 0, 1),
            ),
        )
    if function_id == 3:
        young_hi = np.where((salary >= 25000) & (salary <= 75000), 0, 1)
        young_lo = np.where((salary >= 50000) & (salary <= 100000), 0, 1)
        mid_hi = np.where((salary >= 50000) & (salary <= 100000), 0, 1)
        mid_lo = np.where((salary >= 75000) & (salary <= 125000), 0, 1)
        old_hi = np.where((salary >= 50000) & (salary <= 100000), 0, 1)
        old_lo = np.where((salary >= 25000) & (salary <= 75000), 0, 1)
        return np.where(
            age < 40,
            np.where((elevel == 0) | (elevel == 1), young_hi, young_lo),
            np.where(
                age < 60,
                np.where((elevel == 1) | (elevel == 2) | (elevel == 3), mid_hi, mid_lo),
                np.where((elevel == 2) | (elevel == 3) | (elevel == 4), old_hi, old_lo),
            ),
        )
    if function_id == 4:
        return np.where(
            age < 40,
            np.where(
                (salary >= 50000) & (salary <= 100000),
                np.where((loan >= 100000) & (loan <= 300000), 0, 1),
                np.where((loan >= 200000) & (loan <= 400000), 0, 1),
            ),
            np.where(
                age < 60,
                np.where(
                    (salary >= 75000) & (salary <= 125000),
                    np.where((loan >= 200000) & (loan <= 400000), 0, 1),
                    np.where((loan >= 300000) & (loan <= 500000), 0, 1),
                ),
                np.where(
                    (salary >= 25000) & (salary <= 75000),
                    np.where((loan >= 300000) & (loan <= 500000), 0, 1),
                    np.where((loan >= 100000) & (loan <= 300000), 0, 1),
                ),
            ),
        )
    total_salary = salary + commission
    if function_id == 5:
        return np.where(
            age < 40,
            np.where((total_salary >= 50000) & (total_salary <= 100000), 0, 1),
            np.where(
                age < 60,
                np.where((total_salary >= 75000) & (total_salary <= 125000), 0, 1),
                np.where((total_salary >= 25000) & (total_salary <= 75000), 0, 1),
            ),
        )
    if function_id == 6:
        disposable = 2.0 * total_salary / 3.0 - loan / 5.0 - 20000.0
        return np.where(disposable > 0, 0, 1)
    if function_id == 7:
        disposable = 2.0 * total_salary / 3.0 - 5000.0 * elevel - 20000.0
        return np.where(disposable > 0, 0, 1)
    if function_id == 8:
        disposable = 2.0 * total_salary / 3.0 - 5000.0 * elevel - loan / 5.0 - 10000.0
        return np.where(disposable > 0, 0, 1)
    if function_id == 9:
        equity = np.where(hyears >= 20, hvalue * (hyears - 20.0) / 10.0, 0.0)
        disposable = 2.0 * total_salary / 3.0 - 5000.0 * elevel + equity / 5.0 - 10000.0
        return np.where(disposable > 0, 0, 1)
    raise ValueError(f"Unsupported Agrawal function id: {function_id}")


def generate_agr_a(spec: AGRASpec = AGR_A_SPEC) -> StreamData:
    rng = np.random.default_rng(spec.seed)
    salary = 20000.0 + 130000.0 * rng.random(spec.n_samples)
    commission = np.where(salary >= 75000.0, 0.0, 10000.0 + 65000.0 * rng.random(spec.n_samples))
    age = 20 + rng.integers(0, 61, size=spec.n_samples)
    elevel = rng.integers(0, 5, size=spec.n_samples)
    car = rng.integers(0, 20, size=spec.n_samples)
    zipcode = rng.integers(0, 9, size=spec.n_samples)
    hvalue = (9.0 - zipcode) * 100000.0 * (0.5 + rng.random(spec.n_samples))
    hyears = 1 + rng.integers(0, 30, size=spec.n_samples)
    loan = rng.random(spec.n_samples) * 500000.0

    concept_path = _build_concept_path(
        spec.n_samples,
        spec.agrawal_functions,
        spec.drift_positions,
        spec.drift_width,
        rng,
    )
    y = np.empty(spec.n_samples, dtype=np.int64)
    for fid in sorted(set(spec.agrawal_functions)):
        mask = concept_path == fid
        if np.any(mask):
            y[mask] = _agrawal_label(
                fid,
                salary[mask],
                commission[mask],
                age[mask],
                elevel[mask],
                car[mask],
                zipcode[mask],
                hvalue[mask],
                hyears[mask],
                loan[mask],
            )

    if spec.perturbation > 0.0:
        def perturb(val: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
            rng_range = max_val - min_val
            out = val + rng_range * (2.0 * (rng.random(val.shape[0]) - 0.5)) * spec.perturbation
            return np.clip(out, min_val, max_val)

        salary = perturb(salary, 20000.0, 150000.0)
        has_comm = commission > 0.0
        commission[has_comm] = perturb(commission[has_comm], 10000.0, 75000.0)
        age = np.rint(perturb(age.astype(np.float64), 20.0, 80.0)).astype(np.int64)
        base_hvalue = (9.0 - zipcode) * 100000.0
        hvalue = np.clip(
            hvalue + base_hvalue * (2.0 * (rng.random(spec.n_samples) - 0.5)) * spec.perturbation,
            0.0,
            1350000.0,
        )
        hyears = np.rint(perturb(hyears.astype(np.float64), 1.0, 30.0)).astype(np.int64)
        loan = perturb(loan, 0.0, 500000.0)

    X = np.column_stack(
        [
            salary,
            commission,
            age.astype(np.float64),
            elevel.astype(np.float64),
            car.astype(np.float64),
            zipcode.astype(np.float64),
            hvalue,
            hyears.astype(np.float64),
            loan,
        ]
    ).astype(np.float32)
    return StreamData(
        X=np.ascontiguousarray(X),
        y=np.ascontiguousarray(y),
        feature_names=AGRAWAL_FEATURE_NAMES,
        target_name="class",
    )


class AGRAStreamDataset(BaseStreamDataset):
    def __init__(self, *, spec: AGRASpec = AGR_A_SPEC, data_root: str | Path) -> None:
        self.name = spec.name
        self.spec = spec
        self.data_root = Path(data_root)

    def load(self) -> StreamData:
        return generate_agr_a(self.spec)
