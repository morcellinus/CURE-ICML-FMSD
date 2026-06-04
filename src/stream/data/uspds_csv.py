from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .base_stream_dataset import BaseStreamDataset, StreamData


class USPDSCSVStreamDataset(BaseStreamDataset):
    """Generic headerless USP/UPS CSV stream dataset.

    Assumes:
    - no header row
    - last column is the target label
    - original row order is the stream order
    """

    def __init__(
        self,
        *,
        name: str,
        csv_filename: str,
        data_root: str | Path,
        drop_encoded_labels: tuple[int, ...] = (),
    ) -> None:
        self.name = name
        self.data_root = Path(data_root)
        self.csv_path = self.data_root / csv_filename
        self.drop_encoded_labels = tuple(int(v) for v in drop_encoded_labels)

    def load(self) -> StreamData:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"{self.name} stream file not found: {self.csv_path}")

        rows: list[list[str]] = []
        field_count_hist: dict[int, int] = {}
        with open(self.csv_path, "r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                fields = [x.strip() for x in line.split(",")]
                rows.append(fields)
                n_fields = len(fields)
                field_count_hist[n_fields] = field_count_hist.get(n_fields, 0) + 1

        if not rows:
            raise ValueError(f"No data rows found in {self.csv_path}")

        expected_fields = max(field_count_hist.items(), key=lambda kv: (kv[1], kv[0]))[0]
        valid_rows = [r for r in rows if len(r) == expected_fields]
        dropped_rows = len(rows) - len(valid_rows)
        if expected_fields < 2:
            raise ValueError(f"Expected >=2 columns in {self.csv_path}, got {expected_fields}")
        if not valid_rows:
            raise ValueError(f"No valid rows with {expected_fields} fields found in {self.csv_path}")

        data = np.asarray(valid_rows, dtype=object)
        X = data[:, :-1].astype(np.float32, copy=False)
        y_raw = data[:, -1].astype(str, copy=False)

        le = LabelEncoder()
        y = le.fit_transform(y_raw).astype(np.int64, copy=False)
        if self.drop_encoded_labels:
            drop_set = set(self.drop_encoded_labels)
            keep_mask = np.asarray([int(label) not in drop_set for label in y], dtype=bool)
            X = X[keep_mask]
            y = y[keep_mask]
            if y.size == 0:
                raise ValueError(
                    f"All rows were dropped for {self.name} after removing encoded labels {sorted(drop_set)}."
                )
            kept_labels = sorted(int(v) for v in np.unique(y).tolist())
            remap = {old: new for new, old in enumerate(kept_labels)}
            y = np.asarray([remap[int(v)] for v in y], dtype=np.int64)

        feature_names = [f"f{i}" for i in range(X.shape[1])]
        if dropped_rows > 0:
            import sys

            print(
                f"[dataset:{self.name}] dropped {dropped_rows} malformed rows from {self.csv_path.name} "
                f"(expected_fields={expected_fields})",
                file=sys.stderr,
                flush=True,
            )
        return StreamData(
            X=np.ascontiguousarray(X),
            y=np.ascontiguousarray(y),
            feature_names=feature_names,
            target_name="label",
        )
