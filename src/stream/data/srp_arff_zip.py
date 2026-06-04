from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from .base_stream_dataset import BaseStreamDataset, StreamData


def _strip_quotes(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


class SRPARFFZipStreamDataset(BaseStreamDataset):
    """Dense ARFF dataset loader for bundled SRP-style zip archives."""

    def __init__(
        self,
        *,
        name: str,
        zip_filename: str,
        data_root: str | Path,
    ) -> None:
        self.name = name
        self.data_root = Path(data_root)
        self.cache_dir = self.data_root / "srp_datasets"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.zip_filename = zip_filename
        self.zip_path = self.cache_dir / zip_filename

    def _ensure_local_zip(self) -> Path:
        if self.zip_path.exists():
            return self.zip_path
        raise FileNotFoundError(f"Bundled stream archive not found: {self.zip_path}")

    def _parse_attribute_line(self, line: str) -> tuple[str, str] | None:
        low = line.lower()
        if not low.startswith("@attribute"):
            return None
        rest = line[len("@attribute") :].strip()
        if not rest:
            return None
        if rest[0] in {"'", '"'}:
            quote = rest[0]
            end = rest.find(quote, 1)
            if end < 0:
                raise ValueError(f"Malformed ARFF attribute line: {line}")
            name = rest[1:end]
            atype = rest[end + 1 :].strip()
        else:
            parts = rest.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed ARFF attribute line: {line}")
            name, atype = parts
        return _strip_quotes(name), atype.strip()

    def load(self) -> StreamData:
        zip_path = self._ensure_local_zip()
        with zipfile.ZipFile(zip_path) as zf:
            member = next((n for n in zf.namelist() if not n.endswith("/")), None)
            if member is None:
                raise FileNotFoundError(f"No file found inside zip archive: {zip_path}")
            with zf.open(member) as raw_f:
                text_f = io.TextIOWrapper(raw_f, encoding="utf-8", errors="ignore", newline="")
                attr_names: list[str] = []
                is_numeric: list[bool] = []
                nominal_maps: list[dict[str, int]] = []
                in_data = False
                x_rows: list[list[float]] = []
                y_raw: list[str] = []

                for raw_line in text_f:
                    line = raw_line.strip()
                    if not line or line.startswith("%"):
                        continue
                    low = line.lower()
                    if not in_data:
                        parsed = self._parse_attribute_line(line)
                        if parsed is not None:
                            name, atype = parsed
                            attr_names.append(name)
                            is_num = atype.lower() in {"numeric", "real", "integer"}
                            is_numeric.append(is_num)
                            nominal_maps.append({})
                            continue
                        if low.startswith("@data"):
                            in_data = True
                        continue

                    row_tokens = next(csv.reader([line], skipinitialspace=True))
                    if len(row_tokens) != len(attr_names):
                        raise ValueError(
                            f"Row has {len(row_tokens)} fields but expected {len(attr_names)} in {member}"
                        )
                    features: list[float] = []
                    for idx, token in enumerate(row_tokens[:-1]):
                        token = _strip_quotes(token)
                        if is_numeric[idx]:
                            if token == "?":
                                features.append(float("nan"))
                            else:
                                features.append(float(token))
                        else:
                            if token == "?":
                                features.append(-1.0)
                            else:
                                mapping = nominal_maps[idx]
                                if token not in mapping:
                                    mapping[token] = len(mapping)
                                features.append(float(mapping[token]))
                    x_rows.append(features)
                    y_raw.append(_strip_quotes(row_tokens[-1]))

        if not x_rows:
            raise ValueError(f"No data rows found in {zip_path}")

        X = np.asarray(x_rows, dtype=np.float32)
        y_map: dict[str, int] = {}
        y = np.empty(len(y_raw), dtype=np.int64)
        for i, label in enumerate(y_raw):
            if label not in y_map:
                y_map[label] = len(y_map)
            y[i] = y_map[label]

        return StreamData(
            X=np.ascontiguousarray(X),
            y=np.ascontiguousarray(y),
            feature_names=attr_names[:-1],
            target_name=attr_names[-1],
        )
