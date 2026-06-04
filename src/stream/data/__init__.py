from __future__ import annotations

from .base_stream_dataset import BaseStreamDataset, StreamData
from .moa_synthetic import AGRAStreamDataset
from .noaa import NOAAStreamDataset
from .srp_arff_zip import SRPARFFZipStreamDataset
from .uspds_csv import USPDSCSVStreamDataset


def build_dataset(name: str, data_root: str):
    key = name.lower()
    if key in {"agr_a", "agr_a_30k", "agra"}:
        return AGRAStreamDataset(data_root=data_root)
    if key == "noaa":
        return NOAAStreamDataset(data_root)
    if key in {"meter", "smartmeter", "smart_meter", "ladpu"}:
        return USPDSCSVStreamDataset(name="meter", csv_filename="METER.csv", data_root=data_root)
    if key in {"rialto", "rialto_bridge_timelapse", "rialto_bridge"}:
        return USPDSCSVStreamDataset(name="rialto", csv_filename="RIALTO.csv", data_root=data_root)
    if key in {"posture_no8", "posture_drop8", "posture_10cls"}:
        return USPDSCSVStreamDataset(name="posture_no8", csv_filename="POSTURE.csv", data_root=data_root, drop_encoded_labels=(8,))
    if key in {"poker", "pokerhand", "poker_hand"}:
        return USPDSCSVStreamDataset(name="poker", csv_filename="POKER.csv", data_root=data_root)
    if key == "nomao":
        return SRPARFFZipStreamDataset(name="nomao", zip_filename="nomao.arff.zip", data_root=data_root)
    raise ValueError(f"Unknown dataset for this bundle: {name}")


__all__ = [
    "BaseStreamDataset",
    "StreamData",
    "AGRAStreamDataset",
    "NOAAStreamDataset",
    "USPDSCSVStreamDataset",
    "SRPARFFZipStreamDataset",
    "build_dataset",
]
