from __future__ import annotations

from .uspds_csv import USPDSCSVStreamDataset


class NOAAStreamDataset(USPDSCSVStreamDataset):
    name = "noaa"

    def __init__(self, data_root: str) -> None:
        super().__init__(name="noaa", csv_filename="NOAA.csv", data_root=data_root)
