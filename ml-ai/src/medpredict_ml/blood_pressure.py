"""Registered M2-v5 pressure target for manual and oscillometric exams."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .data import validate_table


def read_xpt(path: Path, columns: list[str]) -> pd.DataFrame:
    frame, _ = pyreadstat.read_xport(path, encoding="latin1", usecols=columns)
    validate_table(frame, columns, path.stem)
    return frame


def bp_columns(*, auscultatory: bool) -> list[str]:
    prefix, readings = ("BPX", 4) if auscultatory else ("BPXO", 3)
    return [f"{prefix}{kind}{i}" for kind in ("SY", "DI") for i in range(1, readings + 1)]


def measurement_target(frame: pd.DataFrame, *, auscultatory: bool) -> tuple[pd.Series, pd.Series]:
    columns = bp_columns(auscultatory=auscultatory)
    systolic, diastolic = np.split(frame[columns].to_numpy(float), 2, axis=1)
    valid = np.isfinite(systolic) & (systolic > 0) & (systolic <= 300)
    valid &= np.isnan(diastolic) | (np.isfinite(diastolic) & (diastolic > 0) & (systolic > diastolic))
    n = valid.sum(axis=1)
    target = np.divide(np.where(valid, systolic, 0).sum(axis=1), n,
                       out=np.full(len(frame), np.nan), where=n >= 2)
    return pd.Series(target, index=frame.index), pd.Series(n, index=frame.index)
