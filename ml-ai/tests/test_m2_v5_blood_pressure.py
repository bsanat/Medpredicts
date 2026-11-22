import numpy as np
import pandas as pd
import pyreadstat
import pytest

from medpredict_ml.blood_pressure import bp_columns, measurement_target, read_xpt


def test_bpx_xpt_roundtrip_and_registered_auscultatory_target(tmp_path):
    columns = bp_columns(auscultatory=True)
    assert columns == ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]
    na = np.nan
    fixture = pd.DataFrame([
        [120, na, na, 130, 78, na, na, 80],
        [118, 126, 140, na, na, 126, 150, na],
        [0, 302, 180, 300, na, 76, 0, 78],
        [300, 140, 130, 120, 85, -1, na, 80],
    ], columns=columns)
    fixture.insert(0, "SEQN", [1, 2, 3, 4])
    path = tmp_path / "synthetic_bpx.xpt"
    pyreadstat.write_xport(fixture, path, file_format_version=5)
    restored = read_xpt(path, ["SEQN", *columns])
    target, count = measurement_target(restored, auscultatory=True)
    np.testing.assert_array_equal(count, [2, 1, 1, 3])
    np.testing.assert_allclose(target, [125, na, na, 550 / 3], equal_nan=True)


def test_xpt_requires_the_fourth_auscultatory_column(tmp_path):
    columns = ["SEQN", *bp_columns(auscultatory=True)]
    path = tmp_path / "synthetic_missing.xpt"
    pyreadstat.write_xport(pd.DataFrame([[1.] * (len(columns) - 1)], columns=columns[:-1]), path,
                          file_format_version=5)
    with pytest.raises(ValueError, match="BPXDI4"):
        read_xpt(path, columns)


def test_oscillometric_target_ignores_a_fourth_reading():
    frame = pd.DataFrame({"BPXOSY1": [120.], "BPXOSY2": [122.], "BPXOSY3": [np.nan], "BPXOSY4": [300.],
                          "BPXODI1": [80.], "BPXODI2": [82.], "BPXODI3": [np.nan], "BPXODI4": [80.]})
    target, count = measurement_target(frame, auscultatory=False)
    assert count.tolist() == [2]
    assert target.tolist() == [121.]


def test_zero_diastolic_sensitivity_changes_only_zero_readings():
    frame = pd.DataFrame([[160, 140, 302, 150, 0, 82, 0, -1], [0, 140, 120, 130, 0, 80, 120, np.nan]],
                         columns=bp_columns(auscultatory=True))
    strict, strict_counts = measurement_target(frame, auscultatory=True)
    relaxed, relaxed_counts = measurement_target(frame, auscultatory=True, allow_zero_diastolic=True)
    np.testing.assert_array_equal(strict_counts, [1, 2])
    np.testing.assert_array_equal(relaxed_counts, [2, 2])
    np.testing.assert_allclose(strict, [np.nan, 135], equal_nan=True)
    np.testing.assert_allclose(relaxed, [150, 135])
