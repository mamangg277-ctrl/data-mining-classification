import pandas as pd

from dc.eda import (
    class_balance,
    format_report,
    inspect_dataframe,
    remove_duplicate_rows,
    save_report,
)


def test_inspect_reports_shape_and_target(binary_frame):
    info = inspect_dataframe(binary_frame, target="approved")
    assert info["rows"] == len(binary_frame)
    assert info["columns"] == binary_frame.shape[1]
    assert info["target"] == "approved"
    assert info["class_balance"]["total"] == len(binary_frame)
    assert "approved" in info["binary_target_candidates"]


def test_class_balance_flags_imbalance():
    balance = class_balance(pd.Series([0] * 90 + [1] * 10))
    assert balance["imbalanced"] is True
    assert balance["classes"][0]["pct"] == 90.0


def test_remove_duplicate_rows_keeps_first_occurrence():
    frame = pd.DataFrame({"a": [1, 1, 2, 3, 3], "b": ["x", "x", "y", "z", "z"]})
    cleaned, removed = remove_duplicate_rows(frame)
    assert removed == 2
    assert len(cleaned) == 3
    assert list(cleaned.index) == [0, 1, 2]
    assert cleaned["a"].tolist() == [1, 2, 3]


def test_report_rendering_and_saving(binary_frame, tmp_path):
    info = inspect_dataframe(binary_frame, target="approved")
    text = format_report(info)
    assert "# Data inspection report" in text
    assert "Target: approved" in text
    markdown_path, json_path = save_report(info, tmp_path / "report")
    assert markdown_path.exists()
    assert json_path.exists()
    assert markdown_path.read_text(encoding="utf-8") == text
