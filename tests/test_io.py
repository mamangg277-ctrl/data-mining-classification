import pandas as pd
import pytest

from dc.io import (
    DataLoadError,
    DataSaveError,
    find_binary_targets,
    find_classification_targets,
    get_feature_columns,
    infer_schema,
    load_dataframe,
    save_dataframe,
    unique_values,
)


@pytest.mark.parametrize("extension", [".csv", ".xlsx", ".json"])
def test_roundtrip_supported_formats(tmp_path, binary_frame, extension):
    path = tmp_path / f"data{extension}"
    save_dataframe(binary_frame, path)
    loaded = load_dataframe(path)
    assert loaded.shape == binary_frame.shape
    assert list(loaded.columns) == list(binary_frame.columns)
    assert loaded["income"].iloc[0] == pytest.approx(binary_frame["income"].iloc[0])


def test_invalid_extension_raises(tmp_path, binary_frame):
    with pytest.raises(DataSaveError):
        save_dataframe(binary_frame, tmp_path / "data.txt")


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataLoadError):
        load_dataframe(tmp_path / "nope.csv")


def test_infer_schema_marks_target(binary_frame):
    schema = infer_schema(binary_frame, target="approved")
    target = next(item for item in schema if item["name"] == "approved")
    assert target["is_target"] is True
    assert "approved" in find_binary_targets(binary_frame)
    assert "region" in find_classification_targets(binary_frame)
    assert "region" not in find_binary_targets(binary_frame)


def test_unique_values_sorted(binary_frame):
    values = unique_values(binary_frame, "region")
    assert values == sorted(values)


def test_feature_columns_respect_excludes(binary_frame):
    columns = get_feature_columns(binary_frame, "approved", ["region", "age"])
    assert "approved" not in columns
    assert "region" not in columns
    assert "age" not in columns


def test_get_feature_columns_missing_target(binary_frame):
    with pytest.raises(KeyError):
        get_feature_columns(binary_frame, "nope")


def test_json_object_input(tmp_path):
    path = tmp_path / "object.json"
    path.write_text('{"a": 1, "b": "x"}', encoding="utf-8")
    loaded = load_dataframe(path)
    assert loaded.shape == (1, 2)


def test_column_dtypes_preserved_for_csv(tmp_path, binary_frame):
    path = tmp_path / "data.csv"
    save_dataframe(binary_frame, path)
    loaded = load_dataframe(path)
    assert pd.api.types.is_bool_dtype(loaded["approved"])
