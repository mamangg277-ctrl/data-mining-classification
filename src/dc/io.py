from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes


class DataError(ValueError):
    """Base error for data input and output problems."""


class DataLoadError(DataError):
    """Raised when a data file cannot be loaded."""


class DataSaveError(DataError):
    """Raised when a data frame cannot be saved."""


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json"}


def _read_json(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataLoadError(f"Could not read JSON file '{path}': {exc}") from exc

    if isinstance(raw, list):
        return pd.json_normalize(raw)
    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            return pd.json_normalize(raw["data"])
        if raw and all(isinstance(value, list) for value in raw.values()):
            return pd.DataFrame(raw)
        return pd.json_normalize([raw])
    raise DataLoadError("JSON must contain an object, a list of objects, or column lists.")


def load_dataframe(path: str | Path, sheet: int | str = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            try:
                frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
            except Exception:
                try:
                    frame = pd.read_csv(path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    frame = pd.read_csv(path, encoding="latin-1")
                except Exception:
                    frame = pd.read_csv(path, sep=";", encoding="latin-1")
        elif suffix == ".xlsx":
            frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        elif suffix == ".json":
            frame = _read_json(path)
        else:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DataLoadError(f"Unsupported file type '{suffix}'. Supported: {supported}")
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(f"Could not load '{path}': {exc}") from exc

    if frame.empty:
        raise DataLoadError(f"'{path}' contains no rows.")
    if not list(frame.columns):
        raise DataLoadError(f"'{path}' contains no columns.")
    return frame


def save_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            frame.to_csv(path, index=False)
        elif suffix == ".xlsx":
            frame.to_excel(path, index=False, engine="openpyxl")
        elif suffix == ".json":
            path.write_text(
                frame.to_json(orient="records", indent=2, force_ascii=False),
                encoding="utf-8",
            )
        else:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DataSaveError(f"Unsupported file type '{suffix}'. Supported: {supported}")
    except DataSaveError:
        raise
    except Exception as exc:
        raise DataSaveError(f"Could not save '{path}': {exc}") from exc
    return path


def column_kind(series: pd.Series, row_count: int) -> str:
    non_null = series.dropna()
    unique = non_null.nunique()
    if row_count > 1 and unique == row_count and unique > 2:
        return "id"
    if ptypes.is_bool_dtype(series):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(series):
        return "datetime"
    if ptypes.is_numeric_dtype(series):
        return "numeric"
    if unique <= 2:
        return "boolean"
    if unique <= max(20, int(row_count * 0.05)):
        return "categorical"
    return "text"


def infer_schema(frame: pd.DataFrame, target: str | None = None) -> list[dict[str, Any]]:
    row_count = len(frame)
    schema: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        unique = series.dropna().nunique()
        schema.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "kind": column_kind(series, row_count),
                "missing": int(series.isna().sum()),
                "missing_pct": round(float(series.isna().mean()) * 100, 2),
                "unique": int(unique),
                "is_target": name == target,
            }
        )
    return schema


def target_class_count(series: pd.Series) -> int:
    """Count distinct target values, ignoring surrounding whitespace for text columns."""
    non_null = series.dropna()
    if non_null.empty:
        return 0
    if non_null.dtype == object or ptypes.is_string_dtype(non_null):
        return int(non_null.astype(str).str.strip().nunique())
    return int(non_null.nunique())


def find_classification_targets(
    frame: pd.DataFrame,
    max_classes: int = 20,
) -> list[str]:
    """Return columns that look like classification targets with 2..max_classes values."""
    row_count = len(frame)
    targets: list[str] = []
    for name in frame.columns:
        non_null = frame[name].dropna()
        if non_null.empty:
            continue
        unique = target_class_count(non_null)
        if unique < 2 or unique > max_classes:
            continue
        if row_count > 1 and unique == row_count:
            continue
        targets.append(str(name))
    return targets


def find_binary_targets(frame: pd.DataFrame) -> list[str]:
    return find_classification_targets(frame, max_classes=2)


def unique_values(frame: pd.DataFrame, column: str, limit: int = 50) -> list[Any]:
    if column not in frame.columns:
        raise KeyError(f"Column '{column}' not found.")
    values = frame[column].dropna().unique().tolist()
    values = [value.item() if isinstance(value, np.generic) else value for value in values]
    try:
        values = sorted(values)
    except TypeError:
        pass
    return values[:limit]


def feature_value_options(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    limit: int = 50,
) -> dict[str, list[Any]]:
    """Collect the distinct values seen in low-cardinality (categorical/boolean) columns."""
    row_count = len(frame)
    names = [str(name) for name in (frame.columns if columns is None else columns)]
    options: dict[str, list[Any]] = {}
    for name in names:
        if name not in frame.columns:
            continue
        if column_kind(frame[name], row_count) not in ("categorical", "boolean"):
            continue
        values = unique_values(frame, name, limit)
        if values:
            options[name] = values
    return options


def get_feature_columns(
    frame: pd.DataFrame,
    target: str,
    exclude: list[str] | None = None,
) -> list[str]:
    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' not found.")
    exclude_set = set(exclude or [])
    columns = [str(col) for col in frame.columns if col != target and col not in exclude_set]
    if not columns:
        raise ValueError("No feature columns available after excluding the target.")
    return columns
