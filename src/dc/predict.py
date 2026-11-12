from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from dc.io import load_dataframe, save_dataframe
from dc.train import load_bundle


def _native(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _task_type(metadata: dict[str, Any]) -> str:
    if metadata.get("task_type") in {"binary", "multiclass"}:
        return str(metadata["task_type"])
    return "binary" if metadata.get("positive_label") is not None else "multiclass"


def _class_labels(bundle: dict[str, Any]) -> list[Any]:
    metadata = bundle.get("metadata", {})
    classes = metadata.get("classes")
    if classes:
        return [_native(value) for value in classes]
    if _task_type(metadata) == "binary":
        return [
            _native(metadata.get("negative_label", 0)),
            _native(metadata.get("positive_label", 1)),
        ]
    return [_native(value) for value in bundle["pipeline"].classes_]


def _probability_matrix(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(bundle["pipeline"].predict_proba(frame), dtype=float)
    metadata = bundle.get("metadata", {})
    classes = metadata.get("classes")
    pipeline_classes = list(bundle["pipeline"].classes_)
    if classes:
        order = [
            pipeline_classes.index(index)
            for index in range(len(classes))
            if index in pipeline_classes
        ]
        if len(order) == len(classes):
            probabilities = probabilities[:, order]
    return probabilities


def _coerce_frame(bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    dtypes = bundle.get("metadata", {}).get("feature_dtypes", {})
    result = frame.copy()
    for column, dtype in dtypes.items():
        if column not in result.columns:
            continue
        try:
            if dtype in ("bool", "boolean"):
                result[column] = result[column].map(_coerce_bool).astype("boolean")
            elif dtype.startswith("datetime"):
                result[column] = pd.to_datetime(result[column], errors="coerce")
            elif _is_numeric_dtype(dtype):
                result[column] = pd.to_numeric(result[column], errors="coerce")
            else:
                values = result[column].astype("object")
                result[column] = values.where(values.map(lambda v: not _blank(v)), np.nan)
        except (TypeError, ValueError):
            continue
    return result


def _is_numeric_dtype(dtype: str) -> bool:
    try:
        return ptypes.is_numeric_dtype(np.dtype(dtype))
    except TypeError:
        return False


def _blank(value: Any) -> bool:
    return isinstance(value, str) and value.strip() == ""


def _coerce_bool(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NA
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "positive"}:
        return True
    if text in {"false", "0", "no", "n", "negative"}:
        return False
    return pd.NA


def required_features(bundle: dict[str, Any]) -> list[str]:
    features = bundle.get("metadata", {}).get("feature_columns")
    if not features:
        raise ValueError("Model metadata is missing the list of feature columns.")
    return list(features)


def predict_proba_matrix(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    features = required_features(bundle)
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise ValueError(f"Input data is missing required column(s): {', '.join(missing)}")
    features_frame = _coerce_frame(bundle, frame[features])
    return _probability_matrix(bundle, features_frame)


def predict_probabilities(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    metadata = bundle.get("metadata", {})
    if _task_type(metadata) == "multiclass":
        raise ValueError("predict_probabilities() is only available for binary models.")
    probabilities = predict_proba_matrix(bundle, frame)
    classes = _class_labels(bundle)
    positive = _native(metadata.get("positive_label", classes[1]))
    for index, label in enumerate(classes):
        if label == positive or str(label) == str(positive):
            return probabilities[:, index]
    return probabilities[:, 1]


def predict_frame(
    bundle: dict[str, Any],
    frame: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    metadata = bundle.get("metadata", {})
    output = frame.copy()
    if _task_type(metadata) == "multiclass":
        probabilities = predict_proba_matrix(bundle, frame)
        classes = _class_labels(bundle)
        indices = np.argmax(probabilities, axis=1)
        output["prediction"] = [_native(classes[int(index)]) for index in indices]
        for index, label in enumerate(classes):
            output[f"probability_{label}"] = np.round(probabilities[:, index], 6)
        return output

    chosen_threshold = float(metadata.get("threshold", 0.5) if threshold is None else threshold)
    if not 0.0 <= chosen_threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probabilities = predict_probabilities(bundle, frame)
    positive_label = _native(metadata.get("positive_label", 1))
    negative_label = _native(metadata.get("negative_label", 0))
    predictions = np.where(probabilities >= chosen_threshold, positive_label, negative_label)

    output["prediction"] = predictions
    output[f"probability_{positive_label}"] = np.round(probabilities, 6)
    output["threshold"] = chosen_threshold
    return output


def predict_record(
    bundle: dict[str, Any],
    record: dict[str, Any],
    threshold: float | None = None,
) -> dict[str, Any]:
    frame = pd.DataFrame([record])
    result = predict_frame(bundle, frame, threshold=threshold)
    row = result.iloc[0]
    metadata = bundle.get("metadata", {})
    prediction = _native(row["prediction"])
    if _task_type(metadata) == "multiclass":
        classes = _class_labels(bundle)
        probabilities = {
            str(label): float(row[f"probability_{label}"]) for label in classes
        }
        return {
            "task_type": "multiclass",
            "prediction": prediction,
            "probability": float(max(probabilities.values())),
            "probabilities": probabilities,
            "classes": [str(label) for label in classes],
        }

    probability_column = next(
        column for column in result.columns if str(column).startswith("probability_")
    )
    return {
        "task_type": "binary",
        "prediction": prediction,
        "probability": float(row[probability_column]),
        "threshold": float(row["threshold"]),
        "positive_label": _native(metadata.get("positive_label", 1)),
        "negative_label": _native(metadata.get("negative_label", 0)),
    }


def predict_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, Path]:
    bundle = load_bundle(model_path)
    frame = load_dataframe(input_path)
    result = predict_frame(bundle, frame, threshold=threshold)
    saved = save_dataframe(result, output_path)
    return result, saved
