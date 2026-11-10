from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from pandas.api import types as ptypes
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

from dc.evaluate import compute_metrics
from dc.io import feature_value_options, get_feature_columns
from dc.models import MODEL_LABELS, build_candidate_models, search_spaces
from dc.preprocess import build_pipeline

BUNDLE_VERSION = 1
ProgressCallback = Callable[[str, float], None]


@dataclass
class TrainConfig:
    target: str
    positive_label: Any | None = None
    exclude: list[str] = field(default_factory=list)
    test_size: float = 0.2
    cv_folds: int = 5
    seed: int = 42
    models: list[str] = field(default_factory=list)
    class_weight: str | None = None
    search: bool = False
    n_iter: int = 15
    scoring: str = "f1_macro"


@dataclass
class TrainResult:
    pipeline: Any
    model_name: str
    metadata: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    test_metrics: dict[str, Any]
    y_test: np.ndarray
    y_proba: np.ndarray
    X_test: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def resolve_positive_label(series: pd.Series, positive_label: Any | None = None) -> tuple[Any, Any]:
    classes = [_native_label(value) for value in pd.unique(series.dropna())]
    if len(classes) != 2:
        raise ValueError(
            f"Target must have exactly 2 classes for binary classification; found {len(classes)}."
        )
    if positive_label is None:
        if any(isinstance(value, (bool, np.bool_)) for value in classes):
            positive = next(value for value in classes if bool(value) is True)
        else:
            try:
                positive = sorted(classes)[-1]
            except TypeError:
                positive = classes[-1]
    else:
        positive = _match_label(classes, positive_label)
    negative = next(value for value in classes if value != positive)
    return negative, positive


def _native_label(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _match_label(classes: list[Any], requested: Any) -> Any:
    for value in classes:
        if value == requested:
            return value
    for value in classes:
        if str(value) == str(requested):
            return value
    options = ", ".join(str(value) for value in classes)
    raise ValueError(f"Positive label '{requested}' is not one of the target classes: {options}")


def train_model(
    frame: pd.DataFrame,
    config: TrainConfig,
    progress: ProgressCallback | None = None,
) -> TrainResult:
    if config.target not in frame.columns:
        raise ValueError(f"Target column '{config.target}' not found.")
    warnings_list: list[str] = []

    clean = frame.dropna(subset=[config.target]).copy()
    if clean[config.target].dtype == object or ptypes.is_string_dtype(clean[config.target]):
        stripped = clean[config.target].astype(str).str.strip()
        if stripped.nunique() == 2:
            clean[config.target] = stripped
        elif stripped.str.upper().nunique() == 2:
            clean[config.target] = stripped.str.upper()

    dropped = len(frame) - len(clean)
    if dropped:
        warnings_list.append(f"Dropped {dropped} row(s) with a missing target value.")
    if len(clean) < 10:
        raise ValueError("At least 10 labeled rows are required to train a model.")

    negative, positive = resolve_positive_label(clean[config.target], config.positive_label)
    y = (clean[config.target] == positive).astype(int).to_numpy()
    features = get_feature_columns(clean, config.target, config.exclude)
    X = clean[features]

    minority = int(min((y == 0).sum(), (y == 1).sum()))
    if minority < 4:
        raise ValueError("Each class needs at least 4 labeled rows to train and evaluate.")
    if minority / len(y) < 0.2:
        warnings_list.append("Target is imbalanced; consider enabling balanced class weights.")

    folds = int(min(config.cv_folds, minority))
    if folds < 2:
        raise ValueError("Not enough samples per class to run cross-validation.")
    if folds < config.cv_folds:
        warnings_list.append(f"Reduced cross-validation from {config.cv_folds} to {folds} folds.")

    test_rows = int(np.ceil(len(clean) * config.test_size))
    if test_rows >= len(clean) or test_rows < 2:
        raise ValueError("test_size must leave both training and test rows in every class.")

    test_size = float(test_rows) / len(clean)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.seed
    )
    train_minority = int(min((y_train == 0).sum(), (y_train == 1).sum()))
    train_folds = int(min(folds, train_minority))
    if train_folds < 2:
        raise ValueError("Not enough training samples per class for cross-validation.")

    splitter = StratifiedKFold(n_splits=train_folds, shuffle=True, random_state=config.seed)
    candidates = build_candidate_models(config.seed, config.class_weight)
    requested = config.models or list(candidates)
    unknown = [name for name in requested if name not in candidates]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(candidates)}"
        )

    spaces = search_spaces()
    leaderboard: list[dict[str, Any]] = []
    best_name: str | None = None
    best_score = -np.inf
    best_estimator = None

    for index, name in enumerate(requested):
        if progress:
            progress(f"Training {MODEL_LABELS.get(name, name)}...", index / len(requested))
        try:
            pipeline = build_pipeline(candidates[name])
            if config.search and spaces.get(name):
                searcher = RandomizedSearchCV(
                    pipeline,
                    spaces[name],
                    n_iter=min(config.n_iter, _space_size(spaces[name])),
                    scoring=config.scoring,
                    cv=splitter,
                    random_state=config.seed,
                    error_score="raise",
                )
                searcher.fit(X_train, y_train)
                score = float(searcher.best_score_)
                std = float(searcher.cv_results_["std_test_score"][searcher.best_index_])
                fitted = searcher.best_estimator_
            else:
                scores = cross_val_score(
                    pipeline,
                    X_train,
                    y_train,
                    cv=splitter,
                    scoring=config.scoring,
                    error_score="raise",
                )
                score = float(np.mean(scores))
                std = float(np.std(scores))
                fitted = pipeline.fit(X_train, y_train)
        except Exception as exc:
            leaderboard.append(
                {
                    "model": name,
                    "label": MODEL_LABELS.get(name, name),
                    "score": None,
                    "std": None,
                    "error": str(exc),
                }
            )
            continue

        leaderboard.append(
            {
                "model": name,
                "label": MODEL_LABELS.get(name, name),
                "score": round(score, 4),
                "std": round(std, 4),
                "error": None,
            }
        )
        if score > best_score:
            best_score = score
            best_name = name
            best_estimator = fitted

    if best_estimator is None or best_name is None:
        errors = "; ".join(
            f"{row['label']}: {row['error']}" for row in leaderboard if row["error"]
        )
        raise ValueError(f"All candidate models failed. {errors}")

    if progress:
        progress("Evaluating on the held-out test set...", 0.95)

    probabilities = best_estimator.predict_proba(X_test)
    classes = list(best_estimator.classes_)
    positive_index = classes.index(1)
    y_proba = probabilities[:, positive_index]
    test_metrics = compute_metrics(y_test, y_proba, threshold=0.5)

    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": config.target,
        "positive_label": _jsonable(positive),
        "negative_label": _jsonable(negative),
        "feature_columns": features,
        "feature_dtypes": {str(column): str(clean[column].dtype) for column in features},
        "feature_values": feature_value_options(clean, features),
        "model_name": best_name,
        "model_label": MODEL_LABELS.get(best_name, best_name),
        "scoring": config.scoring,
        "cv_folds": train_folds,
        "class_weight": config.class_weight,
        "threshold": 0.5,
        "seed": config.seed,
        "rows": int(len(clean)),
        "test_size": round(test_size, 4),
        "test_metrics": test_metrics,
        "leaderboard": leaderboard,
        "warnings": warnings_list,
        "versions": {
            "python": _python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }
    leaderboard.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0)))

    if progress:
        progress("Done.", 1.0)

    return TrainResult(
        pipeline=best_estimator,
        model_name=best_name,
        metadata=metadata,
        leaderboard=leaderboard,
        test_metrics=test_metrics,
        y_test=y_test,
        y_proba=y_proba,
        X_test=X_test,
        warnings=warnings_list,
    )


def _space_size(space: dict[str, list]) -> int:
    size = 1
    for values in space.values():
        size *= max(1, len(values))
    return size


def _python_version() -> str:
    import platform

    return platform.python_version()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def save_bundle(pipeline, metadata: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bundle_version": BUNDLE_VERSION, "pipeline": pipeline, "metadata": metadata}
    joblib.dump(payload, path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def save_result(result: TrainResult, path: str | Path) -> Path:
    return save_bundle(result.pipeline, result.metadata, path)


def load_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    if isinstance(payload, dict) and "pipeline" in payload:
        metadata = payload.get("metadata", {})
        return {"pipeline": payload["pipeline"], "metadata": metadata}
    metadata = {}
    metadata_path = path.with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"pipeline": payload, "metadata": metadata}
