import numpy as np
import pandas as pd
import pytest

from dc.predict import predict_frame, predict_record
from dc.train import TrainConfig, load_bundle, resolve_positive_label, save_result, train_model


def build_config(**overrides):
    defaults = dict(
        target="approved",
        positive_label=None,
        test_size=0.25,
        cv_folds=3,
        seed=3,
        models=["logistic", "dummy"],
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_resolve_positive_label_for_booleans():
    series = pd.Series([True, False, True])
    negative, positive = resolve_positive_label(series)
    assert negative is False
    assert positive is True


def test_resolve_positive_label_matches_strings():
    series = pd.Series(["A", "B", "A"])
    negative, positive = resolve_positive_label(series, "A")
    assert positive == "A"
    assert negative == "B"


def test_train_model_returns_metadata_and_metrics(binary_frame):
    result = train_model(binary_frame, build_config())
    assert result.model_name in {"logistic", "dummy"}
    assert len(result.leaderboard) == 2
    assert set(result.test_metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert result.metadata["target"] == "approved"
    assert result.metadata["positive_label"] is True
    assert result.metadata["feature_columns"]
    assert len(result.y_test) == len(result.y_proba)


def test_train_model_stores_feature_values(binary_frame):
    result = train_model(binary_frame, build_config())
    values = result.metadata["feature_values"]
    assert sorted(values["region"]) == ["east", "north", "south", "west"]
    assert sorted(values["has_default"]) == [False, True]
    assert "age" not in values
    assert "income" not in values
    assert "approved" not in values


def test_training_is_deterministic(binary_frame):
    first = train_model(binary_frame, build_config())
    second = train_model(binary_frame, build_config())
    first_scores = {row["model"]: row["score"] for row in first.leaderboard}
    second_scores = {row["model"]: row["score"] for row in second.leaderboard}
    assert first_scores == second_scores


def test_save_and_load_bundle(binary_frame, tmp_path):
    result = train_model(binary_frame, build_config())
    path = save_result(result, tmp_path / "model.joblib")
    assert path.exists()
    assert path.with_suffix(".json").exists()
    bundle = load_bundle(path)
    assert bundle["metadata"]["positive_label"] is True
    scored = predict_frame(bundle, binary_frame)
    assert "prediction" in scored.columns
    assert "probability_True" in scored.columns


def test_non_binary_target_raises(binary_frame):
    with pytest.raises(ValueError, match="exactly 2 classes"):
        train_model(binary_frame, build_config(target="income"))


def test_predict_record_threshold_extremes(binary_frame):
    result = train_model(binary_frame, build_config())
    bundle = {"pipeline": result.pipeline, "metadata": result.metadata}
    record = binary_frame.drop(columns=["approved"]).iloc[0].to_dict()
    always_positive = predict_record(bundle, record, threshold=0.0)
    always_negative = predict_record(bundle, record, threshold=1.0)
    assert always_positive["prediction"] is True
    assert always_negative["prediction"] is False


def test_predict_record_accepts_text_inputs(binary_frame):
    result = train_model(binary_frame, build_config())
    bundle = {"pipeline": result.pipeline, "metadata": result.metadata}
    record = {
        "age": "41",
        "income": "6400",
        "debt_ratio": "0.25",
        "employment_years": "9",
        "has_default": "false",
        "region": "north",
    }
    output = predict_record(bundle, record)
    assert isinstance(output["prediction"], bool)
    assert 0.0 <= output["probability"] <= 1.0


def test_predict_requires_feature_columns(binary_frame):
    result = train_model(binary_frame, build_config())
    bundle = {"pipeline": result.pipeline, "metadata": result.metadata}
    with pytest.raises(ValueError, match="missing required column"):
        predict_frame(bundle, binary_frame[["age"]])


def test_balanced_class_weight_option(binary_frame):
    result = train_model(binary_frame, build_config(class_weight="balanced"))
    assert result.metadata["class_weight"] == "balanced"
    assert not np.isnan(result.test_metrics["accuracy"])
