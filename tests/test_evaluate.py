import numpy as np

from dc.evaluate import (
    compute_metrics,
    compute_multiclass_metrics,
    format_metrics,
    labels_from_probability,
    roc_points,
)


def test_perfect_metrics():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y, probabilities, threshold=0.5)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["support"]["total"] == 4
    assert metrics["predicted"]["total"] == 4
    assert metrics["correct"] == 4
    assert metrics["incorrect"] == 0


def test_threshold_changes_predictions():
    y = np.array([0, 1, 0, 1])
    probabilities = np.array([0.4, 0.6, 0.4, 0.6])
    low = compute_metrics(y, probabilities, threshold=0.3)
    high = compute_metrics(y, probabilities, threshold=0.7)
    assert low["predicted"]["positive"] == 4
    assert high["predicted"]["positive"] == 0


def test_labels_from_probability():
    result = labels_from_probability(np.array([0.49, 0.5, 0.9]), 0.5)
    assert result.tolist() == [0, 1, 1]


def test_roc_points_require_two_classes():
    assert roc_points(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.4])) is None
    points = roc_points(np.array([0, 1]), np.array([0.2, 0.8]))
    assert points is not None
    assert len(points[0]) == len(points[1])


def test_multiclass_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.2, 0.1, 0.7],
        ]
    )
    metrics = compute_multiclass_metrics(y, probabilities, ["low", "medium", "high"])
    assert metrics["task_type"] == "multiclass"
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert len(metrics["per_class"]) == 3
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert metrics["roc_auc"] == 1.0
    assert metrics["support"]["total"] == 6
    assert metrics["predicted"]["total"] == 6
    assert metrics["correct"] == 6
    assert metrics["incorrect"] == 0
    assert all(item["correct"] == item["support"] for item in metrics["per_class"])
    assert "Classes:" in format_metrics(metrics)
