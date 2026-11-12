from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def labels_from_probability(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probabilities) >= threshold).astype(int)


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = labels_from_probability(probabilities, threshold)
    both_classes = len(np.unique(y_true)) == 2

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    total = int(len(y_true))
    correct = int(matrix[0][0] + matrix[1][1])
    metrics: dict[str, Any] = {
        "task_type": "binary",
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": None,
        "log_loss": None,
        "brier": None,
        "confusion_matrix": matrix.tolist(),
        "correct": correct,
        "incorrect": total - correct,
        "support": {
            "negative": int((y_true == 0).sum()),
            "positive": int((y_true == 1).sum()),
            "total": total,
        },
        "predicted": {
            "negative": int((y_pred == 0).sum()),
            "positive": int((y_pred == 1).sum()),
            "total": int(len(y_pred)),
        },
    }
    if both_classes:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, probabilities)), 4)
        try:
            metrics["log_loss"] = round(float(log_loss(y_true, probabilities, labels=[0, 1])), 4)
        except ValueError:
            metrics["log_loss"] = None
        metrics["brier"] = round(float(brier_score_loss(y_true, probabilities)), 4)
    return metrics


def compute_multiclass_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: list[Any],
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    class_count = len(classes)
    labels = list(range(class_count))
    y_pred = np.argmax(probabilities, axis=1)

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = []
    for index, label in enumerate(classes):
        true_positive = int(matrix[index, index])
        support = int((y_true == index).sum())
        predicted = int((y_pred == index).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "label": str(label),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "support": support,
                "predicted": predicted,
                "correct": true_positive,
            }
        )

    correct = int(sum(matrix[index][index] for index in labels))
    metrics: dict[str, Any] = {
        "task_type": "multiclass",
        "classes": [str(label) for label in classes],
        "class_count": class_count,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(
            float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "recall": round(
            float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            4,
        ),
        "precision_weighted": round(
            float(
                precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
            4,
        ),
        "recall_weighted": round(
            float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            4,
        ),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            4,
        ),
        "roc_auc": None,
        "log_loss": None,
        "brier": None,
        "confusion_matrix": matrix.tolist(),
        "correct": correct,
        "incorrect": int(len(y_true)) - correct,
        "support": {
            "total": int(len(y_true)),
            "per_class": {str(label): int((y_true == index).sum()) for index, label in enumerate(classes)},
        },
        "predicted": {
            "total": int(len(y_pred)),
            "per_class": {str(label): int((y_pred == index).sum()) for index, label in enumerate(classes)},
        },
        "per_class": per_class,
    }

    present = set(np.unique(y_true).tolist())
    if present == set(labels):
        try:
            metrics["roc_auc"] = round(
                float(
                    roc_auc_score(
                        y_true,
                        probabilities,
                        labels=labels,
                        multi_class="ovr",
                        average="macro",
                    )
                ),
                4,
            )
        except ValueError:
            metrics["roc_auc"] = None
    try:
        metrics["log_loss"] = round(
            float(log_loss(y_true, probabilities, labels=labels)),
            4,
        )
    except ValueError:
        metrics["log_loss"] = None
    return metrics


def format_metrics(metrics: dict[str, Any], positive_label: str = "positive") -> str:
    if metrics.get("task_type") == "multiclass":
        return _format_multiclass_metrics(metrics)
    auc = metrics.get("roc_auc")
    lines = [
        f"Threshold: {metrics['threshold']}",
        f"Accuracy:  {metrics['accuracy']}",
        f"Precision: {metrics['precision']}  (positive = {positive_label})",
        f"Recall:    {metrics['recall']}",
        f"F1:        {metrics['f1']}",
        f"ROC AUC:   {auc if auc is not None else 'n/a'}",
        f"Log loss:  {metrics.get('log_loss') if metrics.get('log_loss') is not None else 'n/a'}",
        f"Brier:     {metrics.get('brier') if metrics.get('brier') is not None else 'n/a'}",
        "",
        "Confusion matrix (rows = actual, cols = predicted):",
        "                 pred 0   pred 1",
        f"actual 0         {metrics['confusion_matrix'][0][0]:>6}   {metrics['confusion_matrix'][0][1]:>6}",
        f"actual 1         {metrics['confusion_matrix'][1][0]:>6}   {metrics['confusion_matrix'][1][1]:>6}",
    ]
    return "\n".join(lines)


def _format_multiclass_metrics(metrics: dict[str, Any]) -> str:
    auc = metrics.get("roc_auc")
    lines = [
        f"Classes:   {', '.join(metrics['classes'])}",
        f"Accuracy:  {metrics['accuracy']}",
        f"Precision: {metrics['precision']}  (macro)",
        f"Recall:    {metrics['recall']}  (macro)",
        f"F1:        {metrics['f1']}  (macro)",
        f"ROC AUC:   {auc if auc is not None else 'n/a'}",
        f"Log loss:  {metrics.get('log_loss') if metrics.get('log_loss') is not None else 'n/a'}",
        "",
        "Per-class metrics:",
        f"{'class':<16}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}{'predicted':>10}",
    ]
    for item in metrics.get("per_class", []):
        lines.append(
            f"{item['label']:<16}{item['precision']:>10.4f}{item['recall']:>10.4f}"
            f"{item['f1']:>10.4f}{item['support']:>10}{item['predicted']:>10}"
        )
    lines.extend(["", "Confusion matrix (rows = actual, cols = predicted):"])
    header = "".join(f"{str(label):>10}" for label in metrics["classes"])
    lines.append(f"{'':<16}{header}")
    for label, row in zip(metrics["classes"], metrics["confusion_matrix"], strict=False):
        values = "".join(f"{value:>10}" for value in row)
        lines.append(f"{label:<16}{values}")
    return "\n".join(lines)


def roc_points(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, np.asarray(probabilities, dtype=float))
    return fpr, tpr


def precision_recall_points(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, np.asarray(probabilities, dtype=float))
    return recall, precision


def permutation_importance_frame(
    pipeline,
    frame: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    n_repeats: int = 5,
    top: int = 20,
    scoring: str = "auto",
) -> pd.DataFrame:
    encoded = np.asarray(y).astype(int)
    if scoring == "auto":
        scoring = "roc_auc" if len(np.unique(encoded)) <= 2 else "f1_macro"
    try:
        result = permutation_importance(
            pipeline,
            frame,
            encoded,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])
    names = _transformed_feature_names(pipeline, frame)
    if len(names) != len(result.importances_mean):
        names = [f"feature_{index}" for index in range(len(result.importances_mean))]
    output = pd.DataFrame({"feature": names, "importance": result.importances_mean})
    output = output.reindex(output["importance"].abs().sort_values(ascending=False).index)
    return output.head(top).reset_index(drop=True)


def _transformed_feature_names(pipeline, frame: pd.DataFrame) -> list[str]:
    preprocess = pipeline.named_steps.get("preprocess") if hasattr(pipeline, "named_steps") else None
    if preprocess is None:
        return [str(column) for column in frame.columns]
    try:
        return [str(name) for name in preprocess.get_feature_names_out()]
    except Exception:
        return [str(column) for column in frame.columns]
