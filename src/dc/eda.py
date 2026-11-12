from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api import types as ptypes

from dc.io import find_binary_targets, find_classification_targets, infer_schema


def class_balance(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().value_counts(dropna=False)
    total = int(counts.sum())
    items = [
        {
            "label": str(label),
            "count": int(count),
            "pct": round(float(count) / total * 100, 2) if total else 0.0,
        }
        for label, count in counts.items()
    ]
    return {"total": total, "classes": items, "imbalanced": bool(items and min(i["count"] for i in items) / total < 0.2)}


def inspect_dataframe(frame: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    if target is not None and target not in frame.columns:
        raise KeyError(f"Target column '{target}' not found.")
    info: dict[str, Any] = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "total_missing": int(frame.isna().sum().sum()),
        "binary_target_candidates": find_binary_targets(frame),
        "classification_target_candidates": find_classification_targets(frame),
        "schema": infer_schema(frame, target),
    }
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty:
        describe = numeric.describe().round(4)
        info["numeric_summary"] = {
            str(col): {str(stat): _to_python(value) for stat, value in describe[col].items()}
            for col in describe.columns
        }
    if target is not None:
        info["target"] = target
        info["class_balance"] = class_balance(frame[target])
    return info


def _to_python(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def format_report(info: dict[str, Any]) -> str:
    lines = [
        "# Data inspection report",
        "",
        f"- Rows: {info['rows']}",
        f"- Columns: {info['columns']}",
        f"- Duplicate rows: {info['duplicate_rows']}",
        f"- Missing cells: {info['total_missing']}",
    ]
    candidates = info.get("classification_target_candidates", [])
    if candidates:
        lines.append(f"- Classification target candidates: {', '.join(candidates)}")
    binary_candidates = info.get("binary_target_candidates", [])
    if binary_candidates:
        lines.append(f"- Binary target candidates: {', '.join(binary_candidates)}")
    if "target" in info:
        balance = info["class_balance"]
        lines.extend(["", f"## Target: {info['target']}", ""])
        for item in balance["classes"]:
            lines.append(f"- {item['label']}: {item['count']} ({item['pct']}%)")
        if balance["imbalanced"]:
            lines.append("")
            lines.append("> Warning: the target looks imbalanced (minority below 20%).")
    lines.extend(["", "## Columns", "", "| Column | Type | Kind | Missing | Unique |", "|---|---|---|---|---|"])
    for column in info["schema"]:
        marker = " (target)" if column["is_target"] else ""
        lines.append(
            f"| {column['name']}{marker} | {column['dtype']} | {column['kind']} | "
            f"{column['missing_pct']}% | {column['unique']} |"
        )
    lines.append("")
    return "\n".join(lines)


def save_report(info: dict[str, Any], path: str | Path) -> tuple[Path, Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path = path.with_suffix(".md")
    json_path = path.with_suffix(".json")
    markdown_path.write_text(format_report(info), encoding="utf-8")
    json_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return markdown_path, json_path


def feature_importance_frame(importances: list[float], names: list[str], top: int = 20) -> pd.DataFrame:
    frame = pd.DataFrame({"feature": names, "importance": importances})
    frame = frame.reindex(frame["importance"].abs().sort_values(ascending=False).index)
    return frame.head(top).reset_index(drop=True)


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [str(col) for col in frame.columns if ptypes.is_numeric_dtype(frame[col])]
