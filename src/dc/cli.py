from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from dc.eda import inspect_dataframe, save_report
from dc.evaluate import compute_metrics, compute_multiclass_metrics, format_metrics
from dc.io import load_dataframe
from dc.models import MODEL_LABELS
from dc.predict import predict_file, predict_proba_matrix, predict_probabilities
from dc.train import TrainConfig, encode_target_values, load_bundle, save_result, train_model

app = typer.Typer(
    name="dc",
    help="Train and use binary or multiclass classification models on CSV, Excel, or JSON data.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _parse_label(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return value


@app.command()
def version() -> None:
    """Show the application version."""
    from dc import __version__

    console.print(f"dataclassifier {__version__}")


@app.command()
def inspect(
    path: Path = typer.Argument(..., help="CSV, XLSX, or JSON data file."),
    target: str | None = typer.Option(None, "--target", "-t", help="Target column to analyze."),
    report: Path | None = typer.Option(
        None, "--report", "-r", help="Base path for markdown/JSON inspection reports."
    ),
) -> None:
    """Inspect a dataset and suggest classification target columns."""
    frame = load_dataframe(path)
    info = inspect_dataframe(frame, target)
    console.print(f"[bold]Rows:[/bold] {info['rows']}  [bold]Columns:[/bold] {info['columns']}")
    console.print(f"[bold]Duplicate rows:[/bold] {info['duplicate_rows']}")
    console.print(f"[bold]Missing cells:[/bold] {info['total_missing']}")
    candidates = info.get("classification_target_candidates", [])
    if candidates:
        console.print(f"[bold]Classification target candidates:[/bold] {', '.join(candidates)}")
    schema = Table(title="Columns")
    for column in ("Column", "Type", "Kind", "Missing %", "Unique"):
        schema.add_column(column)
    for column in info["schema"]:
        name = column["name"] + (" (target)" if column["is_target"] else "")
        schema.add_row(
            name,
            column["dtype"],
            column["kind"],
            f"{column['missing_pct']}%",
            str(column["unique"]),
        )
    console.print(schema)
    if "target" in info:
        balance = info["class_balance"]
        console.print(f"[bold]Target:[/bold] {info['target']}")
        for item in balance["classes"]:
            console.print(f"  {item['label']}: {item['count']} ({item['pct']}%)")
        if balance["imbalanced"]:
            console.print("[yellow]Warning: target is imbalanced.[/yellow]")
    if report is not None:
        markdown_path, json_path = save_report(info, report)
        console.print(f"Report written to {markdown_path} and {json_path}")


@app.command()
def train(
    path: Path = typer.Argument(..., help="CSV, XLSX, or JSON training data."),
    target: str = typer.Option(..., "--target", "-t", help="Target column with 2+ classes."),
    positive_label: str | None = typer.Option(
        None, "--positive", "-p", help="Value treated as the positive class (binary targets only)."
    ),
    output: Path = typer.Option(
        Path("models/model.joblib"), "--output", "-o", help="Where to save the trained model."
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude", "-x", help="Feature columns to leave out (repeatable)."
    ),
    model: list[str] | None = typer.Option(
        None, "--model", "-m", help="Candidate model(s); repeatable. Default: all."
    ),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5),
    cv_folds: int = typer.Option(5, "--cv", min=2, max=20),
    seed: int = typer.Option(42, "--seed"),
    balanced: bool = typer.Option(False, "--balanced", help="Use balanced class weights."),
    search: bool = typer.Option(False, "--search", help="Tune hyperparameters with random search."),
    n_iter: int = typer.Option(15, "--n-iter", min=1, max=200, help="Search iterations per model."),
    scoring: str = typer.Option("f1_macro", "--scoring", help="Cross-validation metric."),
) -> None:
    """Train and compare classifiers, then save the best pipeline."""
    frame = load_dataframe(path)
    config = TrainConfig(
        target=target,
        positive_label=_parse_label(positive_label),
        exclude=exclude or [],
        test_size=test_size,
        cv_folds=cv_folds,
        seed=seed,
        models=model or [],
        class_weight="balanced" if balanced else None,
        search=search,
        n_iter=n_iter,
        scoring=scoring,
    )
    with console.status("Training models..."):
        result = train_model(frame, config)
    table = Table(title=f"Leaderboard ({scoring}, CV)")
    for column in ("Model", "Mean", "Std", "Status"):
        table.add_column(column)
    for row in result.leaderboard:
        table.add_row(
            row["label"],
            "n/a" if row["score"] is None else f"{row['score']:.4f}",
            "n/a" if row["std"] is None else f"{row['std']:.4f}",
            row["error"] or "ok",
        )
    console.print(table)
    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"[bold]Best model:[/bold] {MODEL_LABELS.get(result.model_name, result.model_name)}")
    positive = result.metadata.get("positive_label")
    console.print(
        format_metrics(
            result.test_metrics,
            positive_label=str(positive) if positive is not None else "positive",
        )
    )
    saved = save_result(result, output)
    console.print(f"Model saved to {saved}")
    console.print(f"Metadata saved to {saved.with_suffix('.json')}")


@app.command()
def evaluate(
    model_path: Path = typer.Argument(..., help="Saved model (.joblib)."),
    path: Path = typer.Argument(..., help="Labeled data file."),
    target: str | None = typer.Option(None, "--target", "-t", help="Override target column."),
    threshold: float | None = typer.Option(
        None, "--threshold", min=0.0, max=1.0, help="Decision threshold."
    ),
) -> None:
    """Evaluate a saved model on labeled data."""
    bundle = load_bundle(model_path)
    metadata = bundle.get("metadata", {})
    target_column = target or metadata.get("target")
    if not target_column:
        raise typer.BadParameter("No target column given and none stored in the model metadata.")
    frame = load_dataframe(path)
    if target_column not in frame.columns:
        raise typer.BadParameter(f"Target column '{target_column}' not found in {path}.")
    clean = frame.dropna(subset=[target_column])
    task_type = metadata.get("task_type") or (
        "binary" if metadata.get("positive_label") is not None else "multiclass"
    )
    if task_type == "binary":
        positive = metadata.get("positive_label", 1)
        y = clean[target_column].map(
            lambda value: 1 if (value == positive or str(value) == str(positive)) else 0
        ).to_numpy()
        probabilities = predict_probabilities(bundle, clean)
        chosen_threshold = threshold if threshold is not None else metadata.get("threshold", 0.5)
        metrics = compute_metrics(y, probabilities, chosen_threshold)
    else:
        classes = metadata.get("classes") or []
        if not classes:
            raise typer.BadParameter("Model metadata is missing the class labels.")
        y = encode_target_values(clean[target_column], classes)
        probabilities = predict_proba_matrix(bundle, clean)
        metrics = compute_multiclass_metrics(y, probabilities, classes)
    positive = metadata.get("positive_label")
    console.print(
        format_metrics(
            metrics,
            positive_label=str(positive) if positive is not None else "positive",
        )
    )


@app.command()
def predict(
    model_path: Path = typer.Argument(..., help="Saved model (.joblib)."),
    path: Path = typer.Argument(..., help="Unlabeled data file."),
    output: Path = typer.Option(
        Path("reports/predictions.csv"), "--output", "-o", help="Output file (.csv/.xlsx/.json)."
    ),
    threshold: float | None = typer.Option(
        None, "--threshold", min=0.0, max=1.0, help="Override the decision threshold."
    ),
) -> None:
    """Predict labels for a data file and export the results."""
    result, saved = predict_file(model_path, path, output, threshold=threshold)
    distribution = {str(label): int(count) for label, count in result["prediction"].value_counts().items()}
    console.print(f"Rows scored: {len(result)}  Prediction counts: {distribution}")
    console.print(f"Predictions written to {saved}")


@app.command()
def gui() -> None:
    """Launch the desktop graphical interface."""
    from dc.gui.app import main

    main()


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(5000, help="Port to bind"),
    no_browser: bool = typer.Option(False, help="Do not open browser automatically"),
):
    """Launch the browser-based web interface (offline)."""
    from dc.web.cli import main as web_main

    web_main(host=host, port=port, no_browser=no_browser)


if __name__ == "__main__":
    app()
