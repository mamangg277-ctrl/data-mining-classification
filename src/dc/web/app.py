"""Web interface for Data Classifier with persistent history."""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from dc.eda import inspect_dataframe, remove_duplicate_rows, save_report
from dc.evaluate import compute_metrics, compute_multiclass_metrics
from dc.io import (
    SUPPORTED_EXTENSIONS,
    feature_value_options,
    find_binary_targets,
    find_classification_targets,
    load_dataframe,
    save_dataframe,
    target_class_count,
)
from dc.models import MODEL_LABELS
from dc.predict import (
    predict_frame,
    predict_proba_matrix,
    predict_probabilities,
    predict_record,
)
from dc.train import (
    TrainConfig,
    TrainResult,
    encode_target_values,
    load_bundle,
    save_result,
    train_model,
)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("DC_SECRET_KEY") or "dataclassifier-dev-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024  # 256MB

def _workspace_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


UPLOAD_DIR = _workspace_path(os.environ.get("DC_UPLOAD_DIR", Path("data") / "uploads"))
MODELS_DIR = _workspace_path(os.environ.get("DC_MODELS_DIR", "models"))
REPORTS_DIR = _workspace_path(os.environ.get("DC_REPORTS_DIR", "reports"))
DB_PATH = _workspace_path(os.environ.get("DC_DB_PATH", Path(".web_data") / "history.db"))
for _directory in (UPLOAD_DIR, MODELS_DIR, REPORTS_DIR, DB_PATH.parent):
    _directory.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

FILE_TYPES = {".csv", ".xlsx", ".xls", ".json"}
MODEL_SUFFIXES = {".joblib", ".json"}
REPORT_SUFFIXES = {".md", ".json"}

training_jobs: dict[str, dict[str, Any]] = {}
training_queues: dict[str, queue.Queue] = {}


def choose_default_target(frame: pd.DataFrame, preferred: str | None = None) -> str | None:
    """Pick a sensible classification target for the training form."""
    columns = [str(column) for column in frame.columns]
    if not columns:
        return None
    targets = find_classification_targets(frame)
    target_set = set(targets)
    if preferred is not None and str(preferred) in target_set:
        return str(preferred)
    last = columns[-1]
    if last in target_set:
        return last
    binary = set(find_binary_targets(frame))
    binary_columns = [column for column in columns if column in binary]
    if binary_columns:
        return binary_columns[-1]
    if targets:
        return targets[-1]
    return last


def _target_options(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"name": name, "classes": target_class_count(frame[name])}
        for name in find_classification_targets(frame)
    ]


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            data_file TEXT,
            model_path TEXT,
            target TEXT,
            positive_label TEXT,
            config_json TEXT,
            result_json TEXT,
            metadata_json TEXT,
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            run_id TEXT,
            type TEXT NOT NULL,
            input_file TEXT,
            output_file TEXT,
            model_path TEXT,
            threshold REAL,
            count INTEGER,
            created_at TEXT NOT NULL,
            result_json TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS manual_predictions (
            id TEXT PRIMARY KEY,
            model_path TEXT,
            inputs_json TEXT,
            result_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id TEXT PRIMARY KEY,
            data_file TEXT,
            target TEXT,
            report_md TEXT,
            report_json TEXT,
            summary_json TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def save_run(run_data: dict[str, Any]) -> str:
    run_id = run_data.get("id") or str(uuid.uuid4())
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO runs (id, type, status, created_at, updated_at, data_file, model_path, target, positive_label, config_json, result_json, metadata_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            run_data.get("type", "train"),
            run_data.get("status", "completed"),
            run_data.get("created_at", datetime.now().isoformat()),
            run_data.get("updated_at", datetime.now().isoformat()),
            run_data.get("data_file"),
            run_data.get("model_path"),
            run_data.get("target"),
            run_data.get("positive_label"),
            json.dumps(run_data.get("config")) if run_data.get("config") else None,
            json.dumps(run_data.get("result")) if run_data.get("result") else None,
            json.dumps(run_data.get("metadata")) if run_data.get("metadata") else None,
            run_data.get("notes"),
        ),
    )
    conn.commit()
    conn.close()
    return run_id


def get_runs(limit: int = 100) -> list[dict[str, Any]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_run(run_id: str) -> dict[str, Any] | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(size)} B"


def _safe_child(base: Path, name: str | None) -> Path | None:
    if not name:
        return None
    candidate = Path(name)
    if candidate.is_absolute() or candidate.name != name:
        return None
    base_resolved = base.resolve()
    target = base / name
    try:
        if target.resolve().parent != base_resolved:
            return None
    except OSError:
        return None
    return target


def _artifact_in_dir(base: Path, value: str | Path | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        resolved = path.resolve()
        if resolved.parent != base.resolve():
            return None
    except OSError:
        return None
    return resolved


def _path_variants(path: Path) -> tuple[str, ...]:
    values = [str(path)]
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = ""
    if resolved and resolved not in values:
        values.append(resolved)
    return tuple(values)


def _session_matches(key: str, path: Path) -> bool:
    stored = session.get(key)
    if not stored:
        return False
    try:
        return Path(stored).resolve() == path.resolve()
    except OSError:
        return str(stored) == str(path)


def _clear_loaded_file_session() -> None:
    for key in ("current_file", "current_columns", "current_shape", "current_preview", "target"):
        session.pop(key, None)


def _clear_loaded_model_session() -> None:
    for key in ("current_bundle", "current_model_path", "current_run_id"):
        session.pop(key, None)


def _get_active_file_path() -> Path | None:
    """Return the active dataset path if it exists on disk."""
    path_str = session.get("current_file")
    if path_str:
        p = Path(path_str)
        if p.is_file():
            return p
    return None


def _load_current_bundle() -> dict[str, Any] | None:
    """Reload the active model from disk instead of the JSON session cookie."""
    path = session.get("current_model_path")
    if path and Path(path).is_file():
        try:
            return load_bundle(path)
        except Exception:
            pass
    return None


def _remove_file(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass


def _remove_report_files(path: Path) -> None:
    if path.suffix.lower() in REPORT_SUFFIXES:
        _remove_file(path.with_suffix(".md"))
        _remove_file(path.with_suffix(".json"))
    else:
        _remove_file(path)


def _entry(name: str, paths: list[Path]) -> dict[str, Any]:
    total = 0
    latest = 0.0
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        latest = max(latest, stat.st_mtime)
    modified = datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M:%S") if latest else ""
    return {
        "name": name,
        "files": [path.name for path in paths],
        "size": total,
        "size_text": _format_size(total),
        "mtime": modified,
        "mtime_raw": latest,
    }


def list_uploads() -> list[dict[str, Any]]:
    entries = [
        _entry(path.name, [path])
        for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.name != ".gitkeep"
    ]
    entries.sort(key=lambda item: item["mtime_raw"], reverse=True)
    return entries


def list_models() -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = {}
    for path in MODELS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            groups.setdefault(path.stem, []).append(path)
    entries = []
    for paths in groups.values():
        paths.sort(key=lambda path: (path.suffix.lower() != ".joblib", path.name))
        entries.append(_entry(paths[0].name, paths))
    entries.sort(key=lambda item: item["mtime_raw"], reverse=True)
    return entries


def list_reports() -> list[dict[str, Any]]:
    groups: dict[str, list[Path]] = {}
    for path in REPORTS_DIR.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            groups.setdefault(path.stem, []).append(path)
    entries = []
    for paths in groups.values():
        paths.sort(key=lambda path: (path.suffix.lower() != ".md", path.name))
        entries.append(_entry(paths[0].name, paths))
    entries.sort(key=lambda item: item["mtime_raw"], reverse=True)
    return entries


def history_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    conn = get_db()
    try:
        cur = conn.cursor()
        for table in ("runs", "predictions", "inspections"):
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = int(cur.fetchone()[0])
    finally:
        conn.close()
    return counts


def delete_upload(name: str | None) -> bool:
    path = _safe_child(UPLOAD_DIR, name)
    if path is None or not path.is_file():
        return False
    variants = _path_variants(path)
    placeholders = ",".join("?" for _ in variants)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT report_md, report_json FROM inspections WHERE data_file IN ({placeholders})",
            variants,
        )
        for row in cur.fetchall():
            for value in (row["report_md"], row["report_json"]):
                report_path = _artifact_in_dir(REPORTS_DIR, value)
                if report_path is not None:
                    _remove_report_files(report_path)
        cur.execute(
            f"SELECT output_file FROM predictions WHERE input_file IN ({placeholders})",
            variants,
        )
        for row in cur.fetchall():
            output_path = _artifact_in_dir(REPORTS_DIR, row["output_file"])
            if output_path is not None:
                _remove_file(output_path)
        cur.execute(f"DELETE FROM inspections WHERE data_file IN ({placeholders})", variants)
        cur.execute(f"DELETE FROM predictions WHERE input_file IN ({placeholders})", variants)
        cur.execute(f"DELETE FROM runs WHERE data_file IN ({placeholders})", variants)
        conn.commit()
    finally:
        conn.close()
    _remove_file(path)
    if _session_matches("current_file", path):
        _clear_loaded_file_session()
    return True


def delete_model(name: str | None) -> bool:
    path = _safe_child(MODELS_DIR, name)
    if path is None or path.suffix.lower() not in MODEL_SUFFIXES:
        return False
    model_path = path.with_suffix(".joblib")
    metadata_path = path.with_suffix(".json")
    if not model_path.exists() and not metadata_path.exists():
        return False
    variants = _path_variants(model_path)
    placeholders = ",".join("?" for _ in variants)
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            f"SELECT output_file FROM predictions WHERE model_path IN ({placeholders})",
            variants,
        )
        for row in cur.fetchall():
            output_path = _artifact_in_dir(REPORTS_DIR, row["output_file"])
            if output_path is not None:
                _remove_file(output_path)
        cur.execute(f"DELETE FROM predictions WHERE model_path IN ({placeholders})", variants)
        cur.execute(
            f"DELETE FROM manual_predictions WHERE model_path IN ({placeholders})",
            variants,
        )
        cur.execute(f"DELETE FROM runs WHERE model_path IN ({placeholders})", variants)
        conn.commit()
    finally:
        conn.close()
    _remove_file(model_path)
    _remove_file(metadata_path)
    if _session_matches("current_model_path", model_path):
        _clear_loaded_model_session()
    return True


def delete_report(name: str | None) -> bool:
    path = _safe_child(REPORTS_DIR, name)
    if path is None or path.name == ".gitkeep":
        return False
    if path.suffix.lower() in REPORT_SUFFIXES:
        targets = [path.with_suffix(".md"), path.with_suffix(".json")]
    else:
        targets = [path]
    if not any(target.exists() for target in targets):
        return False
    conn = get_db()
    try:
        cur = conn.cursor()
        for target in targets:
            variants = _path_variants(target)
            placeholders = ",".join("?" for _ in variants)
            cur.execute(
                f"DELETE FROM inspections WHERE report_md IN ({placeholders}) OR report_json IN ({placeholders})",
                variants + variants,
            )
            cur.execute(
                f"DELETE FROM predictions WHERE output_file IN ({placeholders})",
                variants,
            )
        conn.commit()
    finally:
        conn.close()
    for target in targets:
        _remove_file(target)
    return True


def clear_uploads() -> int:
    removed = 0
    for entry in list_uploads():
        if delete_upload(entry["name"]):
            removed += 1
    _clear_loaded_file_session()
    return removed


def clear_models() -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT output_file FROM predictions")
        for row in cur.fetchall():
            output_path = _artifact_in_dir(REPORTS_DIR, row["output_file"])
            if output_path is not None:
                _remove_file(output_path)
        cur.execute("DELETE FROM predictions")
        cur.execute("DELETE FROM manual_predictions")
        cur.execute("DELETE FROM runs")
        conn.commit()
    finally:
        conn.close()
    removed = 0
    for path in MODELS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES:
            _remove_file(path)
            removed += 1
    _clear_loaded_model_session()
    return removed


def clear_reports() -> int:
    removed = 0
    for path in REPORTS_DIR.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            _remove_file(path)
            removed += 1
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions")
        cur.execute("DELETE FROM inspections")
        conn.commit()
    finally:
        conn.close()
    return removed


def clear_history() -> None:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM runs")
        cur.execute("DELETE FROM predictions")
        cur.execute("DELETE FROM manual_predictions")
        cur.execute("DELETE FROM inspections")
        conn.commit()
        try:
            conn.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()


def clear_all() -> int:
    removed = clear_uploads()
    removed += clear_models()
    removed += clear_reports()
    clear_history()
    _clear_loaded_file_session()
    _clear_loaded_model_session()
    return removed


@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html", model_labels=MODEL_LABELS)


@app.route("/data")
def data_page():
    path = _get_active_file_path()
    if path and path.is_file():
        try:
            frame = load_dataframe(str(path))
            session["current_file"] = str(path)
            session["current_columns"] = list(frame.columns)
            if "target" not in session or session.get("target") not in frame.columns:
                session["target"] = choose_default_target(frame)
            target_options = _target_options(frame)
            preview = frame.head(100).to_dict(orient="records")
            clean_filename = path.name.split("_", 2)[-1] if "_" in path.name else path.name
            return render_template(
                "data.html",
                model_labels=MODEL_LABELS,
                filename=clean_filename,
                shape=frame.shape,
                columns=list(frame.columns),
                preview=preview,
                target_options=target_options,
            )
        except Exception:
            pass
    return render_template("data.html", model_labels=MODEL_LABELS)


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("data_page"))
    f = request.files["file"]
    if f.filename == "":
        flash("No file selected", "error")
        return redirect(url_for("data_page"))
    ext = Path(f.filename).suffix.lower()
    if ext not in FILE_TYPES:
        flash("Unsupported file type. Use CSV, XLSX, or JSON.", "error")
        return redirect(url_for("data_page"))
    filename = secure_filename(f.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"{timestamp}_{filename}"
    path = UPLOAD_DIR / saved_name
    f.save(path)
    try:
        frame = load_dataframe(str(path))
    except Exception as e:
        flash(f"Failed to load file: {e}", "error")
        return redirect(url_for("data_page"))
    target_options = _target_options(frame)
    session["current_file"] = str(path)
    session["current_columns"] = list(frame.columns)
    session["target"] = choose_default_target(frame)
    preview = frame.head(100).to_dict(orient="records")
    flash(f"Loaded {filename} ({frame.shape[0]} rows × {frame.shape[1]} columns)", "success")
    return render_template(
        "data.html",
        model_labels=MODEL_LABELS,
        filename=filename,
        shape=frame.shape,
        columns=list(frame.columns),
        preview=preview,
        target_options=target_options,
    )


def _inspect_and_save(frame: pd.DataFrame, path: Path, target: str | None) -> dict[str, Any]:
    info = inspect_dataframe(frame, target)
    try:
        base_path = REPORTS_DIR / f"inspection_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.md"
        markdown_path, json_path = save_report(info, str(base_path.with_suffix("")))
        conn = get_db()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO inspections (id, data_file, target, report_md, report_json, summary_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    str(path),
                    target,
                    str(markdown_path),
                    str(json_path),
                    json.dumps(info),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass
    return info


@app.route("/explore", methods=["GET", "POST"])
def explore():
    path = _get_active_file_path()
    if not path or not path.is_file():
        flash("Silakan upload atau pilih file data terlebih dahulu.", "info")
        return redirect(url_for("data_page"))
    try:
        frame = load_dataframe(str(path))
    except Exception as e:
        flash(f"Gagal memuat dataset: {e}", "error")
        return redirect(url_for("data_page"))
    session["current_file"] = str(path)
    target = request.form.get("target") or session.get("target")
    if not target or target not in frame.columns:
        target = choose_default_target(frame)
    if target:
        session["target"] = target
    info = _inspect_and_save(frame, path, target)
    return render_template("explore.html", info=info, target=target, model_labels=MODEL_LABELS)


@app.route("/explore/deduplicate", methods=["POST"])
def deduplicate():
    path = _get_active_file_path()
    if not path or not path.is_file():
        flash("Silakan upload atau pilih file data terlebih dahulu.", "info")
        return redirect(url_for("data_page"))
    try:
        frame = load_dataframe(str(path))
    except Exception as e:
        flash(f"Gagal memuat dataset: {e}", "error")
        return redirect(url_for("data_page"))

    cleaned, removed = remove_duplicate_rows(frame)
    if removed == 0:
        flash("Tidak ada baris duplikat yang perlu dihapus.", "info")
        return redirect(url_for("explore"))

    target = session.get("target")
    if not target or target not in cleaned.columns:
        target = choose_default_target(cleaned)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        suffix = ".csv"
    stem = secure_filename(path.stem) or "dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = UPLOAD_DIR / f"{timestamp}_{stem}_dedup{suffix}"
    counter = 1
    while output_path.exists():
        output_path = UPLOAD_DIR / f"{timestamp}_{stem}_dedup_{counter}{suffix}"
        counter += 1
    try:
        saved_path = save_dataframe(cleaned, str(output_path))
    except Exception as e:
        flash(f"Gagal menyimpan dataset bersih: {e}", "error")
        return redirect(url_for("explore"))

    session["current_file"] = str(saved_path)
    if target:
        session["target"] = target
    _inspect_and_save(cleaned, saved_path, target)
    flash(
        f"Menghapus {removed} baris duplikat. Dataset bersih disimpan sebagai "
        f"{saved_path.name}; file asli tetap tersimpan.",
        "success",
    )
    return redirect(url_for("explore"))


@app.route("/train", methods=["GET", "POST"])
def train_page():
    path = _get_active_file_path()
    if not path or not path.is_file():
        flash("Silakan upload atau pilih file data terlebih dahulu.", "info")
        return redirect(url_for("data_page"))
    try:
        frame = load_dataframe(str(path))
    except Exception as e:
        flash(f"Gagal memuat dataset: {e}", "error")
        return redirect(url_for("data_page"))
    session["current_file"] = str(path)
    columns = list(frame.columns)
    session["current_columns"] = columns
    classification_columns = find_classification_targets(frame)
    binary_columns = find_binary_targets(frame)
    target_options = _target_options(frame)
    class_counts = {item["name"]: item["classes"] for item in target_options}
    if request.method == "GET":
        selected_target = choose_default_target(frame, session.get("target"))
        session["target"] = selected_target
        return render_template(
            "train.html",
            columns=columns,
            selected_target=selected_target,
            classification_columns=classification_columns,
            binary_columns=binary_columns,
            class_counts=class_counts,
            model_labels=MODEL_LABELS,
        )
    # start training
    target = request.form.get("target")
    positive_label = request.form.get("positive_label") or None
    exclude = request.form.getlist("exclude")
    models = request.form.getlist("models") or list(MODEL_LABELS.keys())
    test_size = float(request.form.get("test_size", 0.2))
    cv_folds = int(request.form.get("cv_folds", 5))
    seed = int(request.form.get("seed", 42))
    balanced = request.form.get("balanced") == "on"
    search = request.form.get("search") == "on"
    scoring = request.form.get("scoring", "f1_macro")
    if not target:
        flash("Select a target column", "error")
        return render_template(
            "train.html",
            columns=columns,
            selected_target=None,
            classification_columns=classification_columns,
            binary_columns=binary_columns,
            class_counts=class_counts,
            model_labels=MODEL_LABELS,
        )
    if target not in frame.columns:
        flash(f"Target column '{target}' was not found in the uploaded file.", "error")
        return render_template(
            "train.html",
            columns=columns,
            selected_target=target,
            classification_columns=classification_columns,
            binary_columns=binary_columns,
            class_counts=class_counts,
            model_labels=MODEL_LABELS,
        )
    if target not in classification_columns:
        class_count = target_class_count(frame[target])
        message = (
            f"Target '{target}' has {class_count} class(es); classification needs between "
            "2 and 20 classes."
        )
        if classification_columns:
            message += f" Choose a classification column such as '{classification_columns[-1]}'."
        else:
            message += " No classification columns were detected in this file."
        flash(message, "error")
        return render_template(
            "train.html",
            columns=columns,
            selected_target=target,
            classification_columns=classification_columns,
            binary_columns=binary_columns,
            class_counts=class_counts,
            model_labels=MODEL_LABELS,
        )
    if target not in binary_columns:
        positive_label = None
    session["target"] = target
    config = TrainConfig(
        target=target,
        positive_label=positive_label,
        exclude=exclude,
        test_size=test_size,
        cv_folds=cv_folds,
        seed=seed,
        models=models,
        class_weight="balanced" if balanced else None,
        search=search,
        scoring=scoring,
    )
    job_id = str(uuid.uuid4())
    q = queue.Queue()
    training_queues[job_id] = q
    training_jobs[job_id] = {"status": "running", "progress": []}

    def worker():
        try:
            def progress(msg, frac):
                q.put(("progress", msg, frac))

            result = train_model(frame, config, progress=progress)
            q.put(("result", result))
        except Exception as e:
            q.put(("error", str(e)))

    threading.Thread(target=worker, daemon=True).start()
    session["current_job"] = job_id
    session["target"] = target
    return render_template("train_progress.html", job_id=job_id, model_labels=MODEL_LABELS)


@app.route("/train/status/<job_id>")
def train_status(job_id):
    q = training_queues.get(job_id)
    if not q:
        return jsonify({"status": "not_found"})
    messages = []
    try:
        while True:
            messages.append(q.get_nowait())
    except queue.Empty:
        pass
    if not messages:
        return jsonify({"status": "running", "messages": []})
    for msg in messages:
        if msg[0] == "result":
            result: TrainResult = msg[1]
            try:
                out_path = MODELS_DIR / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
                saved = save_result(result, str(out_path))
                run_id = save_run(
                    {
                        "type": "train",
                        "status": "completed",
                        "data_file": session.get("current_file"),
                        "model_path": str(saved),
                        "target": result.metadata.get("target"),
                        "positive_label": (
                            str(result.metadata.get("positive_label"))
                            if result.metadata.get("positive_label") is not None
                            else None
                        ),
                        "config": {
                            "task_type": result.metadata.get("task_type"),
                            "test_size": result.metadata.get("test_size"),
                            "cv_folds": result.metadata.get("cv_folds"),
                            "scoring": result.metadata.get("scoring"),
                        },
                        "result": {
                            "model_name": result.model_name,
                            "test_metrics": result.test_metrics,
                            "leaderboard": result.leaderboard,
                        },
                        "metadata": result.metadata,
                    }
                )
            except Exception as e:
                return jsonify(
                    {"status": "error", "error": f"Training finished but saving failed: {e}"}
                )
            session["current_model_path"] = str(saved)
            session["current_run_id"] = run_id
            session.pop("current_bundle", None)
            return jsonify({"status": "done", "redirect": url_for("evaluate_page")})
        elif msg[0] == "error":
            return jsonify({"status": "error", "error": msg[1]})
        elif msg[0] == "progress":
            pass
    return jsonify({"status": "running", "messages": messages})


@app.route("/evaluate", methods=["GET", "POST"])
def evaluate_page():
    bundle = _load_current_bundle()
    if not bundle:
        flash("Train or load a model first.", "error")
        return redirect(url_for("train_page"))
    metadata = bundle.get("metadata", {})
    task_type = metadata.get("task_type") or (
        "binary" if metadata.get("positive_label") is not None else "multiclass"
    )
    default_threshold = float(metadata.get("threshold", 0.5))
    if task_type == "binary" and request.method == "POST":
        threshold = float(request.form.get("threshold", default_threshold))
    else:
        threshold = default_threshold
    metrics = None
    metrics_source = None
    # try to evaluate on current file if labeled
    if "current_file" in session and request.method == "POST" and request.form.get("evaluate_on_current"):
        try:
            frame = load_dataframe(session["current_file"])
            target = metadata.get("target")
            if target and target in frame.columns:
                clean = frame.dropna(subset=[target])
                if task_type == "binary":
                    positive = metadata.get("positive_label")
                    y = clean[target].map(
                        lambda v: 1 if (v == positive or str(v) == str(positive)) else 0
                    ).to_numpy()
                    if len(np.unique(y)) >= 2:
                        probs = predict_probabilities(bundle, clean)
                        metrics = compute_metrics(y, probs, threshold)
                else:
                    classes = metadata.get("classes") or []
                    if not classes:
                        raise ValueError("Model metadata is missing the class labels.")
                    y = encode_target_values(clean[target], classes)
                    probs = predict_proba_matrix(bundle, clean)
                    metrics = compute_multiclass_metrics(y, probs, classes)
        except Exception as e:
            flash(f"Evaluation failed: {e}", "error")

    if metrics is not None:
        metrics_source = "current"
    else:
        stored_metrics = metadata.get("test_metrics")
        if isinstance(stored_metrics, dict) and stored_metrics.get("confusion_matrix"):
            metrics = stored_metrics
            metrics_source = "training"
            if task_type == "binary":
                threshold = float(stored_metrics.get("threshold", default_threshold))
    session["threshold"] = threshold
    return render_template(
        "evaluate.html",
        bundle=bundle,
        threshold=threshold,
        metrics=metrics,
        metrics_source=metrics_source,
        task_type=task_type,
        classes=metadata.get("classes") or [],
        model_labels=MODEL_LABELS,
    )


def _numeric_feature_names(metadata: dict[str, Any], features: list[str]) -> set[str]:
    dtypes = metadata.get("feature_dtypes", {})
    numeric = set()
    for name in features:
        try:
            if np.dtype(str(dtypes.get(name, ""))).kind in "iuf":
                numeric.add(name)
        except TypeError:
            continue
    return numeric


def _example_record(frame: pd.DataFrame, features: list[str], target: str | None) -> dict[str, Any] | None:
    if not features or any(f not in frame.columns for f in features):
        return None
    complete = frame.dropna(subset=features)
    row = (complete if not complete.empty else frame).iloc[0]
    record = {}
    for f in features:
        value = row[f]
        if hasattr(value, "item"):
            value = value.item()
        record[f] = "" if pd.isna(value) else value
    result: dict[str, Any] = {"record": record}
    if target and target in frame.columns:
        actual = row[target]
        result["target"] = actual.item() if hasattr(actual, "item") else actual
    return result


def save_manual_prediction(
    model_path: str | None,
    record: dict[str, Any],
    result: dict[str, Any],
) -> None:
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO manual_predictions (id, model_path, inputs_json, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                model_path,
                json.dumps(record, default=str),
                json.dumps(result, default=str),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_manual_predictions(
    model_path: str | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT inputs_json, result_json, created_at FROM manual_predictions "
            "WHERE model_path = ? ORDER BY created_at DESC LIMIT ?",
            (model_path, limit),
        ).fetchall()
    finally:
        conn.close()
    entries = []
    for row in rows:
        try:
            inputs = json.loads(row["inputs_json"] or "{}")
            result = json.loads(row["result_json"] or "{}")
        except (TypeError, ValueError):
            continue
        entries.append(
            {
                "inputs": inputs,
                "result": result,
                "created_at": row["created_at"],
            }
        )
    return entries


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    bundle = _load_current_bundle()
    if not bundle:
        flash("Train or load a model first.", "error")
        return redirect(url_for("train_page"))
    metadata = bundle.get("metadata", {})
    task_type = metadata.get("task_type") or (
        "binary" if metadata.get("positive_label") is not None else "multiclass"
    )
    features = metadata.get("feature_columns", [])
    threshold = float(session.get("threshold", metadata.get("threshold", 0.5)))
    feature_values = dict(metadata.get("feature_values") or {})
    example = None
    path = _get_active_file_path()
    if path:
        try:
            frame = load_dataframe(str(path))
        except Exception:
            frame = None
        if frame is not None:
            unresolved = [f for f in features if f not in feature_values and f in frame.columns]
            if unresolved:
                feature_values.update(feature_value_options(frame, unresolved))
            example = _example_record(frame, features, metadata.get("target"))
    example_target_positive = bool(
        example
        and example.get("target") is not None
        and str(example.get("target")) == str(metadata.get("positive_label"))
    )
    example_target_label = (
        str(example.get("target"))
        if example and example.get("target") is not None
        else None
    )
    numeric_features = _numeric_feature_names(metadata, features)
    model_path = session.get("current_model_path")
    result = None
    if request.method == "POST" and request.form.get("mode") == "manual":
        record = {}
        for f in features:
            record[f] = request.form.get(f)
        try:
            result = predict_record(bundle, record, threshold=threshold)
        except Exception as e:
            flash(f"Prediction failed: {e}", "error")
        else:
            save_manual_prediction(model_path, record, result)
    recent_predictions = get_recent_manual_predictions(model_path)
    return render_template(
        "predict.html",
        bundle=bundle,
        features=features,
        threshold=threshold,
        result=result,
        recent_predictions=recent_predictions,
        feature_values=feature_values,
        numeric_features=numeric_features,
        example=example,
        example_target_positive=example_target_positive,
        example_target_label=example_target_label,
        task_type=task_type,
        classes=metadata.get("classes") or [],
        model_labels=MODEL_LABELS,
    )


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    bundle = _load_current_bundle()
    if not bundle:
        flash("Train or load a model first.", "error")
        return redirect(url_for("predict_page"))
    if "file" not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for("predict_page"))
    f = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in FILE_TYPES:
        flash("Unsupported file type", "error")
        return redirect(url_for("predict_page"))
    filename = secure_filename(f.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_name = f"batch_{timestamp}_{filename}"
    path = UPLOAD_DIR / saved_name
    threshold = float(session.get("threshold", bundle.get("metadata", {}).get("threshold", 0.5)))
    try:
        f.save(path)
        frame = load_dataframe(str(path))
        out = predict_frame(bundle, frame, threshold=threshold)
    except Exception as e:
        flash(f"Batch prediction failed: {e}", "error")
        return redirect(url_for("predict_page"))
    out_path = REPORTS_DIR / f"predictions_{timestamp}.csv"
    try:
        save_dataframe(out, str(out_path))
    except Exception as e:
        flash(f"Could not save batch prediction: {e}", "error")
        return redirect(url_for("predict_page"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (id, type, input_file, output_file, model_path, threshold, count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), "batch", str(path), str(out_path), session.get("current_model_path"), threshold, len(out), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return send_file(str(out_path.resolve()), as_attachment=True, download_name=out_path.name)


def _read_model_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _model_task_type(metadata: dict[str, Any]) -> str | None:
    task_type = metadata.get("task_type")
    if task_type in {"binary", "multiclass"}:
        return str(task_type)
    classes = metadata.get("classes") or []
    if metadata.get("positive_label") is not None or len(classes) == 2:
        return "binary"
    if len(classes) > 2:
        return "multiclass"
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_entry(path: Path) -> dict[str, Any]:
    metadata = _read_model_metadata(path)
    metrics = metadata.get("test_metrics") or {}
    classes = metadata.get("classes") or []
    feature_columns = metadata.get("feature_columns") or []
    performance = _as_float(metrics.get("f1"))
    if performance is None:
        performance = _as_float(metrics.get("accuracy"))
    return {
        "path": str(path),
        "name": path.name,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        "metadata": metadata,
        "has_metadata": bool(metadata),
        "target": metadata.get("target"),
        "task_type": _model_task_type(metadata),
        "model_label": metadata.get("model_label") or metadata.get("model_name"),
        "model_name": metadata.get("model_name"),
        "classes": classes,
        "class_count": metadata.get("class_count") or (len(classes) if classes else None),
        "positive_label": metadata.get("positive_label"),
        "threshold": metadata.get("threshold"),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "rows": metadata.get("rows"),
        "scoring": metadata.get("scoring"),
        "cv_folds": metadata.get("cv_folds"),
        "created_at": metadata.get("created_at"),
        "performance": performance,
        "roc_auc": _as_float(metrics.get("roc_auc")),
        "metrics": metrics,
        "leaderboard": metadata.get("leaderboard") or [],
        "warnings": metadata.get("warnings") or [],
        "bundle_version": metadata.get("bundle_version"),
        "seed": metadata.get("seed"),
    }


@app.route("/model")
def model_page():
    bundle = _load_current_bundle()
    models = [_model_entry(p) for p in MODELS_DIR.glob("*.joblib")]
    models.sort(key=lambda x: x["mtime"], reverse=True)
    return render_template("model.html", bundle=bundle, models=models, model_labels=MODEL_LABELS)


@app.route("/model/load", methods=["POST"])
def model_load():
    path = request.form.get("path")
    if not path or not Path(path).exists():
        flash("Model not found", "error")
        return redirect(url_for("model_page"))
    try:
        load_bundle(path)
    except Exception as e:
        flash(f"Failed to load model: {e}", "error")
    else:
        session["current_model_path"] = path
        session.pop("current_bundle", None)
        flash(f"Loaded model {Path(path).name}", "success")
    return redirect(url_for("predict_page"))


@app.route("/history")
def history_page():
    runs = get_runs(200)
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 100")
    preds = [dict(r) for r in cur.fetchall()]
    cur.execute("SELECT * FROM inspections ORDER BY created_at DESC LIMIT 100")
    insps = [dict(r) for r in cur.fetchall()]
    conn.close()
    return render_template("history.html", runs=runs, predictions=preds, inspections=insps, model_labels=MODEL_LABELS)


@app.route("/storage")
def storage_page():
    uploads = list_uploads()
    models = list_models()
    reports = list_reports()
    counts = history_counts()
    uploads_size = sum(item["size"] for item in uploads)
    models_size = sum(item["size"] for item in models)
    reports_size = sum(item["size"] for item in reports)
    db_size = DB_PATH.stat().st_size if DB_PATH.exists() else 0
    total_size = uploads_size + models_size + reports_size + db_size
    return render_template(
        "storage.html",
        uploads=uploads,
        models=models,
        reports=reports,
        counts=counts,
        uploads_size=_format_size(uploads_size),
        models_size=_format_size(models_size),
        reports_size=_format_size(reports_size),
        db_size=_format_size(db_size),
        total_size=_format_size(total_size),
        model_labels=MODEL_LABELS,
    )


@app.route("/storage/delete", methods=["POST"])
def storage_delete():
    kind = request.form.get("kind")
    name = request.form.get("name")
    deleted = False
    if kind == "upload":
        deleted = delete_upload(name)
        label = "Uploaded file"
    elif kind == "model":
        deleted = delete_model(name)
        label = "Model"
    elif kind == "report":
        deleted = delete_report(name)
        label = "Report"
    else:
        label = "Item"
    if deleted:
        flash(f"{label} deleted.", "success")
    else:
        flash(f"{label} could not be deleted.", "error")
    if request.form.get("return_to") == "model":
        return redirect(url_for("model_page"))
    return redirect(url_for("storage_page"))


@app.route("/storage/clear", methods=["POST"])
def storage_clear():
    scope = request.form.get("scope")
    if scope == "uploads":
        removed = clear_uploads()
        flash(f"Deleted {removed} uploaded file(s) and related history.", "success")
    elif scope == "models":
        removed = clear_models()
        flash(f"Deleted {removed} model file(s) and related history.", "success")
    elif scope == "reports":
        removed = clear_reports()
        flash(f"Deleted {removed} report file(s) and related history.", "success")
    elif scope == "history":
        clear_history()
        flash("History cleared.", "success")
    elif scope == "all":
        removed = clear_all()
        flash(f"Deleted {removed} artifact file(s) and cleared history.", "success")
    else:
        flash("Unknown cleanup option.", "error")
    return redirect(url_for("storage_page"))


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    return train_status(job_id)


def main():
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    import threading
    import webbrowser

    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://{host}:{port}/")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host=host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
