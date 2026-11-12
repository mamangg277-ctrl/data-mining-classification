import json
from pathlib import Path

import pandas as pd
import pytest

import dc.web.app as web_app
from dc.train import TrainConfig, save_result, train_model
from dc.web.app import app, choose_default_target


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 34, 47, 58, 63, 29],
            "sex": [0, 1, 0, 1, 0, 1],
            "approved": [1, 0, 1, 1, 0, 0],
        }
    )


@pytest.fixture()
def storage_dirs(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    db_path = tmp_path / "web_data" / "history.db"
    for directory in (uploads, models, reports, db_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(web_app, "UPLOAD_DIR", uploads)
    monkeypatch.setattr(web_app, "MODELS_DIR", models)
    monkeypatch.setattr(web_app, "REPORTS_DIR", reports)
    monkeypatch.setattr(web_app, "DB_PATH", db_path)
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(uploads))
    web_app.init_db()
    return {"uploads": uploads, "models": models, "reports": reports, "db": db_path}


def insert_run(conn, data_file, model_path):
    conn.execute(
        "INSERT INTO runs (id, type, status, created_at, updated_at, data_file, model_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("run-1", "train", "completed", "2026-01-01", "2026-01-01", str(data_file), str(model_path)),
    )


def insert_prediction(conn, input_file, output_file, model_path):
    conn.execute(
        "INSERT INTO predictions (id, type, input_file, output_file, model_path, threshold, count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "pred-1",
            "batch",
            str(input_file),
            str(output_file),
            str(model_path),
            0.5,
            1,
            "2026-01-01",
        ),
    )


def insert_inspection(conn, data_file, report_md, report_json):
    conn.execute(
        "INSERT INTO inspections (id, data_file, target, report_md, report_json, summary_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("insp-1", str(data_file), "approved", str(report_md), str(report_json), "{}", "2026-01-01"),
    )


def counts() -> dict[str, int]:
    conn = web_app.get_db()
    try:
        return {
            table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("runs", "predictions", "inspections")
        }
    finally:
        conn.close()


def test_storage_page_renders_artifacts(storage_dirs):
    (storage_dirs["uploads"] / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    client = app.test_client()
    response = client.get("/storage")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Storage Management" in html
    assert "data.csv" in html
    assert "Hapus Seluruh Data Penyimpanan" in html


def test_storage_delete_model_removes_files_history_and_reports(storage_dirs):
    model_path = storage_dirs["models"] / "model.joblib"
    metadata_path = storage_dirs["models"] / "model.json"
    model_path.write_bytes(b"model")
    metadata_path.write_text("{}", encoding="utf-8")
    report_path = storage_dirs["reports"] / "predictions_1.csv"
    report_path.write_text("label\n1\n", encoding="utf-8")
    conn = web_app.get_db()
    insert_run(conn, storage_dirs["uploads"] / "data.csv", model_path)
    insert_prediction(conn, storage_dirs["uploads"] / "data.csv", report_path, model_path)
    conn.commit()
    conn.close()

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)

    response = client.post(
        "/storage/delete",
        data={"kind": "model", "name": model_path.name},
    )

    assert response.status_code == 302
    assert not model_path.exists()
    assert not metadata_path.exists()
    assert not report_path.exists()
    assert counts() == {"runs": 0, "predictions": 0, "inspections": 0}
    with client.session_transaction() as session:
        assert "current_model_path" not in session


def test_storage_delete_upload_removes_related_records(storage_dirs):
    upload_path = storage_dirs["uploads"] / "data.csv"
    upload_path.write_text("a,b\n1,2\n", encoding="utf-8")
    report_md = storage_dirs["reports"] / "inspection_1.md"
    report_json = storage_dirs["reports"] / "inspection_1.json"
    prediction_path = storage_dirs["reports"] / "predictions_1.csv"
    for path in (report_md, report_json, prediction_path):
        path.write_text("report", encoding="utf-8")
    conn = web_app.get_db()
    insert_run(conn, upload_path, storage_dirs["models"] / "model.joblib")
    insert_inspection(conn, upload_path, report_md, report_json)
    insert_prediction(conn, upload_path, prediction_path, storage_dirs["models"] / "model.joblib")
    conn.commit()
    conn.close()

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(upload_path)

    response = client.post(
        "/storage/delete",
        data={"kind": "upload", "name": upload_path.name},
    )

    assert response.status_code == 302
    assert not upload_path.exists()
    assert not report_md.exists()
    assert not report_json.exists()
    assert not prediction_path.exists()
    assert counts() == {"runs": 0, "predictions": 0, "inspections": 0}
    with client.session_transaction() as session:
        assert "current_file" not in session


def test_storage_clear_all_removes_artifacts_but_keeps_gitkeep(storage_dirs):
    for directory in ("uploads", "models", "reports"):
        (storage_dirs[directory] / ".gitkeep").write_text("", encoding="utf-8")
    upload_path = storage_dirs["uploads"] / "data.csv"
    upload_path.write_text("a,b\n1,2\n", encoding="utf-8")
    model_path = storage_dirs["models"] / "model.joblib"
    model_path.write_bytes(b"model")
    (storage_dirs["models"] / "model.json").write_text("{}", encoding="utf-8")
    report_path = storage_dirs["reports"] / "predictions_1.csv"
    report_path.write_text("label\n1\n", encoding="utf-8")
    conn = web_app.get_db()
    insert_run(conn, upload_path, model_path)
    insert_prediction(conn, upload_path, report_path, model_path)
    insert_inspection(conn, upload_path, storage_dirs["reports"] / "inspection_1.md", "")
    conn.commit()
    conn.close()

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(upload_path)
        session["current_model_path"] = str(model_path)

    response = client.post("/storage/clear", data={"scope": "all"})

    assert response.status_code == 302
    assert [p.name for p in storage_dirs["uploads"].iterdir()] == [".gitkeep"]
    assert [p.name for p in storage_dirs["models"].iterdir()] == [".gitkeep"]
    assert [p.name for p in storage_dirs["reports"].iterdir()] == [".gitkeep"]
    assert counts() == {"runs": 0, "predictions": 0, "inspections": 0}
    with client.session_transaction() as session:
        assert "current_file" not in session
        assert "current_model_path" not in session


def test_storage_delete_rejects_path_traversal(storage_dirs, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("keep me", encoding="utf-8")

    client = app.test_client()
    response = client.post(
        "/storage/delete",
        data={"kind": "upload", "name": "../secret.txt"},
    )

    assert response.status_code == 302
    assert outside.exists()


def test_choose_default_target_prefers_last_binary_column():
    frame = sample_frame()
    assert choose_default_target(frame) == "approved"
    assert choose_default_target(frame, preferred="sex") == "sex"
    assert choose_default_target(frame, preferred="age") == "approved"


def test_choose_default_target_falls_back_to_last_column():
    frame = pd.DataFrame({"a": range(5), "b": range(5, 10)})
    assert choose_default_target(frame) == "b"


def test_train_page_preselects_binary_target(tmp_path):
    frame = sample_frame()
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(path)
        session["current_columns"] = list(frame.columns)

    response = client.get("/train")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'value="approved" selected' in html
    assert 'value="age" selected' not in html


def test_train_page_rejects_non_binary_target(tmp_path):
    frame = sample_frame()
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(path)
        session["current_columns"] = list(frame.columns)

    response = client.post(
        "/train",
        data={
            "target": "age",
            "test_size": "0.2",
            "cv_folds": "5",
            "seed": "42",
            "scoring": "f1_macro",
            "models": ["logistic"],
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "classification needs between 2 and 20" in html
    assert "approved" in html


def test_train_page_renders_exclude_checkboxes(tmp_path):
    frame = sample_frame()
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(path)

    response = client.get("/train")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'name="exclude" value="age"' in html
    assert 'name="exclude" value="sex"' in html
    assert 'name="exclude" value="approved"' in html
    assert 'id="excludeSelectAll"' in html
    assert 'id="excludeClearAll"' in html


def test_predict_page_renders_dropdowns_example_and_numbers(storage_dirs, binary_frame, tmp_path):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    response = client.get("/predict")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<select class="form-select form-select-sm" name="region" required>' in html
    assert '<option value="north">north</option>' in html
    assert 'type="number" step="any" name="age"' in html
    assert "Contoh Data" in html
    assert "region" in html


def test_predict_page_falls_back_to_dataset_for_legacy_model(storage_dirs, binary_frame, tmp_path):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    result.metadata.pop("feature_values", None)
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    response = client.get("/predict")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert '<option value="north">north</option>' in html


def test_predict_page_manual_submission_with_dropdown_values(storage_dirs, binary_frame, tmp_path):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    response = client.post(
        "/predict",
        data={
            "mode": "manual",
            "age": "41",
            "income": "6400",
            "debt_ratio": "0.25",
            "employment_years": "9",
            "has_default": "True",
            "region": "north",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Hasil Prediksi" in html
    assert "Probabilitas" in html


def test_predict_page_shows_recent_manual_predictions(storage_dirs, binary_frame, tmp_path):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    payload = {
        "mode": "manual",
        "age": "41",
        "income": "6400",
        "debt_ratio": "0.25",
        "employment_years": "9",
        "has_default": "True",
        "region": "north",
    }
    assert client.post("/predict", data=payload).status_code == 200
    payload["age"] = "52"
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    assert "Riwayat Prediksi Manual" in html
    assert "2 terakhir" in html
    assert ">41<" in html
    assert ">52<" in html


def test_train_status_does_not_store_pipeline_in_session(storage_dirs, binary_frame, tmp_path):
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    job_id = "job-pipeline"
    q = web_app.queue.Queue()
    q.put(("result", result))
    web_app.training_queues[job_id] = q
    web_app.training_jobs[job_id] = {"status": "running", "progress": []}
    try:
        client = app.test_client()
        with client.session_transaction() as session:
            session["current_file"] = str(data_path)

        response = client.get(f"/train/status/{job_id}")

        assert response.status_code == 200
        assert response.get_json()["status"] == "done"
        with client.session_transaction() as session:
            assert "current_bundle" not in session
            assert session["current_model_path"]
        assert list(storage_dirs["models"].glob("*.joblib"))
    finally:
        web_app.training_queues.pop(job_id, None)
        web_app.training_jobs.pop(job_id, None)


def test_evaluate_and_predict_reload_model_from_path(storage_dirs, binary_frame, tmp_path):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")
    data_path = tmp_path / "data.csv"
    binary_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    assert client.get("/evaluate").status_code == 200
    assert client.get("/predict").status_code == 200
    assert client.get("/model").status_code == 200
    with client.session_transaction() as session:
        assert "current_bundle" not in session


def _train_and_save(storage_dirs, frame):
    result = train_model(
        frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    return save_result(result, storage_dirs["models"] / "model.joblib")


def test_predict_batch_downloads_csv(storage_dirs, binary_frame, tmp_path):
    model_path = _train_and_save(storage_dirs, binary_frame)
    input_path = tmp_path / "batch.csv"
    binary_frame.drop(columns=["approved"]).to_csv(input_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)

    with input_path.open("rb") as handle:
        response = client.post(
            "/predict/batch",
            data={"file": (handle, "batch.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert "attachment" in response.headers["Content-Disposition"]
    outputs = list(storage_dirs["reports"].glob("predictions_*.csv"))
    assert len(outputs) == 1
    out = pd.read_csv(outputs[0])
    assert "prediction" in out.columns
    assert len(out) == len(binary_frame)

    conn = web_app.get_db()
    try:
        count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    finally:
        conn.close()
    assert count == 1


def test_predict_batch_with_relative_reports_dir(
    storage_dirs, binary_frame, tmp_path, monkeypatch
):
    model_path = _train_and_save(storage_dirs, binary_frame)
    input_path = tmp_path / "batch.csv"
    binary_frame.drop(columns=["approved"]).to_csv(input_path, index=False)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_app, "REPORTS_DIR", Path("reports"))

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)

    with input_path.open("rb") as handle:
        response = client.post(
            "/predict/batch",
            data={"file": (handle, "batch.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    assert list((tmp_path / "reports").glob("predictions_*.csv"))


def test_predict_batch_handles_unreadable_file(storage_dirs, binary_frame, tmp_path):
    model_path = _train_and_save(storage_dirs, binary_frame)
    input_path = tmp_path / "empty.csv"
    input_path.write_text("", encoding="utf-8")

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)

    with input_path.open("rb") as handle:
        response = client.post(
            "/predict/batch",
            data={"file": (handle, "empty.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 302
    assert not list(storage_dirs["reports"].glob("predictions_*.csv"))
    conn = web_app.get_db()
    try:
        count = int(conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    finally:
        conn.close()
    assert count == 0


def test_model_page_shows_saved_model_context(storage_dirs, binary_frame):
    result = train_model(
        binary_frame,
        TrainConfig(target="approved", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "model.joblib")

    client = app.test_client()
    response = client.get("/model")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Target &amp; Tugas" in html
    assert "approved" in html
    assert "binary" in html
    assert "Logistic regression" in html
    assert "F1:" in html
    assert "Detail" in html
    assert "Metrik Uji" in html
    assert "model.joblib" in html

    sidecar = model_path.with_suffix(".json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.pop("task_type", None)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    response = client.get("/model")
    assert response.status_code == 200
    assert "binary" in response.get_data(as_text=True)


def test_upload_keeps_session_cookie_small(storage_dirs, tmp_path):
    frame = pd.DataFrame({"a": range(20), "b": range(20, 40), "approved": [True, False] * 10})
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)

    client = app.test_client()
    with path.open("rb") as handle:
        response = client.post(
            "/upload",
            data={"file": (handle, "data.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 200
    with client.session_transaction() as session:
        assert "current_preview" not in session
        assert "current_shape" not in session
        assert session["current_file"]



def test_multiclass_train_evaluate_and_predict(storage_dirs, multiclass_frame, tmp_path):
    data_path = tmp_path / "data.csv"
    multiclass_frame.to_csv(data_path, index=False)

    client = app.test_client()
    with client.session_transaction() as session:
        session["current_file"] = str(data_path)

    response = client.get("/train")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "risk" in html
    assert "3 kelas" in html

    response = client.post(
        "/train",
        data={
            "target": "risk",
            "test_size": "0.2",
            "cv_folds": "3",
            "seed": "42",
            "scoring": "f1_macro",
            "models": ["logistic"],
        },
    )
    assert response.status_code == 200
    assert "Training..." in response.get_data(as_text=True)

    result = train_model(
        multiclass_frame,
        TrainConfig(target="risk", models=["logistic"], cv_folds=3, seed=7),
    )
    model_path = save_result(result, storage_dirs["models"] / "multiclass.joblib")
    with client.session_transaction() as session:
        session["current_model_path"] = str(model_path)
        session["current_file"] = str(data_path)

    response = client.post("/evaluate", data={"evaluate_on_current": "on"})
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Metrik Per Kelas" in html
    assert "Multikelas" in html

    response = client.get("/predict")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "argmax" in html or "Kelas:" in html

    response = client.post(
        "/predict",
        data={
            "mode": "manual",
            "age": "41",
            "income": "6400",
            "debt_ratio": "0.25",
            "region": "north",
        },
    )
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Hasil Prediksi" in html
    assert "terpilih" in html
