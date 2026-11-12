from typer.testing import CliRunner

from dc.cli import app
from dc.io import save_dataframe

runner = CliRunner()


def test_inspect_command(tmp_path, binary_frame):
    path = tmp_path / "data.csv"
    save_dataframe(binary_frame, path)
    result = runner.invoke(app, ["inspect", str(path), "--target", "approved"])
    assert result.exit_code == 0, result.output
    assert "Classification target candidates" in result.output


def test_train_and_predict_commands(tmp_path, binary_frame):
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "predictions.csv"
    save_dataframe(binary_frame, data_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--target",
            "approved",
            "--model",
            "logistic",
            "--cv",
            "2",
            "--output",
            str(model_path),
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert model_path.exists()

    predict_result = runner.invoke(
        app,
        ["predict", str(model_path), str(data_path), "--output", str(output_path)],
    )
    assert predict_result.exit_code == 0, predict_result.output
    assert output_path.exists()


def test_train_and_predict_multiclass_commands(tmp_path, multiclass_frame):
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "predictions.csv"
    save_dataframe(multiclass_frame, data_path)

    train_result = runner.invoke(
        app,
        [
            "train",
            str(data_path),
            "--target",
            "risk",
            "--model",
            "logistic",
            "--cv",
            "2",
            "--output",
            str(model_path),
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert "Classes:" in train_result.output
    assert model_path.exists()

    predict_result = runner.invoke(
        app,
        ["predict", str(model_path), str(data_path), "--output", str(output_path)],
    )
    assert predict_result.exit_code == 0, predict_result.output
    assert output_path.exists()

    evaluate_result = runner.invoke(app, ["evaluate", str(model_path), str(data_path)])
    assert evaluate_result.exit_code == 0, evaluate_result.output
    assert "Classes:" in evaluate_result.output
