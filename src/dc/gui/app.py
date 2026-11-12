from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dc.eda import inspect_dataframe, save_report
from dc.evaluate import compute_metrics, compute_multiclass_metrics, format_metrics, roc_points
from dc.io import find_classification_targets, load_dataframe, save_dataframe, unique_values
from dc.models import MODEL_LABELS
from dc.predict import predict_frame, predict_proba_matrix, predict_probabilities, predict_record
from dc.train import (
    TrainConfig,
    TrainResult,
    encode_target_values,
    load_bundle,
    save_bundle,
    save_result,
    train_model,
)

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

FILE_TYPES = [
    ("Data files", "*.csv *.xlsx *.json"),
    ("CSV", "*.csv"),
    ("Excel", "*.xlsx"),
    ("JSON", "*.json"),
]


class ClassifierApp:
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("Data Classifier")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 680)
        self.frame: pd.DataFrame | None = None
        self.bundle: dict[str, Any] | None = None
        self.result: TrainResult | None = None
        self.eval_data: dict[str, Any] | None = None
        self.manual_widgets: dict[str, Any] = {}
        self.batch_frame: pd.DataFrame | None = None
        self.batch_result: pd.DataFrame | None = None
        self.charts: dict[str, tuple[Any, Any]] = {}
        self.worker_queue: queue.Queue = queue.Queue()
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.status_var = tk.StringVar(value="Load a data file to begin.")
        self.path_var = tk.StringVar(value="No file loaded")
        self.threshold_label_var = tk.StringVar(value="0.50")
        self.threshold_widgets: list[Any] = []
        self.progress_var = tk.DoubleVar(value=0.0)
        self.positive_var = tk.StringVar()
        self.target_var = tk.StringVar()
        self.scoring_var = tk.StringVar(value="f1_macro")
        self.test_size_var = tk.StringVar(value="0.2")
        self.cv_var = tk.StringVar(value="5")
        self.seed_var = tk.StringVar(value="42")
        self.balanced_var = tk.BooleanVar(value=False)
        self.search_var = tk.BooleanVar(value=False)
        self.model_vars = {name: tk.BooleanVar(value=True) for name in MODEL_LABELS}
        self._build()
        self.root.after(100, self._poll_queue)

    def _build(self) -> None:
        header = ttk.Frame(self.root, padding=(16, 12, 16, 4))
        header.pack(fill="x")
        ttk.Label(header, text="Data Classifier", font=("TkDefaultFont", 16, "bold")).pack(
            side="left"
        )
        ttk.Label(
            header,
            text="Train and use binary or multiclass classifiers on CSV, Excel, or JSON data",
        ).pack(side="left", padx=12)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=8)
        self.tab_data = ttk.Frame(self.notebook, padding=10)
        self.tab_explore = ttk.Frame(self.notebook, padding=10)
        self.tab_train = ttk.Frame(self.notebook, padding=10)
        self.tab_evaluate = ttk.Frame(self.notebook, padding=10)
        self.tab_predict = ttk.Frame(self.notebook, padding=10)
        self.tab_model = ttk.Frame(self.notebook, padding=10)
        for tab, title in (
            (self.tab_data, "1. Data"),
            (self.tab_explore, "2. Explore"),
            (self.tab_train, "3. Train"),
            (self.tab_evaluate, "4. Evaluate"),
            (self.tab_predict, "5. Predict"),
            (self.tab_model, "6. Model"),
        ):
            self.notebook.add(tab, text=title)
        self._build_data_tab()
        self._build_explore_tab()
        self._build_train_tab()
        self._build_evaluate_tab()
        self._build_predict_tab()
        self._build_model_tab()

        status = ttk.Frame(self.root, padding=(12, 4))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

    def _build_data_tab(self) -> None:
        tab = self.tab_data
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Button(top, text="Open data file...", command=self.load_file).pack(side="left")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=12)

        options = ttk.LabelFrame(tab, text="Classification setup", padding=10)
        options.grid(row=1, column=0, sticky="ew", pady=8)
        ttk.Label(options, text="Target column").grid(row=0, column=0, sticky="w")
        self.target_combo = ttk.Combobox(
            options, textvariable=self.target_var, state="readonly", width=28
        )
        self.target_combo.grid(row=0, column=1, padx=8, sticky="w")
        self.target_combo.bind("<<ComboboxSelected>>", self._on_target_change)
        ttk.Label(options, text="Positive class").grid(row=0, column=2, padx=(20, 0))
        self.positive_combo = ttk.Combobox(
            options, textvariable=self.positive_var, state="readonly", width=24
        )
        self.positive_combo.grid(row=0, column=3, padx=8, sticky="w")
        ttk.Label(options, text="Exclude columns (optional)").grid(
            row=1, column=0, sticky="nw", pady=(10, 0)
        )
        exclude_frame = ttk.Frame(options)
        exclude_frame.grid(row=1, column=1, rowspan=2, columnspan=3, sticky="w", pady=(10, 0))
        self.exclude_list = tk.Listbox(
            exclude_frame, selectmode="extended", height=4, exportselection=False
        )
        self.exclude_list.pack(side="left", fill="y")
        scroll = ttk.Scrollbar(exclude_frame, orient="vertical", command=self.exclude_list.yview)
        scroll.pack(side="left", fill="y")
        self.exclude_list.configure(yscrollcommand=scroll.set)
        ttk.Label(
            options,
            text="Hold Ctrl to select several columns. ID-like columns are not dropped automatically.",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        preview = ttk.LabelFrame(tab, text="Preview", padding=6)
        preview.grid(row=2, column=0, sticky="nsew")
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.preview_tree = ttk.Treeview(preview, show="headings")
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(preview, orient="vertical", command=self.preview_tree.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(preview, orient="horizontal", command=self.preview_tree.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        self.preview_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

    def _build_explore_tab(self) -> None:
        tab = self.tab_explore
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(controls, text="Run inspection", command=self.run_inspection).pack(side="left")
        ttk.Button(controls, text="Save report...", command=self.save_inspection).pack(
            side="left", padx=8
        )

        text_frame = ttk.LabelFrame(tab, text="Summary", padding=6)
        text_frame.grid(row=1, column=0, sticky="nsew", pady=8, padx=(0, 6))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.inspect_text = tk.Text(text_frame, wrap="word", height=20)
        self.inspect_text.grid(row=0, column=0, sticky="nsew")
        text_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.inspect_text.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.inspect_text.configure(yscrollcommand=text_scroll.set)

        chart_frame = ttk.LabelFrame(tab, text="Class balance", padding=6)
        chart_frame.grid(row=1, column=1, sticky="nsew", pady=8, padx=(6, 0))
        self._make_chart(chart_frame, "balance")

    def _build_train_tab(self) -> None:
        tab = self.tab_train
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)

        config = ttk.LabelFrame(tab, text="Training configuration", padding=10)
        config.grid(row=0, column=0, sticky="ew")
        fields = (
            ("Test size", self.test_size_var, 8),
            ("CV folds", self.cv_var, 8),
            ("Random seed", self.seed_var, 8),
        )
        for index, (label, variable, width) in enumerate(fields):
            ttk.Label(config, text=label).grid(row=0, column=index * 2, sticky="w", padx=(0, 6))
            ttk.Entry(config, textvariable=variable, width=width).grid(
                row=0, column=index * 2 + 1, sticky="w", padx=(0, 16)
            )
        ttk.Label(config, text="Selection metric").grid(row=0, column=6, sticky="w", padx=(0, 6))
        ttk.Combobox(
            config,
            textvariable=self.scoring_var,
            state="readonly",
            width=18,
            values=["f1_macro", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall"],
        ).grid(row=0, column=7, sticky="w")
        ttk.Checkbutton(config, text="Balanced class weights", variable=self.balanced_var).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0)
        )
        ttk.Checkbutton(
            config, text="Tune hyperparameters (random search)", variable=self.search_var
        ).grid(row=1, column=2, columnspan=3, sticky="w", pady=(10, 0))

        models = ttk.LabelFrame(tab, text="Candidate models", padding=10)
        models.grid(row=1, column=0, sticky="ew", pady=8)
        for index, (name, label) in enumerate(MODEL_LABELS.items()):
            ttk.Checkbutton(models, text=label, variable=self.model_vars[name]).grid(
                row=0, column=index, sticky="w", padx=(0, 20)
            )

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, sticky="ew")
        self.train_button = ttk.Button(actions, text="Train models", command=self.start_training)
        self.train_button.pack(side="left")
        self.progress = ttk.Progressbar(actions, variable=self.progress_var, maximum=100, length=260)
        self.progress.pack(side="left", padx=12)

        results = ttk.Frame(tab)
        results.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        results.columnconfigure(0, weight=1)
        results.columnconfigure(1, weight=1)
        results.rowconfigure(0, weight=1)

        board_frame = ttk.LabelFrame(results, text="Leaderboard", padding=6)
        board_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        board_frame.rowconfigure(0, weight=1)
        board_frame.columnconfigure(0, weight=1)
        self.leaderboard_tree = ttk.Treeview(
            board_frame, columns=("model", "mean", "std", "status"), show="headings", height=8
        )
        for column, heading, width in (
            ("model", "Model", 170),
            ("mean", "Mean CV", 90),
            ("std", "Std", 80),
            ("status", "Status", 180),
        ):
            self.leaderboard_tree.heading(column, text=heading)
            self.leaderboard_tree.column(column, width=width)
        self.leaderboard_tree.grid(row=0, column=0, sticky="nsew")

        metrics_frame = ttk.LabelFrame(results, text="Held-out test metrics", padding=6)
        metrics_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.columnconfigure(0, weight=1)
        self.train_metrics_text = tk.Text(metrics_frame, wrap="word", height=12)
        self.train_metrics_text.grid(row=0, column=0, sticky="nsew")

    def _build_evaluate_tab(self) -> None:
        tab = self.tab_evaluate
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(2, weight=1)
        tab.rowconfigure(1, weight=1)

        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, columnspan=3, sticky="ew")
        ttk.Button(
            controls, text="Evaluate model on loaded data", command=self.evaluate_current
        ).pack(side="left")
        threshold_label = ttk.Label(controls, text="Threshold")
        threshold_label.pack(side="left", padx=(20, 6))
        threshold_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            command=self._on_threshold,
            length=260,
        )
        threshold_scale.pack(side="left")
        threshold_value = ttk.Label(controls, textvariable=self.threshold_label_var)
        threshold_value.pack(side="left", padx=8)
        self.threshold_widgets.extend([threshold_label, threshold_scale, threshold_value])

        metrics_frame = ttk.LabelFrame(tab, text="Metrics", padding=6)
        metrics_frame.grid(row=1, column=0, sticky="nsew", pady=8, padx=(0, 6))
        metrics_frame.rowconfigure(0, weight=1)
        metrics_frame.columnconfigure(0, weight=1)
        self.eval_text = tk.Text(metrics_frame, wrap="word", width=34)
        self.eval_text.grid(row=0, column=0, sticky="nsew")

        cm_frame = ttk.LabelFrame(tab, text="Confusion matrix", padding=6)
        cm_frame.grid(row=1, column=1, sticky="nsew", pady=8, padx=6)
        self._make_chart(cm_frame, "confusion")

        roc_frame = ttk.LabelFrame(tab, text="ROC curve", padding=6)
        roc_frame.grid(row=1, column=2, sticky="nsew", pady=8, padx=(6, 0))
        self._make_chart(roc_frame, "roc")

    def _build_predict_tab(self) -> None:
        tab = self.tab_predict
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew")
        predict_threshold_label = ttk.Label(controls, text="Decision threshold")
        predict_threshold_label.pack(side="left")
        predict_threshold_scale = ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.threshold_var,
            command=self._on_threshold,
            length=220,
        )
        predict_threshold_scale.pack(side="left", padx=8)
        predict_threshold_value = ttk.Label(controls, textvariable=self.threshold_label_var)
        predict_threshold_value.pack(side="left")
        self.threshold_widgets.extend(
            [predict_threshold_label, predict_threshold_scale, predict_threshold_value]
        )

        manual = ttk.LabelFrame(tab, text="Predict a single case", padding=6)
        manual.grid(row=1, column=0, sticky="nsew", pady=8, padx=(0, 6))
        manual.rowconfigure(0, weight=1)
        manual.columnconfigure(0, weight=1)
        self.manual_canvas = tk.Canvas(manual, highlightthickness=0)
        self.manual_canvas.grid(row=0, column=0, sticky="nsew")
        manual_scroll = ttk.Scrollbar(manual, orient="vertical", command=self.manual_canvas.yview)
        manual_scroll.grid(row=0, column=1, sticky="ns")
        self.manual_canvas.configure(yscrollcommand=manual_scroll.set)
        self.manual_form = ttk.Frame(self.manual_canvas)
        self.manual_window = self.manual_canvas.create_window(
            (0, 0), window=self.manual_form, anchor="nw"
        )
        self.manual_form.bind(
            "<Configure>",
            lambda event: self.manual_canvas.configure(scrollregion=self.manual_canvas.bbox("all")),
        )
        self.manual_canvas.bind(
            "<Configure>",
            lambda event: self.manual_canvas.itemconfigure(self.manual_window, width=event.width),
        )
        self.manual_button = ttk.Button(manual, text="Predict", command=self.predict_manual)
        self.manual_button.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.manual_result = ttk.Label(
            manual, text="Train or load a model first.", font=("TkDefaultFont", 11, "bold")
        )
        self.manual_result.grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        batch = ttk.LabelFrame(tab, text="Predict a batch file", padding=6)
        batch.grid(row=1, column=1, sticky="nsew", pady=8, padx=(6, 0))
        batch.columnconfigure(0, weight=1)
        batch.rowconfigure(1, weight=1)
        batch_controls = ttk.Frame(batch)
        batch_controls.grid(row=0, column=0, sticky="ew")
        ttk.Button(batch_controls, text="Open file...", command=self.load_batch_file).pack(
            side="left"
        )
        ttk.Button(batch_controls, text="Predict", command=self.predict_batch).pack(
            side="left", padx=6
        )
        ttk.Button(batch_controls, text="Save results...", command=self.save_batch).pack(
            side="left"
        )
        self.batch_tree = ttk.Treeview(batch, show="headings")
        self.batch_tree.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        batch_scroll = ttk.Scrollbar(batch, orient="vertical", command=self.batch_tree.yview)
        batch_scroll.grid(row=1, column=1, sticky="ns", pady=(8, 0))
        self.batch_tree.configure(yscrollcommand=batch_scroll.set)

    def _build_model_tab(self) -> None:
        tab = self.tab_model
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        controls = ttk.Frame(tab)
        controls.grid(row=0, column=0, sticky="ew")
        ttk.Button(controls, text="Save trained model...", command=self.save_model).pack(
            side="left"
        )
        ttk.Button(controls, text="Load model...", command=self.load_model).pack(
            side="left", padx=8
        )

        info_frame = ttk.LabelFrame(tab, text="Model metadata", padding=6)
        info_frame.grid(row=1, column=0, sticky="nsew", pady=8)
        info_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        self.metadata_text = tk.Text(info_frame, wrap="word")
        self.metadata_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(info_frame, orient="vertical", command=self.metadata_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.metadata_text.configure(yscrollcommand=scroll.set)

    def _make_chart(self, parent, name: str) -> None:
        if not MATPLOTLIB_AVAILABLE:
            ttk.Label(parent, text="matplotlib is not available.").pack()
            return
        figure = Figure(figsize=(4.2, 3.2), dpi=100)
        axes = figure.add_subplot(111)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.charts[name] = (figure, axes, canvas)

    def _chart(self, name: str):
        return self.charts.get(name)

    def load_file(self) -> None:
        path = filedialog.askopenfilename(title="Open data file", filetypes=FILE_TYPES)
        if not path:
            return
        try:
            frame = load_dataframe(path)
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self.frame = frame
        self.path_var.set(f"{Path(path).name}  ({len(frame)} rows x {frame.shape[1]} columns)")
        columns = [str(column) for column in frame.columns]
        self.target_combo["values"] = columns
        self.exclude_list.delete(0, "end")
        for column in columns:
            self.exclude_list.insert("end", column)
        self._populate_tree(self.preview_tree, frame)
        candidates = find_classification_targets(frame)
        if candidates:
            self.target_var.set(candidates[-1])
            self._on_target_change()
        elif columns:
            self.target_var.set(columns[-1])
            self._on_target_change()
        self.status_var.set(f"Loaded {path}")
        self.notebook.select(self.tab_data)

    def _on_target_change(self, _event=None) -> None:
        if self.frame is None:
            return
        target = self.target_var.get()
        if target not in self.frame.columns:
            return
        values = unique_values(self.frame, target)
        labels = [str(value) for value in values]
        self.positive_combo["values"] = labels
        if len(labels) == 2:
            self.positive_combo.configure(state="readonly")
            self.positive_var.set(labels[-1])
        else:
            self.positive_combo.configure(state="disabled")
            self.positive_var.set("")
        for index in range(self.exclude_list.size()):
            if self.exclude_list.get(index) == target:
                self.exclude_list.selection_clear(index)

    def run_inspection(self) -> None:
        if self.frame is None:
            messagebox.showinfo("No data", "Load a data file first.")
            return
        target = self.target_var.get() or None
        info = inspect_dataframe(self.frame, target)
        self.last_inspection = info
        self.inspect_text.delete("1.0", "end")
        self.inspect_text.insert("1.0", self._format_inspection(info))
        self._draw_balance(info)
        self.notebook.select(self.tab_explore)

    def _format_inspection(self, info: dict[str, Any]) -> str:
        lines = [
            f"Rows: {info['rows']}",
            f"Columns: {info['columns']}",
            f"Duplicate rows: {info['duplicate_rows']}",
            f"Missing cells: {info['total_missing']}",
            f"Classification target candidates: "
            f"{', '.join(info.get('classification_target_candidates', []))}",
            "",
        ]
        if "target" in info:
            lines.append(f"Target: {info['target']}")
            for item in info["class_balance"]["classes"]:
                lines.append(f"  {item['label']}: {item['count']} ({item['pct']}%)")
            if info["class_balance"]["imbalanced"]:
                lines.append("  Warning: target is imbalanced.")
            lines.append("")
        lines.append("Columns:")
        for column in info["schema"]:
            marker = " (target)" if column["is_target"] else ""
            lines.append(
                f"  {column['name']}{marker}: {column['dtype']}, {column['kind']}, "
                f"missing {column['missing_pct']}%, unique {column['unique']}"
            )
        return "\n".join(lines)

    def _draw_balance(self, info: dict[str, Any]) -> None:
        chart = self._chart("balance")
        if chart is None:
            return
        figure, axes, canvas = chart
        axes.clear()
        if "target" in info:
            labels = [str(item["label"]) for item in info["class_balance"]["classes"]]
            counts = [item["count"] for item in info["class_balance"]["classes"]]
            axes.bar(labels, counts, color=["#4c72b0", "#dd8452"][: len(labels)])
            axes.set_title(f"Target: {info['target']}")
            axes.set_ylabel("Rows")
        else:
            axes.text(0.5, 0.5, "Choose a target column", ha="center", va="center")
            axes.set_axis_off()
        figure.tight_layout()
        canvas.draw_idle()

    def save_inspection(self) -> None:
        if not hasattr(self, "last_inspection"):
            messagebox.showinfo("No report", "Run an inspection first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save inspection report", defaultextension=".md", initialfile="inspection.md"
        )
        if not path:
            return
        markdown_path, json_path = save_report(self.last_inspection, path)
        self.status_var.set(f"Report saved to {markdown_path} and {json_path}")

    def _selected_models(self) -> list[str]:
        return [name for name, variable in self.model_vars.items() if variable.get()]

    def start_training(self) -> None:
        if self.frame is None:
            messagebox.showinfo("No data", "Load a data file first.")
            return
        target = self.target_var.get()
        if not target:
            messagebox.showinfo("No target", "Choose a target column.")
            return
        models = self._selected_models()
        if not models:
            messagebox.showinfo("No models", "Select at least one candidate model.")
            return
        try:
            test_size = float(self.test_size_var.get())
            cv_folds = int(self.cv_var.get())
            seed = int(self.seed_var.get())
            if not 0.05 <= test_size <= 0.5:
                raise ValueError("Test size must be between 0.05 and 0.5.")
            if not 2 <= cv_folds <= 20:
                raise ValueError("CV folds must be between 2 and 20.")
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        exclude = [self.exclude_list.get(index) for index in self.exclude_list.curselection()]
        target_values = unique_values(self.frame, target)
        positive_label = self.positive_var.get() or None if len(target_values) == 2 else None
        config = TrainConfig(
            target=target,
            positive_label=positive_label,
            exclude=exclude,
            test_size=test_size,
            cv_folds=cv_folds,
            seed=seed,
            models=models,
            class_weight="balanced" if self.balanced_var.get() else None,
            search=self.search_var.get(),
            scoring=self.scoring_var.get(),
        )
        self.train_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self.status_var.set("Training...")
        frame = self.frame

        def worker() -> None:
            try:
                result = train_model(
                    frame,
                    config,
                    progress=lambda message, fraction: self.worker_queue.put(
                        ("progress", message, fraction)
                    ),
                )
            except Exception as exc:
                self.worker_queue.put(("error", exc))
                return
            self.worker_queue.put(("result", result))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                message = self.worker_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self._set_progress(message[1], message[2])
                elif kind == "result":
                    self._training_done(message[1])
                elif kind == "error":
                    self._training_failed(message[1])
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _set_progress(self, message: str, fraction: float) -> None:
        self.progress_var.set(max(0.0, min(1.0, fraction)) * 100)
        self.status_var.set(message)

    def _training_failed(self, exc: Exception) -> None:
        self.train_button.configure(state="normal")
        self.progress_var.set(0.0)
        self.status_var.set("Training failed.")
        messagebox.showerror("Training failed", str(exc))

    def _training_done(self, result: TrainResult) -> None:
        self.train_button.configure(state="normal")
        self.progress_var.set(100)
        self.result = result
        self.bundle = {"pipeline": result.pipeline, "metadata": result.metadata}
        self.threshold_var.set(float(result.metadata.get("threshold", 0.5)))
        self._on_threshold()
        for item in self.leaderboard_tree.get_children():
            self.leaderboard_tree.delete(item)
        for row in result.leaderboard:
            self.leaderboard_tree.insert(
                "",
                "end",
                values=(
                    row["label"],
                    "n/a" if row["score"] is None else f"{row['score']:.4f}",
                    "n/a" if row["std"] is None else f"{row['std']:.4f}",
                    row["error"] or "ok",
                ),
            )
        self.train_metrics_text.delete("1.0", "end")
        positive = result.metadata.get("positive_label")
        self.train_metrics_text.insert(
            "1.0",
            format_metrics(
                result.test_metrics,
                positive_label=str(positive) if positive is not None else "positive",
            )
            + "\n\n"
            + "\n".join(f"Warning: {warning}" for warning in result.warnings),
        )
        self.eval_data = {
            "y": result.y_test,
            "probabilities": result.y_proba,
            "data": result.X_test,
            "task_type": result.metadata.get("task_type", "binary"),
            "classes": result.metadata.get("classes") or [],
        }
        self._set_binary_controls(result.metadata.get("task_type", "binary") == "binary")
        self._refresh_evaluation()
        self._build_manual_form()
        self._show_metadata(result.metadata)
        self.status_var.set(
            f"Training complete. Best model: "
            f"{MODEL_LABELS.get(result.model_name, result.model_name)}"
        )
        self.notebook.select(self.tab_evaluate)

    def evaluate_current(self) -> None:
        if self.bundle is None:
            messagebox.showinfo("No model", "Train or load a model first.")
            return
        if self.frame is None:
            messagebox.showinfo("No data", "Load a labeled data file first.")
            return
        metadata = self.bundle.get("metadata", {})
        target = metadata.get("target")
        if not target or target not in self.frame.columns:
            messagebox.showinfo("No target", "The loaded data has no usable target column.")
            return
        task_type = metadata.get("task_type") or (
            "binary" if metadata.get("positive_label") is not None else "multiclass"
        )
        clean = self.frame.dropna(subset=[target])
        try:
            if task_type == "binary":
                positive = metadata.get("positive_label", 1)
                y = clean[target].map(
                    lambda value: 1 if (value == positive or str(value) == str(positive)) else 0
                ).to_numpy()
                if len(np.unique(y)) < 2:
                    messagebox.showinfo(
                        "Not enough classes", "The loaded data must contain both classes."
                    )
                    return
                probabilities = predict_probabilities(self.bundle, clean)
            else:
                classes = metadata.get("classes") or []
                if not classes:
                    messagebox.showinfo("No classes", "The model metadata has no class labels.")
                    return
                y = encode_target_values(clean[target], classes)
                probabilities = predict_proba_matrix(self.bundle, clean)
        except Exception as exc:
            messagebox.showerror("Evaluation failed", str(exc))
            return
        self.eval_data = {
            "y": y,
            "probabilities": probabilities,
            "data": clean,
            "task_type": task_type,
            "classes": metadata.get("classes") or [],
        }
        self._set_binary_controls(task_type == "binary")
        self._refresh_evaluation()
        self.notebook.select(self.tab_evaluate)

    def _set_binary_controls(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in self.threshold_widgets:
            try:
                widget.configure(state=state)
            except Exception:
                pass

    def _on_threshold(self, _value=None) -> None:
        value = float(self.threshold_var.get())
        self.threshold_label_var.set(f"{value:.2f}")
        if self.eval_data is not None:
            self._refresh_evaluation()

    def _refresh_evaluation(self) -> None:
        if self.eval_data is None:
            return
        y = self.eval_data["y"]
        probabilities = self.eval_data["probabilities"]
        metadata = (self.bundle or {}).get("metadata", {})
        task_type = self.eval_data.get("task_type") or metadata.get("task_type", "binary")
        if task_type == "binary":
            threshold = float(self.threshold_var.get())
            metrics = compute_metrics(y, probabilities, threshold)
            positive = metadata.get("positive_label", 1)
            negative = metadata.get("negative_label", 0)
            self.eval_text.delete("1.0", "end")
            self.eval_text.insert("1.0", format_metrics(metrics, positive_label=str(positive)))
            self._draw_confusion(metrics, negative, positive)
            self._draw_roc(y, probabilities)
        else:
            classes = self.eval_data.get("classes") or metadata.get("classes") or []
            metrics = compute_multiclass_metrics(y, probabilities, classes)
            self.eval_text.delete("1.0", "end")
            self.eval_text.insert("1.0", format_metrics(metrics))
            self._draw_confusion(metrics, None, None)
            self._draw_roc(y, probabilities)

    def _draw_confusion(self, metrics: dict[str, Any], negative: Any, positive: Any) -> None:
        chart = self._chart("confusion")
        if chart is None:
            return
        figure, axes, canvas = chart
        matrix = np.array(metrics["confusion_matrix"])
        axes.clear()
        axes.imshow(matrix, cmap="Blues")
        size = matrix.shape[0]
        for row in range(size):
            for column in range(size):
                axes.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                    fontsize=12,
                )
        if metrics.get("task_type") == "multiclass":
            labels = [str(label) for label in metrics.get("classes", [])]
            axes.set_xticks(range(size), [f"Pred {label}" for label in labels])
            axes.set_yticks(range(size), [f"Actual {label}" for label in labels])
            axes.set_title("Multiclass confusion matrix")
        else:
            axes.set_xticks([0, 1], [f"Pred {negative}", f"Pred {positive}"])
            axes.set_yticks([0, 1], [f"Actual {negative}", f"Actual {positive}"])
            axes.set_title(f"Threshold {metrics['threshold']}")
        figure.tight_layout()
        canvas.draw_idle()

    def _draw_roc(self, y: np.ndarray, probabilities: np.ndarray) -> None:
        chart = self._chart("roc")
        if chart is None:
            return
        figure, axes, canvas = chart
        axes.clear()
        if np.asarray(probabilities).ndim > 1:
            axes.text(
                0.5,
                0.5,
                "ROC curve is available for binary models only",
                ha="center",
                va="center",
            )
            axes.set_axis_off()
        else:
            points = roc_points(y, probabilities)
            if points is None:
                axes.text(0.5, 0.5, "ROC needs both classes", ha="center", va="center")
                axes.set_axis_off()
            else:
                fpr, tpr = points
                axes.plot(fpr, tpr, color="#4c72b0", label="ROC")
                axes.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
                axes.set_xlabel("False positive rate")
                axes.set_ylabel("True positive rate")
                axes.set_title("ROC curve")
                axes.legend(loc="lower right")
        figure.tight_layout()
        canvas.draw_idle()

    def _build_manual_form(self) -> None:
        for child in self.manual_form.winfo_children():
            child.destroy()
        self.manual_widgets = {}
        if self.bundle is None:
            return
        features = self.bundle.get("metadata", {}).get("feature_columns", [])
        for index, feature in enumerate(features):
            ttk.Label(self.manual_form, text=str(feature)).grid(
                row=index, column=0, sticky="w", padx=(4, 12), pady=3
            )
            values = self._feature_values(feature)
            if values is not None:
                widget = ttk.Combobox(
                    self.manual_form, values=values, width=28, state="normal"
                )
            else:
                widget = ttk.Entry(self.manual_form, width=30)
            widget.grid(row=index, column=1, sticky="w", pady=3)
            self.manual_widgets[str(feature)] = widget
        self.manual_form.update_idletasks()
        self.manual_canvas.configure(scrollregion=self.manual_canvas.bbox("all"))

    def _feature_values(self, feature: str) -> list[str] | None:
        if self.frame is None or feature not in self.frame.columns:
            return None
        series = self.frame[feature].dropna()
        if len(series) == 0:
            return None
        unique = series.unique()
        if len(unique) > 30:
            return None
        try:
            unique = sorted(unique)
        except TypeError:
            pass
        return [str(value) for value in unique]

    def predict_manual(self) -> None:
        if self.bundle is None:
            messagebox.showinfo("No model", "Train or load a model first.")
            return
        if not self.manual_widgets:
            messagebox.showinfo("No form", "The model has no feature columns.")
            return
        record = {feature: widget.get() for feature, widget in self.manual_widgets.items()}
        try:
            result = predict_record(
                self.bundle, record, threshold=float(self.threshold_var.get())
            )
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return
        if result.get("task_type") == "multiclass":
            probabilities = "    ".join(
                f"{label}: {probability:.3f}"
                for label, probability in result.get("probabilities", {}).items()
            )
            self.manual_result.configure(
                text=f"Prediction: {result['prediction']}    {probabilities}"
            )
        else:
            self.manual_result.configure(
                text=(
                    f"Prediction: {result['prediction']}    "
                    f"probability of {result['positive_label']}: {result['probability']:.3f}    "
                    f"threshold: {result['threshold']:.2f}"
                )
            )

    def load_batch_file(self) -> None:
        path = filedialog.askopenfilename(title="Open batch file", filetypes=FILE_TYPES)
        if not path:
            return
        try:
            self.batch_frame = load_dataframe(path)
        except Exception as exc:
            messagebox.showerror("Could not load file", str(exc))
            return
        self._populate_tree(self.batch_tree, self.batch_frame)
        self.status_var.set(f"Batch file loaded: {path}")

    def predict_batch(self) -> None:
        if self.bundle is None:
            messagebox.showinfo("No model", "Train or load a model first.")
            return
        if self.batch_frame is None:
            messagebox.showinfo("No data", "Open a batch file first.")
            return
        try:
            self.batch_result = predict_frame(
                self.bundle, self.batch_frame, threshold=float(self.threshold_var.get())
            )
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return
        self._populate_tree(self.batch_tree, self.batch_result)
        self.status_var.set(f"Predicted {len(self.batch_result)} rows.")

    def save_batch(self) -> None:
        if self.batch_result is None:
            messagebox.showinfo("No results", "Run batch prediction first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save predictions",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx"), ("JSON", "*.json")],
        )
        if not path:
            return
        try:
            saved = save_dataframe(self.batch_result, path)
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self.status_var.set(f"Predictions saved to {saved}")

    def save_model(self) -> None:
        if self.result is None and self.bundle is None:
            messagebox.showinfo("No model", "Train or load a model first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".joblib",
            initialfile="model.joblib",
            filetypes=[("Joblib model", "*.joblib")],
        )
        if not path:
            return
        try:
            if self.result is not None:
                saved = save_result(self.result, path)
            else:
                saved = save_bundle(
                    self.bundle["pipeline"], self.bundle.get("metadata", {}), path
                )
        except Exception as exc:
            messagebox.showerror("Could not save", str(exc))
            return
        self.status_var.set(f"Model saved to {saved}")

    def load_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Load model", filetypes=[("Joblib model", "*.joblib")]
        )
        if not path:
            return
        try:
            bundle = load_bundle(path)
        except Exception as exc:
            messagebox.showerror("Could not load model", str(exc))
            return
        self.bundle = bundle
        self.result = None
        self.eval_data = None
        metadata = bundle.get("metadata", {})
        self.threshold_var.set(float(metadata.get("threshold", 0.5)))
        self._set_binary_controls(
            (metadata.get("task_type") or "binary") == "binary"
        )
        self._on_threshold()
        self._build_manual_form()
        self._show_metadata(bundle.get("metadata", {}))
        self.status_var.set(f"Model loaded: {path}")
        self.notebook.select(self.tab_predict)

    def _show_metadata(self, metadata: dict[str, Any]) -> None:
        lines = []
        for key in (
            "created_at",
            "target",
            "task_type",
            "classes",
            "class_count",
            "positive_label",
            "negative_label",
            "model_name",
            "feature_columns",
            "cv_folds",
            "scoring",
            "class_weight",
            "seed",
            "rows",
            "test_size",
            "test_metrics",
            "leaderboard",
            "warnings",
            "versions",
        ):
            if key in metadata:
                lines.append(f"{key}: {metadata[key]}")
        self.metadata_text.delete("1.0", "end")
        self.metadata_text.insert("1.0", "\n".join(lines))

    def _populate_tree(self, tree, frame: pd.DataFrame, limit: int = 200) -> None:
        tree.delete(*tree.get_children())
        columns = [str(column) for column in frame.columns]
        tree["columns"] = columns
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=120, stretch=True)
        for _, row in frame.head(limit).iterrows():
            values = []
            for column in frame.columns:
                value = row[column]
                if pd.isna(value):
                    values.append("")
                else:
                    values.append(str(value))
            tree.insert("", "end", values=values)



def main() -> None:
    if not TK_AVAILABLE:
        raise SystemExit(
            "Tkinter is not installed. Install it with: sudo apt install python3-tk"
        )
    root = tk.Tk()
    ClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
