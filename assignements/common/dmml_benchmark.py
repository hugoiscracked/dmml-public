"""Course-long benchmark table for DMML.

Each week's notebook ends with a single `save_results(...)` call that appends
that week's model scores to one shared CSV file. The shared notebook
`common/benchmark.ipynb` reads the file back, pivots it to the wide view, and
shows the table growing across the semester.

The weekly notebooks therefore stay short: they produce numbers, this module
stores them, and one notebook does all the comparing.

Storage format is *long*, because long tables are the ones you can append to:

    week | dataset | task_type | target | model | metric | score | split | notes

Typical use at the end of a weekly notebook::

    import sys
    sys.path.append("../common")
    from dmml_benchmark import save_results

    save_results(
        results=all_results,          # dataframe with a "model" column + metrics
        week="W05",
        dataset="Wine",
        task_type="classification",
        target="wine_class",
        split="75_25_stratified_random_state_42",
    )

Re-running a week is safe: rows with the same
(week, dataset, task_type, target, split, model, metric) key are replaced, not
duplicated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "BENCHMARK_COLUMNS",
    "BENCHMARK_PATH",
    "KEY_COLUMNS",
    "NON_METRIC_COLUMNS",
    "clear_benchmark",
    "coverage",
    "load_benchmark",
    "pivot_wide",
    "repeated_measurements",
    "save_results",
    "to_long",
]

#: The shared CSV lives next to this module, so it is found from any week folder.
BENCHMARK_PATH = Path(__file__).resolve().with_name("benchmark_results.csv")

#: Column order of the long table.
BENCHMARK_COLUMNS = [
    "week",
    "dataset",
    "task_type",
    "target",
    "model",
    "metric",
    "score",
    "split",
    "notes",
]

#: Identity of a single measurement. Re-saving the same key overwrites it.
KEY_COLUMNS = ["week", "dataset", "task_type", "target", "split", "model", "metric"]

#: Numeric columns that describe cost or configuration, not model quality.
#: These are never recorded as metrics unless you ask for them explicitly.
NON_METRIC_COLUMNS = {
    "fit_time_sec",
    "predict_time_sec",
    "train_time_sec",
    "n_features",
    "n_params",
    "epochs",
    "k",
    "eps",
    "min_samples",
    "C",
    "gamma",
    "max_depth",
    "n_estimators",
}


def _metric_columns(results: pd.DataFrame, metrics: list[str] | None) -> list[str]:
    """Decide which columns of `results` count as metrics."""
    if metrics is not None:
        missing = [m for m in metrics if m not in results.columns]
        if missing:
            raise KeyError(f"metrics not found in results: {missing}")
        return list(metrics)

    numeric = results.select_dtypes(include="number").columns
    return [c for c in numeric if c not in NON_METRIC_COLUMNS]


def to_long(
    results: pd.DataFrame,
    *,
    week: str,
    dataset: str,
    task_type: str,
    target: str,
    split: str,
    metrics: list[str] | None = None,
    notes: str = "",
    model_column: str = "model",
) -> pd.DataFrame:
    """Reshape a wide results table into the long benchmark format.

    Parameters
    ----------
    results
        One row per model, with a model-name column plus numeric metric columns.
    week, dataset, task_type, target, split
        Context that makes two scores comparable. Only rows agreeing on
        task_type, target, split and metric may be compared numerically.
    metrics
        Explicit metric columns. By default every numeric column is used except
        the cost/configuration columns in `NON_METRIC_COLUMNS`.
    notes
        Free text attached to every row, e.g. "frozen backbone, 3 epochs".
    """
    if model_column not in results.columns:
        raise KeyError(f"results has no '{model_column}' column")

    metric_cols = _metric_columns(results, metrics)
    if not metric_cols:
        raise ValueError("no metric columns found in results")

    long = (
        results
        .melt(
            id_vars=[model_column],
            value_vars=metric_cols,
            var_name="metric",
            value_name="score",
        )
        .rename(columns={model_column: "model"})
    )
    long["week"] = week
    long["dataset"] = dataset
    long["task_type"] = task_type
    long["target"] = target
    long["split"] = split
    long["notes"] = notes
    return long[BENCHMARK_COLUMNS]


def load_benchmark(path: Path | str = BENCHMARK_PATH) -> pd.DataFrame:
    """Read the shared benchmark table. Returns an empty table if none exists."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=BENCHMARK_COLUMNS)
    table = pd.read_csv(path)
    table["notes"] = table["notes"].fillna("")
    return table


def save_results(
    results: pd.DataFrame,
    *,
    week: str,
    dataset: str,
    task_type: str,
    target: str,
    split: str,
    metrics: list[str] | None = None,
    notes: str = "",
    model_column: str = "model",
    path: Path | str = BENCHMARK_PATH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append this week's results to the shared benchmark CSV.

    Returns the rows that were written. Safe to re-run: rows sharing the same
    (week, dataset, task_type, target, split, model, metric) key are replaced.
    """
    new_rows = to_long(
        results,
        week=week,
        dataset=dataset,
        task_type=task_type,
        target=target,
        split=split,
        metrics=metrics,
        notes=notes,
        model_column=model_column,
    )

    path = Path(path)
    combined = pd.concat([load_benchmark(path), new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    combined = combined.sort_values(["week", "dataset", "metric", "model"]).reset_index(drop=True)
    combined.to_csv(path, index=False)

    if verbose:
        print(f"Saved {len(new_rows)} rows for {week} ({dataset}) to {path.name}")
        print(f"Benchmark now holds {len(combined)} rows across {combined['week'].nunique()} week(s).")
        print("Re-run ../common/benchmark.ipynb to see the updated comparison.")
    return new_rows


def pivot_wide(
    long: pd.DataFrame,
    task_type: str | None = None,
    index: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot the long table so each model becomes a column.

    Pass `task_type` to restrict the view to comparable rows; mixing
    forecasting, clustering and classification in one table is not meaningful.
    """
    if task_type is not None:
        long = long[long["task_type"] == task_type]
    if long.empty:
        return pd.DataFrame()

    index = index or ["dataset", "task_type", "target", "metric", "split"]
    wide = (
        long
        .pivot_table(index=index, columns="model", values="score", aggfunc="first")
        .reset_index()
    )
    wide.columns.name = None
    return wide


def repeated_measurements(long: pd.DataFrame) -> pd.DataFrame:
    """Find the same model measured in more than one week on the same setup.

    This happens on purpose: a dummy baseline recorded in Week 03 and again in
    Week 05 should give the identical number. If it does not, one of the two
    runs used a different split, seed, or preprocessing than it claims - worth
    knowing, because the wide view keeps only the first value.
    """
    if long.empty:
        return pd.DataFrame()

    keys = ["dataset", "task_type", "target", "split", "model", "metric"]
    grouped = long.groupby(keys, as_index=False).agg(
        n_weeks=("week", "nunique"),
        weeks=("week", lambda s: ", ".join(sorted(set(s)))),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    repeats = grouped[grouped["n_weeks"] > 1].copy()
    repeats["agrees"] = (repeats["max_score"] - repeats["min_score"]).abs() < 1e-9
    return repeats.reset_index(drop=True)


def coverage(long: pd.DataFrame, weeks: list[str] | None = None) -> pd.DataFrame:
    """Summarise what has been recorded so far, one row per week."""
    weeks = weeks or [f"W{i:02d}" for i in range(2, 11)]
    if long.empty:
        recorded = pd.DataFrame(columns=["week", "dataset", "task_type", "n_models", "n_rows"])
    else:
        recorded = (
            long
            .groupby(["week", "dataset", "task_type"], as_index=False)
            .agg(n_models=("model", "nunique"), n_rows=("score", "size"))
        )
    missing = sorted(set(weeks) - set(recorded["week"]))
    if missing:
        gaps = pd.DataFrame({
            "week": missing,
            "dataset": "-",
            "task_type": "not recorded yet",
            "n_models": 0,
            "n_rows": 0,
        })
        recorded = pd.concat([recorded, gaps], ignore_index=True)
    return recorded.sort_values("week").reset_index(drop=True)


def clear_benchmark(week: str | None = None, path: Path | str = BENCHMARK_PATH) -> None:
    """Delete every recorded row, or only the rows of one week."""
    path = Path(path)
    if week is None:
        path.unlink(missing_ok=True)
        print("Benchmark cleared.")
        return
    table = load_benchmark(path)
    kept = table[table["week"] != week]
    kept.to_csv(path, index=False)
    print(f"Removed {len(table) - len(kept)} rows for {week}.")
