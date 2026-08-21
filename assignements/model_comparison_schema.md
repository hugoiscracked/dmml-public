# DMML Benchmark Table

The comparison table is a cumulative numeric benchmark, not a skill-tree summary
and not a prose model report.

The idea is simple: as the course introduces more modelling tools, students add
new model results to one shared table. Over time, the table expands:

- new datasets add rows;
- new models add columns in the wide view;
- repeated metrics make strengths and weaknesses visible.

## Where It Lives

One table for the whole semester, not one per notebook:

- `common/dmml_benchmark.py` - the helper module (`save_results`, `load_benchmark`,
  `pivot_wide`, `repeated_measurements`, `coverage`, `clear_benchmark`).
- `common/benchmark_results.csv` - the shared long table, created on first save.
  Student-generated; gitignored.
- `common/benchmark.ipynb` - the notebook students re-run after every week to see
  the table grow.

Each weekly notebook ends with a single *Save to the Course Benchmark* cell. The
weekly notebooks produce numbers; the shared notebook does all the comparing.

## Storage Format: Long

Long format is what gets appended to:

- `week`
- `dataset`
- `task_type`
- `target`
- `model`
- `metric`
- `score`
- `split`
- `notes`

Example:

| week | dataset | task_type | target | model | metric | score | split |
|---|---|---|---|---|---|---:|---|
| W02 | AirPassengers | forecasting | passengers | seasonal_naive_12 | rmse | 42.1 | last_24_months |

Writing a week takes one call:

```python
import sys
sys.path.append("../common")
from dmml_benchmark import save_results

save_results(
    all_results,                 # one row per model, plus metric columns
    week="W05",
    dataset="Wine",
    task_type="classification",
    target="wine_class",
    split="75_25_stratified_random_state_42",
    metrics=["accuracy", "f1_macro"],
    notes="single decision tree vs bagging-style ensembles",
)
```

Rows are keyed by
(`week`, `dataset`, `task_type`, `target`, `split`, `model`, `metric`), so
re-running a weekly notebook replaces its rows instead of duplicating them.

Cost and configuration columns (`fit_time_sec`, `n_features`, `k`, `eps`, ...)
are never stored as metrics. Pass `metrics=[...]` to be explicit.

## Display Format: Wide

`benchmark.ipynb` pivots to wide format, one table per dataset:

| target | metric | dummy | knn_5 | rbf_svm_tuned | mlp_64_32 |
|---|---|---:|---:|---:|---:|
| digit | f1_macro | 0.019 | 0.964 | 0.980 | 0.966 |

This is where the thread pays off. Wine is scored in W03 and again in W05;
Digits in W04 and again in W08. Those weeks land on the same row, months apart,
so students can read straight across and see whether the newer, heavier method
actually won.

## Compatibility Rule

Only compare models numerically when the task, target, split, and metric are
compatible. Forecasting models belong with forecasting datasets; classifiers
belong with classification datasets; clustering needs its own metrics. The wide
view is built per dataset and per task type for exactly this reason.

## Repeated Measurements

A dummy baseline recorded in W03 and again in W05 on the same split must give the
identical number. `repeated_measurements()` finds these repeats and flags any
that disagree - a cheap, honest check that a split or seed did not silently
drift between weeks.
