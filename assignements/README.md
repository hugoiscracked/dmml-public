# DMML Exercises

The exercises are designed as self-contained Jupyter notebooks. The notebook is
the source of truth: task text, scaffolding, visible self-checks, final analysis,
and optional challenge work should all live there.

## Environment

Use Python 3.10, 3.11, or 3.12. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Weeks 9 and 10 download datasets and pretrained models on first run. Students
who want GPU-enabled PyTorch should install the matching `torch` and
`torchvision` wheels from the official PyTorch instructions, then install the
remaining requirements.

## Shared Material

In `common/`:

- `dmml_benchmark.py` - helper module for the course-long benchmark table.
- `benchmark.ipynb` - the shared comparison notebook, re-run after every week.

In `../docs/`:

- `libraries.html` - the library reference: what each library is for, how it
  thinks, links to its real documentation, and the functions used in this
  course. Filterable by week, so students can see what a given week will ask
  them to import before they open the notebook.
- `ai-tutor.html` - a suggested prompt that puts an AI assistant into tutor mode
  for these assignments, plus what constructive and destructive use look like.
- `index.html` - the model-selection skill tree. `glossary.html` - the vocabulary.

## Benchmark Table Thread

There is **one** benchmark table for the whole semester.

Each weekly notebook ends with a single *Save to the Course Benchmark* cell that
appends its scores to `common/benchmark_results.csv` in long format:

- `week`
- `dataset`
- `task_type`
- `target`
- `model`
- `metric`
- `score`
- `split`
- `notes`

Students then re-run `common/benchmark.ipynb`, which pivots to wide format - one
table per dataset, metrics as rows, models as columns - and shows how the table
has grown. Because Wine appears in W03 and W05, and Digits in W04 and W08, the
same row accumulates models from different weeks, which is the whole point: a
course-long numeric record of which tools work well on which kinds of datasets.

See `model_comparison_schema.md` for the full contract.
