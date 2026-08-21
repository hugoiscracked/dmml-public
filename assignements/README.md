# DMML Exercises

The exercises are designed as self-contained Jupyter notebooks. The notebook is
the source of truth: task text, scaffolding, visible self-checks, final analysis,
and optional challenge work should all live there.

## Environment

Use Python 3.11, 3.12, or 3.13 - the pinned versions of numpy, pandas and
scikit-learn all require 3.11 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Weeks 9 and 10 download datasets and pretrained models on first run. Students
who want GPU-enabled PyTorch should install the matching `torch` and
`torchvision` wheels from the official PyTorch instructions, then install the
remaining requirements.

## Working On An Assignment, And Getting Updates

**Work in a copy, never in `template.ipynb`.** Before you start a week, duplicate
its template:

```bash
cp assignements/w03_regression_classification/template.ipynb \
   assignements/w03_regression_classification/submission.ipynb
```

Do your work in `submission.ipynb` and submit that file. `submission.ipynb` is
gitignored, so it is yours alone and no course update can ever touch it.

This matters because the templates are corrected during the semester. If your
work lives in `template.ipynb`, every fix collides with it, and merging two
versions of a Jupyter notebook by hand is genuinely unpleasant.

### Pulling a correction

With your work in `submission.ipynb`, taking an update is one command:

```bash
git pull
```

If you did already edit a `template.ipynb`, rescue it first, then take the
update cleanly:

```bash
cp assignements/w01_eda/template.ipynb assignements/w01_eda/submission.ipynb
git checkout -- assignements/w01_eda/template.ipynb
git pull
```

Your work is now in `submission.ipynb`, and `template.ipynb` is the current
course version again.

### What a correction does and does not change

The grader does **not** run the notebook you upload as-is. It takes the
functions you wrote, drops them into the current reference template, and runs
that. So the self-check cells that decide your grade always come from the
up-to-date template, whichever copy you happen to have locally. Pulling a
correction keeps what you read in step with what is graded; it is not the thing
that makes the grade correct.

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
