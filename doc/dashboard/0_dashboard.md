# Metrics Dashboard

PyRIT tracks numeric "how good is this component" metrics for several parts of the framework.
This section renders those metrics as leaderboard tables, so you can compare configurations at
a glance instead of digging through JSONL files by hand.

## What's here today

- **[Scorer Quality](1_scorer_quality.ipynb)** — an Objective Scorer Leaderboard (accuracy,
  F1, precision, recall) and a Harm Scorer Leaderboard (mean absolute error, Krippendorff's
  alpha), built from the evaluation registries the team already maintains under
  `pyrit/datasets/scorer_evals/`. See [Scorer Metrics](../code/scoring/4_scorer_metrics.ipynb)
  for what these numbers mean and [Scoring Scorers](../blog/2026_04_14_scoring_scorers.md) for
  the full evaluation framework.

## What's planned

Benchmark leaderboards — adversarial-model effectiveness and objective-target robustness,
built from `AdversarialBenchmark` scenario runs — are the natural next addition. The exporter
that produces the underlying data (`build_scripts/export_adversarial_benchmark_result.py`) now
records enough identity information (`objective_target`, `objective_scorer`, `dataset`) to
support that, but the leaderboard page itself isn't built yet.

## Refreshing the data

These pages read committed JSONL files, not live services, so refreshing the dashboard is a
two-step, human-in-the-loop process: regenerate the data, then re-render the page.

1. Regenerate scorer metrics locally:

   ```bash
   python -m build_scripts.evaluate_scorers
   ```

   This is safe to re-run — scorer configurations that already have up-to-date metrics are
   skipped automatically (see [Scorer Metrics](../code/scoring/4_scorer_metrics.ipynb)).
   Commit the updated files under `pyrit/datasets/scorer_evals/` through a normal PR.
2. Rebuild this page. The notebook re-reads the committed JSONL files each time it runs, and
   the site rebuilds automatically on every merge to `main`.

There's no CI automation that runs `evaluate_scorers.py` or opens a PR for you yet — both steps
above are manual today.
