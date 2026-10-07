# Code Learning Session

## Topic

Statistical comparison of the four Assignment 2 experiment conditions using
anytime performance and final fitness.

## Learning objective

Understand how AOCC summarizes a full convergence curve and how overall and
pairwise significance tests fit together.

## Current understanding

The analysis should first test all conditions together. Pairwise comparisons
should only run when the overall result is significant.

## Relevant files

- `assignments/assignment_2/statistical_tests.py`
- `assignments/assignment_2/plot_results.py`
- `assignments/assignment_2/our_answer.py`
- `assignments/assignment_2/helpers.py`

## Execution flow

```text
CSV files
  -> common budget completed by every run
  -> normalized AOCC score for every run
  -> overall Friedman test
  -> pairwise Wilcoxon tests only if significant
  -> library-provided Holm correction

CSV files
  -> final best fitness for each matched run
  -> overall Friedman test
  -> pairwise Wilcoxon tests only if significant
  -> Holm correction for the six pairwise p-values
```

## Root cause

The original threshold was based on the single best fitness across all runs.
Only three of forty runs reached it, leaving too little information for a useful
time-to-threshold comparison.

## Solution

Use normalized Area Over the Convergence Curve (AOCC) over the common fixed
budget. Use Friedman tests because matching run numbers use matching random
seeds, followed by paired Wilcoxon tests when an overall result is significant.

## Design decisions and tradeoffs

- AOCC uses the entire best-so-far curve instead of one difficult threshold.
- Non-parametric tests avoid assuming normally distributed fitness values.
- `statsmodels` applies Holm correction across multiple pairwise tests.
- Ten matched runs is a small sample, so results should be interpreted with
  suitable caution.

## Edge cases

- Missing conditions cause a clear error.
- Mismatched run numbers cause a clear error because pairing would be invalid.
- Every AOCC score uses the same generation budget.
- Missing generations inside that budget cause a clear error.
- Pairwise tests are skipped after a non-significant overall test.

## Testing and debugging

Run:

```bash
uv run python assignments/assignment_2/statistical_tests.py
```

Check that all four conditions and ten runs appear in the printed summaries.

## Broader impact

The script reads existing result CSVs and does not modify experiment output.
Changing the fitness function or seed strategy would require reconsidering the
threshold definition or paired-test design.

## Remaining gaps

The interpretation of the final numerical results still needs to be written in
the assignment report.

## Mastery checklist

- [x] Understand the problem
- [ ] Understand the root cause
- [ ] Understand the important branches
- [ ] Understand the execution flow
- [ ] Understand the solution
- [ ] Understand why the solution was chosen
- [ ] Understand the important edge cases
- [ ] Understand how to test the change
- [ ] Understand how to debug similar issues
- [ ] Understand the broader impact
- [ ] Can explain the implementation independently
- [ ] Can modify similar code independently

## Final learner explanation

Not requested during this session.
