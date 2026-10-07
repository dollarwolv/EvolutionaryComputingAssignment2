"""Compare anytime performance and final fitness between the four conditions.

Run this file from the repository root with:

    uv run python assignments/assignment_2/statistical_tests.py

Lower fitness is better throughout this assignment.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

from plot_results import LABELS, load_results

ALPHA = 0.05


def print_pairwise_results(
    title: str,
    pairs: list[tuple[str, str]],
    statistics: list[float],
    raw_p_values: list[float],
) -> None:
    """Print pairwise results with one Holm correction over all six pairs."""
    significant_results, adjusted_p_values, _, _ = multipletests(
        raw_p_values,
        alpha=ALPHA,
        method="holm",
    )

    print(f"\n{title}")
    for pair, statistic, raw_p, adjusted_p, significant in zip(
        pairs,
        statistics,
        raw_p_values,
        adjusted_p_values,
        significant_results,
        strict=True,
    ):
        first, second = pair
        significant_text = "yes" if significant else "no"
        print(
            f"  {LABELS[first]} vs {LABELS[second]}: "
            f"statistic = {statistic:.3f}, raw p = {raw_p:.4f}, "
            f"Holm-adjusted p = {adjusted_p:.4f}, "
            f"significant = {significant_text}"
        )


def matching_run_ids(results: dict[str, pd.DataFrame]) -> list[int]:
    """Return the run numbers after checking that every condition has them."""
    run_ids_by_condition = [
        set(df["run"].astype(int).unique()) for df in results.values()
    ]
    first_run_ids = run_ids_by_condition[0]

    if any(run_ids != first_run_ids for run_ids in run_ids_by_condition):
        raise ValueError(
            "All conditions must contain the same run numbers for the paired tests."
        )

    return sorted(first_run_ids)


def aocc_by_run(
    results: dict[str, pd.DataFrame],
) -> tuple[list[int], int, dict[str, list[float]]]:
    """Calculate one normalized Area Over the Convergence Curve per run."""
    run_ids = matching_run_ids(results)

    # Every run is compared over the longest budget that all runs completed
    common_budget = min(
        int(df.groupby("run")["generation"].max().min()) for df in results.values()
    )

    curves = {}
    for condition, df in results.items():
        condition_curves = []
        for run_id in run_ids:
            run = df[df["run"] == run_id].set_index("generation")
            curve = run["best_so_far"].reindex(range(common_budget + 1))

            condition_curves.append(curve.to_numpy(dtype=float))
        curves[condition] = condition_curves

    all_fitness_values = np.concatenate(
        [curve for condition_curves in curves.values() for curve in condition_curves]
    )
    lower_bound = float(all_fitness_values.min())
    upper_bound = float(all_fitness_values.max())

    scores = {
        condition: [
            float(np.mean((upper_bound - curve) / (upper_bound - lower_bound)))
            for curve in condition_curves
        ]
        for condition, condition_curves in curves.items()
    }

    return run_ids, common_budget, scores


def run_aocc_tests(results: dict[str, pd.DataFrame]) -> None:
    """Compare anytime performance over one shared generation budget."""
    run_ids, common_budget, scores = aocc_by_run(results)
    conditions = list(scores)

    print(f"\nNormalized AOCC through generation {common_budget} (higher is better)")
    for condition, values in scores.items():
        median = float(np.median(values))
        print(f"  {LABELS[condition]}: median = {median:.3f}")

    overall = friedmanchisquare(*(scores[name] for name in conditions))

    print(f"\nAOCC: overall Friedman test ({len(run_ids)} matched runs)")
    print(
        f"  chi-square({len(conditions) - 1}) = {overall.statistic:.3f}, "
        f"p = {overall.pvalue:.4f}"
    )

    if overall.pvalue >= ALPHA:
        print("  Not significant, so pairwise Wilcoxon tests are skipped.")
        return

    pairs = list(combinations(conditions, 2))
    statistics = []
    raw_p_values = []

    for first, second in pairs:
        differences = np.round(
            np.asarray(scores[first]) - np.asarray(scores[second]),
            decimals=12,
        )
        result = wilcoxon(differences, alternative="two-sided", method="auto")
        statistics.append(float(result.statistic))
        raw_p_values.append(float(result.pvalue))

    print_pairwise_results(
        "AOCC: pairwise Wilcoxon signed-rank tests",
        pairs,
        statistics,
        raw_p_values,
    )


def final_fitness_by_run(
    results: dict[str, pd.DataFrame],
) -> tuple[list[int], dict[str, list[float]]]:
    """Return final best fitness values aligned by matching run number."""
    run_ids = matching_run_ids(results)
    final_values = {}

    for condition, df in results.items():
        final_rows = df.loc[df.groupby("run")["generation"].idxmax()]
        final_rows = final_rows.set_index("run").loc[run_ids]
        final_values[condition] = final_rows["best_so_far"].astype(float).tolist()

    return run_ids, final_values


def run_final_fitness_tests(results: dict[str, pd.DataFrame]) -> None:
    """Compare final fitness using the matching random seed as a block."""
    run_ids, final_values = final_fitness_by_run(results)
    conditions = list(final_values)

    print("\nFinal best fitness (lower is better)")
    for condition, values in final_values.items():
        median = float(np.median(values))
        print(f"  {LABELS[condition]}: median = {median:.3f}")

    overall = friedmanchisquare(*(final_values[name] for name in conditions))

    print(f"\nFinal fitness: overall Friedman test ({len(run_ids)} matched runs)")
    print(
        f"  chi-square({len(conditions) - 1}) = {overall.statistic:.3f}, "
        f"p = {overall.pvalue:.4f}"
    )

    if overall.pvalue >= ALPHA:
        print("  Not significant, so pairwise Wilcoxon tests are skipped.")
        return

    pairs = list(combinations(conditions, 2))
    statistics = []
    raw_p_values = []

    for first, second in pairs:
        # Supplying rounded paired differences avoids tiny floating-point
        # errors changing how equal ranks are treated by the test.
        differences = np.round(
            np.asarray(final_values[first]) - np.asarray(final_values[second]),
            decimals=12,
        )
        result = wilcoxon(differences, alternative="two-sided", method="auto")
        statistics.append(float(result.statistic))
        raw_p_values.append(float(result.pvalue))

    print_pairwise_results(
        "Final fitness: pairwise Wilcoxon signed-rank tests",
        pairs,
        statistics,
        raw_p_values,
    )


def main() -> None:
    results = load_results()
    missing_conditions = set(LABELS) - set(results)
    if missing_conditions:
        missing = ", ".join(sorted(missing_conditions))
        raise ValueError(f"missing result files for: {missing}")

    print(f"Significance level: alpha = {ALPHA}")

    run_aocc_tests(results)
    run_final_fitness_tests(results)


if __name__ == "__main__":
    main()
