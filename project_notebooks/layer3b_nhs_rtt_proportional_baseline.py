"""Layer 3b: Proportional-allocation baseline comparison for the MILP optimisation layer.

Answers Research Question 3 (Chapter 1, section 1.5): whether the MILP identifies
materially better theatre session allocations than a simple proportional distribution.

This script does not retrain or modify any existing model, does not touch
nhs_rtt_pipeline/optimisation.py, and does not overwrite any existing output file.
It reads the real, already-generated central-scenario MILP allocation
(outputs/lp_allocation_output.csv, the 5,000-session central-productivity run) and
computes what a simple proportional allocator would have achieved with the identical
budget, the identical per-combination productivity rates, and the identical
12-session-per-Trust-specialty-month physical capacity cap the MILP itself respects.

Proportional allocation method:
1. Each Trust-specialty combination receives a session share proportional to its
   share of the total forecast backlog: budget * (combination backlog / total backlog).
2. That share is capped at the same max_feasible_additional_sessions limit the MILP
   uses (12 sessions per combination), and floored to an integer, since sessions are
   discrete and cannot exceed a combination's real physical capacity.
3. Unlike the MILP, a simple proportional allocator does not redistribute the budget
   left over when a combination's proportional share exceeds its cap; that unused
   budget is reported explicitly rather than hidden.
4. Completed pathways for each combination are capped at that combination's own
   forecast backlog, using the same productivity_link logic the MILP applies.

Output: outputs/lp_proportional_baseline_comparison.csv, one row summarising the
national comparison between the MILP allocation and this proportional baseline.
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOCATION_PATH = PROJECT_ROOT / "outputs" / "lp_allocation_output.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "lp_proportional_baseline_comparison.csv"


def main() -> None:
    df = pd.read_csv(ALLOCATION_PATH)

    budget = int(df["sessions_allocated"].sum())
    total_backlog = float(df["baseline_predicted_backlog"].sum())

    df["proportional_raw_share"] = budget * df["baseline_predicted_backlog"] / total_backlog
    df["proportional_sessions"] = np.minimum(
        np.floor(df["proportional_raw_share"]), df["max_feasible_additional_sessions"]
    ).clip(lower=0).astype(int)

    sessions_used_proportional = int(df["proportional_sessions"].sum())
    unused_budget_proportional = budget - sessions_used_proportional

    df["proportional_completed"] = np.minimum(
        df["patients_completed_per_session"] * df["proportional_sessions"],
        df["baseline_predicted_backlog"],
    )
    total_completed_proportional = float(df["proportional_completed"].sum())
    total_completed_milp = float(df["simulated_completed_pathways"].sum())

    pct_reduction_milp = 100 * total_completed_milp / total_backlog
    pct_reduction_proportional = 100 * total_completed_proportional / total_backlog

    summary = pd.DataFrame(
        [
            {
                "session_budget": budget,
                "total_baseline_backlog": total_backlog,
                "milp_sessions_used": budget,
                "milp_completed_pathways": total_completed_milp,
                "milp_percent_reduction": pct_reduction_milp,
                "proportional_sessions_used": sessions_used_proportional,
                "proportional_unused_budget": unused_budget_proportional,
                "proportional_completed_pathways": total_completed_proportional,
                "proportional_percent_reduction": pct_reduction_proportional,
                "milp_advantage_percentage_points": pct_reduction_milp - pct_reduction_proportional,
                "milp_relative_improvement_percent": 100
                * (pct_reduction_milp - pct_reduction_proportional)
                / pct_reduction_proportional,
            }
        ]
    )
    summary.to_csv(OUTPUT_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
