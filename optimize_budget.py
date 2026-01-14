"""Budget allocation script for the Bayesian MMM project.

This script reads in the final_dataset CSV (containing spend and sales
data) and a CSV specifying monthly revenue targets for each channel.
It then computes advertising ROI estimates and allocates a specified
total budget across channels and months based on a simple formula.

The allocation uses a combination of:

1. **Channel ROI** – estimated as elasticity (β) times total sales
   divided by total spend for each channel.
2. **Monthly revenue targets** – months with higher revenue targets
   receive proportionally more budget.
3. **Equal distribution component** – a fraction of each channel's
   budget is spread evenly across months to ensure baseline coverage.

The mix between equal distribution and ROI/target based distribution
is controlled by the `mix` parameter (default 0.5).

Usage
-----
Run this script from the repository root with Python 3. For example:

    python optimize_budget.py \
        --dataset final_dataset.csv \
        --targets targets_2026.csv \
        --budget 2400000000 \
        --mix 0.5 \
        --output allocation.csv

where `targets_2026.csv` is a CSV with columns: month (1-12),
oy_target, naver_target. The output file will contain the monthly
budget allocation for each channel.
"""

from __future__ import annotations

import argparse
import pandas as pd


def compute_roi(df: pd.DataFrame, beta: float, spend_col: str, sales_col: str) -> float:
    """Estimate channel ROI given elasticity and total sales/spend.

    Parameters
    ----------
    df : DataFrame
        Dataset containing spend and sales columns.
    beta : float
        Estimated elasticity for the channel (from the MMM).
    spend_col : str
        Column name for advertising spend.
    sales_col : str
        Column name for sales.

    Returns
    -------
    roi : float
        Estimated return on investment (unitless). Higher means more
        efficient.
    """
    total_sales = df[sales_col].sum()
    total_spend = df[spend_col].sum()
    if total_spend <= 0:
        return 0.0
    return beta * (total_sales / total_spend)


def allocate_budget(
    df: pd.DataFrame,
    targets: pd.DataFrame,
    total_budget: float,
    beta_n: float = 0.345,
    beta_o: float = 0.236,
    mix: float = 0.5,
) -> pd.DataFrame:
    """Allocate total budget across months and channels.

    Parameters
    ----------
    df : DataFrame
        Dataset with columns `naver_spend`, `oy_spend`, `naver_sales`,
        `oy_sales`, and `date`. Only the year of interest should be
        included.
    targets : DataFrame
        Monthly revenue targets with columns: `month`, `oy_target`,
        `naver_target`.
    total_budget : float
        Total advertising budget for both channels combined.
    beta_n : float, optional
        Elasticity coefficient for NAVER channel.
    beta_o : float, optional
        Elasticity coefficient for OY channel.
    mix : float, optional
        Mix between equal distribution (0.5) and ROI/target based
        distribution. 0 means all based on equal monthly share, 1
        means all based on ROI/target.

    Returns
    -------
    allocation : DataFrame
        Monthly budget allocation with columns: `month`,
        `naver_budget`, `oy_budget`.
    """
    # Compute ROI for each channel
    roi_n = compute_roi(df, beta_n, spend_col="naver_spend", sales_col="naver_sales")
    roi_o = compute_roi(df, beta_o, spend_col="oy_spend", sales_col="oy_sales")
    if roi_n + roi_o == 0:
        # avoid division by zero; default equal split
        channel_share_n = 0.5
        channel_share_o = 0.5
    else:
        channel_share_n = roi_n / (roi_n + roi_o)
        channel_share_o = roi_o / (roi_n + roi_o)
    # Separate budgets for each channel
    budget_n = total_budget * channel_share_n
    budget_o = total_budget * channel_share_o
    # Compute base equal weight across months for each channel
    months = targets["month"].astype(int)
    # total months count
    n_months = len(months)
    equal_weight = 1.0 / n_months
    # ROI/target based weight for each channel
    # For each channel, weight_j = (target_j / total_target_channel)
    n_targets = targets["naver_target"].astype(float)
    o_targets = targets["oy_target"].astype(float)
    total_n_target = n_targets.sum()
    total_o_target = o_targets.sum()
    if total_n_target <= 0:
        n_weights = pd.Series([equal_weight] * n_months)
    else:
        n_weights = n_targets / total_n_target
    if total_o_target <= 0:
        o_weights = pd.Series([equal_weight] * n_months)
    else:
        o_weights = o_targets / total_o_target
    # Combine equal and target weights
    n_final_weights = (1 - mix) * equal_weight + mix * n_weights
    o_final_weights = (1 - mix) * equal_weight + mix * o_weights
    # Normalize weights to ensure they sum to 1
    n_final_weights = n_final_weights / n_final_weights.sum()
    o_final_weights = o_final_weights / o_final_weights.sum()
    # Compute monthly budgets
    n_month_budgets = budget_n * n_final_weights
    o_month_budgets = budget_o * o_final_weights
    allocation = pd.DataFrame(
        {
            "month": months,
            "naver_budget": n_month_budgets.round(2),
            "oy_budget": o_month_budgets.round(2),
        }
    )
    return allocation


def main() -> None:
    parser = argparse.ArgumentParser(description="Allocate budget across channels and months.")
    parser.add_argument("--dataset", required=True, help="Path to final_dataset CSV (with date, spend, sales).")
    parser.add_argument("--targets", required=True, help="CSV file with columns month, oy_target, naver_target.")
    parser.add_argument("--budget", type=float, required=True, help="Total annual budget to allocate (e.g. 2.4e9 for 24억원).")
    parser.add_argument("--mix", type=float, default=0.5, help="Mix between equal and target based distribution (0-1).")
    parser.add_argument("--output", required=False, help="Optional path to save allocation CSV.")
    args = parser.parse_args()
    # Load dataset and ensure date column exists
    df = pd.read_csv(args.dataset)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("Dataset must contain a 'date' column")
    targets = pd.read_csv(args.targets)
    allocation = allocate_budget(df, targets, args.budget, mix=args.mix)
    if args.output:
        allocation.to_csv(args.output, index=False)
        print(f"Allocation saved to {args.output}")
    else:
        print(allocation)


if __name__ == "__main__":
    main()