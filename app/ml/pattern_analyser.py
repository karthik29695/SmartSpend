"""
Spending Pattern Analyser.

Analyses historical expense data to surface:
  - Per-category breakdown (totals, averages, % of spend)
  - Month-over-month trend per category  (increasing / stable / decreasing)
  - Anomaly detection via Z-score on transaction amounts
  - Auto-generated savings tips
"""

from collections import defaultdict
from datetime import datetime
import statistics


def analyse_patterns(expenses: list[dict], budget: float) -> dict:
    """
    Parameters
    ----------
    expenses : list of dicts with keys: amount, category, date
    budget   : user's total monthly budget

    Returns
    -------
    dict with keys: patterns, anomalies, monthly_totals, savings_tip
    """
    if not expenses:
        return {
            "patterns": [],
            "anomalies": [],
            "monthly_totals": {},
            "savings_tip": "Start adding expenses to get personalised insights!",
        }

    # ── group by category ──────────────────────────────────────────────────
    by_cat: dict[str, list[float]] = defaultdict(list)
    for e in expenses:
        by_cat[e["category"]].append(float(e["amount"]))

    total_spend = sum(a for amounts in by_cat.values() for a in amounts)

    patterns = []
    for cat, amounts in by_cat.items():
        patterns.append({
            "category":          cat,
            "total_spent":       round(sum(amounts), 2),
            "percentage":        round(sum(amounts) / total_spend * 100, 1) if total_spend else 0,
            "avg_transaction":   round(statistics.mean(amounts), 2),
            "transaction_count": len(amounts),
            "trend":             _compute_trend(expenses, cat),
        })

    patterns.sort(key=lambda x: x["total_spent"], reverse=True)

    # ── monthly totals ─────────────────────────────────────────────────────
    monthly: dict[str, float] = defaultdict(float)
    for e in expenses:
        month = e["date"][:7]          # "YYYY-MM"
        monthly[month] += float(e["amount"])
    monthly_totals = {k: round(v, 2) for k, v in sorted(monthly.items())}

    # ── anomaly detection (Z-score > 2.5) ─────────────────────────────────
    all_amounts = [float(e["amount"]) for e in expenses]
    anomalies: list[str] = []
    if len(all_amounts) >= 3:
        mean = statistics.mean(all_amounts)
        std  = statistics.stdev(all_amounts)
        if std > 0:
            for e in expenses:
                z = (float(e["amount"]) - mean) / std
                if abs(z) > 2.5:
                    anomalies.append(
                        f"Unusual transaction: ₹{e['amount']:.2f} on {e['date']} "
                        f"({e['category']}) — {abs(z):.1f}σ from mean"
                    )

    # ── savings tip ────────────────────────────────────────────────────────
    tip = _generate_tip(patterns, total_spend, budget)

    return {
        "patterns":      patterns,
        "anomalies":     anomalies,
        "monthly_totals": monthly_totals,
        "total_spent":   round(total_spend, 2),
        "budget":        budget,
        "savings_tip":   tip,
    }


# ── helpers ────────────────────────────────────────────────────────────────────

def _compute_trend(expenses: list[dict], category: str) -> str:
    """Compare last-month spend vs previous month for this category."""
    monthly: dict[str, float] = defaultdict(float)
    for e in expenses:
        if e["category"] == category:
            month = e["date"][:7]
            monthly[month] += float(e["amount"])

    sorted_months = sorted(monthly.keys())
    if len(sorted_months) < 2:
        return "stable"

    last   = monthly[sorted_months[-1]]
    prev   = monthly[sorted_months[-2]]
    change = (last - prev) / prev if prev else 0

    if change > 0.10:
        return "increasing"
    if change < -0.10:
        return "decreasing"
    return "stable"


def _generate_tip(patterns: list[dict], total: float, budget: float) -> str:
    if not patterns:
        return "No data yet."

    top = patterns[0]
    over_budget = total > budget

    if over_budget:
        excess = total - budget
        return (
            f"You've exceeded your budget by ₹{excess:.2f}. "
            f"Your biggest spend is {top['category']} ({top['percentage']}%). "
            f"Try setting a sub-budget for it."
        )

    increasing = [p for p in patterns if p["trend"] == "increasing"]
    if increasing:
        cats = ", ".join(p["category"] for p in increasing[:2])
        return f"Spending is rising in: {cats}. Consider reviewing these categories."

    remaining = budget - total
    return (
        f"Great job! You're ₹{remaining:.2f} under budget. "
        f"Top spend category: {top['category']} ({top['percentage']}%)."
    )
