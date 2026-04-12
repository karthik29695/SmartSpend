from fastapi import APIRouter, HTTPException
from app.database import get_connection
from app.ml.pattern_analyser import analyse_patterns

router = APIRouter()


@router.get("/{user_id}/insights")
def get_insights(user_id: int, month: str | None = None):
    """Full spending analysis for a user (optionally filtered to a month YYYY-MM)."""
    conn = get_connection()

    user = conn.execute("SELECT budget FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(404, "User not found")

    if month:
        rows = conn.execute(
            "SELECT amount, category, date FROM expenses WHERE user_id = ? AND strftime('%Y-%m', date) = ?",
            (user_id, month),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT amount, category, date FROM expenses WHERE user_id = ?",
            (user_id,),
        ).fetchall()
    conn.close()

    expenses = [dict(r) for r in rows]
    result = analyse_patterns(expenses, float(user["budget"]))

    # Build weekly spend for current week (Mon–Sun) from DB
    weekly_rows = conn2 = None
    try:
        from app.database import get_connection as _gc
        conn2 = _gc()
        weekly_rows = conn2.execute(
            """
            SELECT strftime('%w', date) AS dow, SUM(amount) AS total
            FROM expenses
            WHERE user_id = ?
              AND date >= date('now', 'weekday 0', '-7 days')
            GROUP BY dow
            """,
            (user_id,),
        ).fetchall()
        conn2.close()
    except Exception:
        if conn2:
            conn2.close()

    # Map sqlite dow (0=Sun..6=Sat) → Mon-indexed array [Mon,Tue,Wed,Thu,Fri,Sat,Sun]
    dow_map = {str(i): 0.0 for i in range(7)}
    for r in (weekly_rows or []):
        dow_map[str(r["dow"])] = round(float(r["total"]), 2)
    # Reorder: sqlite 1=Mon,2=Tue,...,6=Sat,0=Sun
    weekly_spend = [dow_map[str(d)] for d in [1,2,3,4,5,6,0]]

    result["weekly_spend"] = weekly_spend
    return result


@router.get("/{user_id}/summary")
def get_summary(user_id: int):
    """Quick dashboard summary: total spent, budget used %, top category, chart data."""
    conn = get_connection()
    user = conn.execute("SELECT budget FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise HTTPException(404, "User not found")

    row = conn.execute(
        "SELECT SUM(amount) AS total, COUNT(*) AS txns FROM expenses WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    top_cat = conn.execute(
        """
        SELECT category, SUM(amount) AS s
        FROM expenses WHERE user_id = ?
        GROUP BY category ORDER BY s DESC LIMIT 1
        """,
        (user_id,),
    ).fetchone()

    # Category breakdown for pie chart
    cat_rows = conn.execute(
        """
        SELECT category, SUM(amount) AS s
        FROM expenses WHERE user_id = ?
        GROUP BY category ORDER BY s DESC
        """,
        (user_id,),
    ).fetchall()

    # Monthly trend for line chart (last 6 months, oldest first)
    trend_rows = conn.execute(
        """
        SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS total
        FROM expenses WHERE user_id = ?
        GROUP BY month ORDER BY month ASC
        LIMIT 6
        """,
        (user_id,),
    ).fetchall()

    conn.close()

    total  = float(row["total"]) if row["total"] else 0.0
    budget = float(user["budget"])

    return {
        "total_spent":        round(total, 2),
        "budget":             budget,
        "budget_used_pct":    round(total / budget * 100, 1) if budget else 0,
        "transactions":       row["txns"],
        "top_category":       top_cat["category"] if top_cat else "N/A",
        "status":             "over budget" if total > budget else "on track",
        "category_breakdown": {r["category"]: round(float(r["s"]), 2) for r in cat_rows},
        "monthly_trend":      [{"month": r["month"], "amount": round(float(r["total"]), 2)} for r in trend_rows],
    }