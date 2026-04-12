"""
Budget-Aware Recommendation Engine.

Logic:
  1. Compute average spend per category from user's history.
  2. Compare against their per-category budget limit.
  3. Surface products/brands that are cheaper than what the user typically spends,
     ranked by a composite score: (savings × category_relevance).

In production you'd replace PRODUCT_CATALOG with live web-scraped data.
The scraper module (app/services/scraper.py) shows the recommended integration point.
"""

from app.database import get_connection

# ── Static product catalogue (replace / augment with scraped data) ─────────────
PRODUCT_CATALOG = [
    # Food & Dining
    {"name": "Aashirvaad Atta 10kg",    "brand": "ITC",       "price": 380,  "category": "Food & Dining"},
    {"name": "Tata Salt 1kg",           "brand": "Tata",      "price":  28,  "category": "Food & Dining"},
    {"name": "Amul Butter 500g",        "brand": "Amul",      "price": 275,  "category": "Food & Dining"},
    {"name": "Sunfeast Biscuits 1kg",   "brand": "ITC",       "price":  85,  "category": "Food & Dining"},
    # Transport
    {"name": "Monthly Metro Pass",      "brand": "DMRC",      "price": 800,  "category": "Transport"},
    {"name": "Bus Monthly Pass",        "brand": "DTC",       "price": 450,  "category": "Transport"},
    {"name": "Rapido Bike Taxi Pack",   "brand": "Rapido",    "price": 199,  "category": "Transport"},
    # Shopping
    {"name": "Puma T-shirt",            "brand": "Puma",      "price": 599,  "category": "Shopping"},
    {"name": "Peter England Shirt",     "brand": "PE",        "price": 799,  "category": "Shopping"},
    {"name": "Noise Smartwatch",        "brand": "Noise",     "price":1499,  "category": "Shopping"},
    # Utilities
    {"name": "Jio 28-day Recharge",     "brand": "Jio",       "price": 179,  "category": "Utilities"},
    {"name": "BSNL 30-day Pack",        "brand": "BSNL",      "price": 107,  "category": "Utilities"},
    {"name": "Amazon Prime Monthly",    "brand": "Amazon",    "price": 179,  "category": "Utilities"},
    # Health
    {"name": "Dolo 650 strip",          "brand": "Micro",     "price":  30,  "category": "Health"},
    {"name": "HealthifyMe Basic",       "brand": "HealthifyMe", "price": 199,  "category": "Health"},
    # Entertainment
    {"name": "Zee5 Monthly",            "brand": "Zee",       "price": 99,   "category": "Entertainment"},
    {"name": "Hotstar Mobile Monthly",  "brand": "Disney",    "price": 149,  "category": "Entertainment"},
    # Education
    {"name": "Coursera 1-month",        "brand": "Coursera",  "price":1499,  "category": "Education"},
    {"name": "YouTube Premium Monthly", "brand": "Google",    "price": 129,  "category": "Education"},
]


def get_recommendations(user_id: int, category: str, max_budget: float) -> list[dict]:
    """
    Return products in `category` whose price ≤ max_budget,
    sorted by savings (highest first), with reason text.
    """
    avg_spend = _avg_spend_in_category(user_id, category)

    candidates = [
        p for p in PRODUCT_CATALOG
        if p["category"] == category and p["price"] <= max_budget
    ]

    if not candidates:
        return []

    results = []
    for product in candidates:
        savings = round(avg_spend - product["price"], 2) if avg_spend > product["price"] else 0.0
        results.append({
            **product,
            "savings": savings,
            "reason":  _build_reason(product, savings, avg_spend),
        })

    results.sort(key=lambda x: x["savings"], reverse=True)
    return results[:5]  # top 5


def get_cross_category_recommendations(user_id: int, total_budget: float) -> list[dict]:
    """
    Find categories where user is overspending and suggest cheaper alternatives.
    """
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT category, SUM(amount) AS total
        FROM expenses
        WHERE user_id = ?
        GROUP BY category
        ORDER BY total DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    recs = []
    for row in rows:
        cat   = row["category"]
        spent = float(row["total"])
        per_cat_budget = total_budget / max(len(rows), 1)

        if spent > per_cat_budget:
            recs.extend(get_recommendations(user_id, cat, per_cat_budget))

    return recs[:8]


# ── helpers ───────────────────────────────────────────────────────────────────

def _avg_spend_in_category(user_id: int, category: str) -> float:
    conn = get_connection()
    row  = conn.execute(
        """
        SELECT AVG(amount) AS avg_amt
        FROM expenses
        WHERE user_id = ? AND category = ?
        """,
        (user_id, category),
    ).fetchone()
    conn.close()
    return float(row["avg_amt"]) if row and row["avg_amt"] else 0.0


def _build_reason(product: dict, savings: float, avg_spend: float) -> str:
    if savings > 0:
        pct = round(savings / avg_spend * 100) if avg_spend else 0
        return f"Save ₹{savings:.0f} ({pct}% less than your usual spend in this category)"
    return f"Affordable option from {product['brand']} within your budget"
