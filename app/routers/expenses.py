from fastapi import APIRouter, HTTPException
from datetime import date
from pydantic import BaseModel
from app.models.schemas import ExpenseCreate, ExpenseResponse, BudgetCreate, BudgetResponse
from app.database import get_connection
from app.ml.categorizer import categorizer

router = APIRouter()


# ── User profile & budget setup ───────────────────────────────────────────────

class UserRegister(BaseModel):
    username: str
    email:    str

class UserLogin(BaseModel):
    email: str

@router.post("/register", status_code=201)
def register_user(payload: UserRegister):
    """Create a new user row. Returns the new user's id."""
    conn = get_connection()
    # If email already exists, return that user (idempotent)
    existing = conn.execute("SELECT id, username, email, budget FROM users WHERE email = ?", (payload.email,)).fetchone()
    if existing:
        conn.close()
        return dict(existing)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email, budget) VALUES (?, ?, 0)",
        (payload.username, payload.email)
    )
    conn.commit()
    row = conn.execute("SELECT id, username, email, budget FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
    conn.close()
    return dict(row)

@router.post("/login-lookup")
def login_lookup(payload: UserLogin):
    """Look up a user by email. Returns user data or 404."""
    conn = get_connection()
    row  = conn.execute("SELECT id, username, email, budget FROM users WHERE email = ?", (payload.email,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "No account found with that email. Please sign up.")
    return dict(row)

@router.get("/profile/{user_id}")
def get_profile(user_id: int):
    conn = get_connection()
    row  = conn.execute("SELECT id, username, email, budget FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "User not found")
    return dict(row)

class MonthlyBudgetUpdate(BaseModel):
    monthly_budget: float
    username: str | None = None

@router.put("/profile/{user_id}/budget")
def set_monthly_budget(user_id: int, payload: MonthlyBudgetUpdate):
    if payload.monthly_budget <= 0:
        raise HTTPException(400, "Budget must be greater than 0")
    conn = get_connection()
    if payload.username:
        conn.execute(
            "UPDATE users SET budget = ?, username = ? WHERE id = ?",
            (payload.monthly_budget, payload.username, user_id)
        )
    else:
        conn.execute("UPDATE users SET budget = ? WHERE id = ?", (payload.monthly_budget, user_id))
    conn.commit()
    row = conn.execute("SELECT id, username, email, budget FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row)


# ── Categorize (must be before /{user_id} to avoid route shadowing) ───────────

@router.post("/categorize")
def categorize_text(description: str):
    """Quick NLP categorization endpoint — no data saved."""
    return categorizer.predict(description)


@router.post("/retrain")
def retrain_model(descriptions: list[str], labels: list[str]):
    from fastapi import HTTPException
    if len(descriptions) != len(labels):
        raise HTTPException(400, "descriptions and labels must have equal length")
    return categorizer.retrain(descriptions, labels)


# ── Expenses ──────────────────────────────────────────────────────────────────

@router.post("/", response_model=ExpenseResponse, status_code=201)
def add_expense(expense: ExpenseCreate):
    # Auto-categorize if not provided
    category = expense.category
    if not category:
        result   = categorizer.predict(expense.description)
        category = result["category"]

    expense_date = expense.date if expense.date else str(date.today())

    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO expenses (user_id, description, amount, category, date) VALUES (?,?,?,?,?)",
        (expense.user_id, expense.description, expense.amount, category, expense_date),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM expenses WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    conn.close()
    return dict(row)


@router.get("/{user_id}", response_model=list[ExpenseResponse])
def get_expenses(user_id: int, limit: int = 50, offset: int = 0):
    conn  = get_connection()
    rows  = conn.execute(
        "SELECT * FROM expenses WHERE user_id = ? ORDER BY date DESC LIMIT ? OFFSET ?",
        (user_id, limit, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.delete("/{expense_id}", status_code=204)
def delete_expense(expense_id: int):
    conn = get_connection()
    conn.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
    conn.commit()
    conn.close()


# ── Budgets ───────────────────────────────────────────────────────────────────

@router.post("/budgets/", response_model=BudgetResponse, status_code=201)
def set_budget(budget: BudgetCreate):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO budgets (user_id, category, limit_amt, month)
        VALUES (?,?,?,?)
        ON CONFLICT(user_id, category, month) DO UPDATE SET limit_amt=excluded.limit_amt
        """,
        (budget.user_id, budget.category, budget.limit_amt, budget.month),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM budgets WHERE id = ?", (cursor.lastrowid,)
    ).fetchone()
    conn.close()
    return dict(row)


@router.get("/budgets/{user_id}", response_model=list[BudgetResponse])
def get_budgets(user_id: int, month: str | None = None):
    from datetime import date
    current_month = month or date.today().strftime("%Y-%m")
    conn = get_connection()

    # Get budgets for the requested/current month
    rows = conn.execute(
        "SELECT * FROM budgets WHERE user_id = ? AND month = ?",
        (user_id, current_month)
    ).fetchall()

    # If no budgets exist for this month, carry forward from the most recent past month
    if not rows:
        last_rows = conn.execute(
            """
            SELECT category, limit_amt
            FROM budgets
            WHERE user_id = ?
              AND month < ?
            ORDER BY month DESC
            """,
            (user_id, current_month)
        ).fetchall()

        # Get unique categories from most recent month only
        seen = set()
        to_carry = []
        for r in last_rows:
            if r["category"] not in seen:
                seen.add(r["category"])
                to_carry.append(r)

        # Insert carried-forward budgets for current month
        if to_carry:
            cursor = conn.cursor()
            for r in to_carry:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO budgets (user_id, category, limit_amt, month)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, r["category"], r["limit_amt"], current_month)
                )
            conn.commit()

            # Re-fetch the newly inserted rows
            rows = conn.execute(
                "SELECT * FROM budgets WHERE user_id = ? AND month = ?",
                (user_id, current_month)
            ).fetchall()

    conn.close()
    return [dict(r) for r in rows]