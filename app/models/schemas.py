from pydantic import BaseModel, Field
from typing import Optional

# ── User ──────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str
    email: str
    budget: float = 0.0

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    budget: float

# ── Expense ───────────────────────────────────────────────────────────────────

class ExpenseCreate(BaseModel):
    user_id: int
    description: str = Field(..., example="Bought groceries at Walmart")
    amount: float = Field(..., gt=0, example=45.99)
    date: Optional[str] = None          # accept ISO string "YYYY-MM-DD" or null
    category: Optional[str] = None      # auto-filled by ML if omitted

class ExpenseResponse(BaseModel):
    id: int
    user_id: int
    description: str
    amount: float
    category: str
    date: str
    created_at: str

# ── Budget ────────────────────────────────────────────────────────────────────

class BudgetCreate(BaseModel):
    user_id: int
    category: str
    limit_amt: float
    month: str = Field(..., example="2025-06")

class BudgetResponse(BaseModel):
    id: int
    user_id: int
    category: str
    limit_amt: float
    month: str

# ── Recommendation ────────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    user_id: int
    category: str = Field(..., example="Food")
    max_budget: float = Field(..., example=100.0)

class ProductRecommendation(BaseModel):
    name: str
    brand: str
    price: float
    category: str
    savings: float           # vs. user's usual spend in this category
    reason: str

# ── Analytics ─────────────────────────────────────────────────────────────────

class SpendingPattern(BaseModel):
    category: str
    total_spent: float
    percentage: float
    avg_transaction: float
    transaction_count: int
    trend: str               # "increasing" | "stable" | "decreasing"

class MonthlyInsight(BaseModel):
    month: str
    total_spent: float
    budget: float
    top_category: str
    patterns: list[SpendingPattern]
    anomalies: list[str]
    savings_tip: str