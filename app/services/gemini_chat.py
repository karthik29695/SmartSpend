"""
SmartSpend AI — Gemini Chatbot Service

Capabilities:
  1. Answer questions about the user's spending history
  2. Give budget advice & personalised savings tips
  3. Help categorize expenses (calls the ML categorizer)
  4. Recommend products based on conversation context

Architecture:
  - Each conversation turn includes a rich financial context block so
    Gemini always has up-to-date data about the user.
  - Multi-turn history is maintained per session (in-memory dict).
  - Intent detection routes the message to specialised handlers before
    calling Gemini, enriching the prompt with live data.
  - Tool-call pattern: when Gemini's reply contains [ACTION:...] tags
    the service executes the action and appends the result automatically.
"""

import os
import json
import re
import logging
from datetime import datetime

import httpx

from app.database import get_connection
from app.ml.categorizer import categorizer
from app.ml.pattern_analyser import analyse_patterns
from app.ml.recommender import get_recommendations

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# ── In-memory conversation store  {session_id: [{"role":..,"parts":[{"text":...}]}]}
_sessions: dict[str, list[dict]] = {}

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are SmartSpend AI, a friendly and knowledgeable personal finance assistant.
You help users manage their expenses, understand their spending habits, and make smarter financial decisions.

Your core capabilities:
1. SPENDING HISTORY — Answer questions about past expenses using the financial context provided.
2. BUDGET ADVICE — Give actionable, personalised savings tips based on actual spend data.
3. EXPENSE CATEGORIZATION — When a user describes a purchase, identify its category.
4. PRODUCT RECOMMENDATIONS — Suggest affordable alternatives when users want to save money.

Tone: Friendly, concise, non-judgmental. Use ₹ for currency. Keep responses under 400 words unless a detailed breakdown is explicitly requested.

When you need live data you can emit one of these action tags on its own line:
  [ACTION:CATEGORIZE:<description>]
  [ACTION:RECOMMEND:<category>:<budget_amount>]

The system will execute the action and append the result before your final reply.

Always ground your advice in the financial context block provided — never invent numbers.
If data is missing, say so honestly and suggest what the user can do."""


# ── Public API ─────────────────────────────────────────────────────────────────

async def chat(
    user_id: int,
    session_id: str,
    message: str,
) -> dict:
    """
    Send a message and get a response.

    Returns
    -------
    {
        "reply": str,
        "session_id": str,
        "actions_taken": list[str],   # any tool calls that ran
        "suggested_questions": list[str],
    }
    """
    if not GEMINI_API_KEY:
        return _no_key_response()

    # Build financial context for this turn
    context     = _build_financial_context(user_id)
    history     = _sessions.setdefault(session_id, [])
    actions_taken: list[str] = []

    # Pre-process: detect explicit intents and enrich the message
    enriched_message, pre_actions = _pre_process(message, user_id, context)
    actions_taken.extend(pre_actions)

    # Assemble the full prompt
    user_turn = (
        f"[FINANCIAL CONTEXT]\n{context}\n\n"
        f"[USER MESSAGE]\n{enriched_message}"
    )

    history.append({"role": "user", "parts": [{"text": user_turn}]})

    # First Gemini call
    raw_reply = await _call_gemini(history)

    # Post-process: execute any [ACTION:...] tags in the reply
    final_reply, post_actions = await _execute_actions(raw_reply, user_id)
    actions_taken.extend(post_actions)

    # If actions ran, do a second Gemini pass with the enriched context
    if post_actions:
        history[-1]["parts"][0]["text"] += f"\n\n[ACTION RESULTS]\n{final_reply}"
        history.append({"role": "model", "parts": [{"text": raw_reply}]})
        final_reply = await _call_gemini(history)
        history.append({"role": "model", "parts": [{"text": final_reply}]})
    else:
        history.append({"role": "model", "parts": [{"text": final_reply}]})

    # Keep history bounded (last 20 turns = 10 exchanges)
    if len(history) > 20:
        _sessions[session_id] = history[-20:]

    return {
        "reply":               final_reply,
        "session_id":          session_id,
        "actions_taken":       actions_taken,
        "suggested_questions": _suggest_follow_ups(message, context),
    }


def clear_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


# ── Financial context builder ──────────────────────────────────────────────────

def _build_financial_context(user_id: int) -> str:
    conn = get_connection()

    user = conn.execute(
        "SELECT username, budget FROM users WHERE id = ?", (user_id,)
    ).fetchone()

    expenses = conn.execute(
        """
        SELECT amount, category, description, date
        FROM expenses
        WHERE user_id = ?
        ORDER BY date DESC
        LIMIT 100
        """,
        (user_id,),
    ).fetchall()

    budgets = conn.execute(
        "SELECT category, limit_amt, month FROM budgets WHERE user_id = ?",
        (user_id,),
    ).fetchall()

    conn.close()

    if not user:
        return "No user data found."

    expense_list = [dict(e) for e in expenses]
    analysis     = analyse_patterns(expense_list, float(user["budget"]))

    # Category totals
    cat_lines = "\n".join(
        f"  • {p['category']}: ₹{p['total_spent']} ({p['percentage']}%, trend: {p['trend']})"
        for p in analysis.get("patterns", [])
    )

    # Recent transactions (last 5)
    recent = "\n".join(
        f"  • {e['date']} | ₹{e['amount']} | {e['category']} | {e['description']}"
        for e in expense_list[:5]
    )

    # Per-category budgets
    budget_lines = "\n".join(
        f"  • {b['category']} ({b['month']}): limit ₹{b['limit_amt']}"
        for b in budgets
    ) or "  No sub-budgets set."

    current_month = datetime.now().strftime("%Y-%m")
    monthly_totals = analysis.get("monthly_totals", {})
    this_month_spend = monthly_totals.get(current_month, 0)

    return f"""
User: {user['username']}
Total Budget: ₹{user['budget']}
Total Spent (all time): ₹{analysis.get('total_spent', 0)}
This Month ({current_month}): ₹{this_month_spend}
Budget Status: {analysis.get('savings_tip', 'N/A')}

Spending by Category:
{cat_lines or '  No expenses yet.'}

Sub-category Budgets:
{budget_lines}

Recent Transactions:
{recent or '  No recent transactions.'}

Anomalies: {', '.join(analysis.get('anomalies', [])) or 'None detected'}
""".strip()


# ── Intent pre-processing ──────────────────────────────────────────────────────

def _pre_process(message: str, user_id: int, context: str) -> tuple[str, list[str]]:
    """Detect intents and enrich the message with live data before Gemini call."""
    lower    = message.lower()
    actions  = []
    enriched = message

    # Categorization intent: "what category is X?" / "categorize X"
    cat_match = re.search(
        r"(?:categori[sz]e|what category|which category)\s+(?:is\s+)?['\"]?(.+?)['\"]?\??$",
        lower,
    )
    if cat_match:
        desc   = cat_match.group(1).strip()
        result = categorizer.predict(desc)
        enriched += (
            f"\n\n[SYSTEM NOTE: ML categorizer result for '{desc}': "
            f"category='{result['category']}', confidence={result['confidence']}]"
        )
        actions.append(f"categorize:'{desc}' → {result['category']} ({result['confidence']*100:.0f}%)")

    # Recommendation intent: "recommend", "suggest", "cheaper", "affordable"
    if any(kw in lower for kw in ["recommend", "suggest", "cheaper", "affordable", "alternative"]):
        # Extract budget hint if present
        budget_match = re.search(r"(?:under|within|below|budget of?)\s*₹?\s*(\d+)", lower)
        budget = float(budget_match.group(1)) if budget_match else 1000.0

        # Extract category hint
        category_keywords = {
            "food": "Food & Dining", "grocery": "Food & Dining", "groceries": "Food & Dining",
            "transport": "Transport", "travel": "Travel",
            "shopping": "Shopping", "clothes": "Shopping",
            "medicine": "Health", "gym": "Health", "health": "Health",
            "entertainment": "Entertainment", "movie": "Entertainment",
            "education": "Education", "course": "Education",
            "utility": "Utilities", "internet": "Utilities", "recharge": "Utilities",
        }
        category = "Shopping"
        for kw, cat in category_keywords.items():
            if kw in lower:
                category = cat
                break

        recs = get_recommendations(user_id, category, budget)
        if recs:
            rec_lines = "\n".join(
                f"  - {r['name']} by {r['brand']}: ₹{r['price']} (save ₹{r['savings']})"
                for r in recs[:4]
            )
            enriched += f"\n\n[SYSTEM NOTE: Top recommendations for {category} under ₹{budget}:\n{rec_lines}]"
            actions.append(f"recommend:{category} under ₹{budget} → {len(recs)} results")

    return enriched, actions


# ── Action executor ────────────────────────────────────────────────────────────

async def _execute_actions(reply: str, user_id: int) -> tuple[str, list[str]]:
    """Execute [ACTION:...] tags emitted by Gemini and replace with results."""
    actions_taken = []
    result_text   = reply

    for match in re.finditer(r"\[ACTION:(\w+):([^\]]+)\]", reply):
        action_type = match.group(1).upper()
        params      = match.group(2).split(":")

        if action_type == "CATEGORIZE":
            desc   = params[0].strip()
            result = categorizer.predict(desc)
            replacement = (
                f"[Category for '{desc}': **{result['category']}** "
                f"(confidence: {result['confidence']*100:.0f}%)]"
            )
            result_text = result_text.replace(match.group(0), replacement)
            actions_taken.append(f"CATEGORIZE: {desc} → {result['category']}")

        elif action_type == "RECOMMEND" and len(params) >= 2:
            category = params[0].strip()
            budget   = float(params[1].strip()) if params[1].strip().isdigit() else 1000.0
            recs     = get_recommendations(user_id, category, budget)
            if recs:
                lines = "\n".join(
                    f"  • {r['name']} — ₹{r['price']} (save ₹{r['savings']})"
                    for r in recs[:4]
                )
                replacement = f"[Recommendations for {category} under ₹{budget}:\n{lines}]"
            else:
                replacement = f"[No recommendations found for {category} under ₹{budget}]"
            result_text = result_text.replace(match.group(0), replacement)
            actions_taken.append(f"RECOMMEND: {category} under ₹{budget}")

    return result_text, actions_taken


# ── Gemini API call ────────────────────────────────────────────────────────────

async def _call_gemini(history: list[dict]) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents":           history,
        "generationConfig": {
            "temperature":     0.7,
            "maxOutputTokens": 512,
            "topP":            0.9,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=20,
        )

    if resp.status_code != 200:
        logger.error(f"[Gemini] HTTP {resp.status_code}: {resp.text[:300]}")
        return "I'm having trouble connecting right now. Please try again in a moment."

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        logger.error(f"[Gemini] Unexpected response shape: {json.dumps(data)[:300]}")
        return "I received an unexpected response. Please try again."


# ── Follow-up suggestions ──────────────────────────────────────────────────────

def _suggest_follow_ups(message: str, context: str) -> list[str]:
    lower = message.lower()
    if "spend" in lower or "total" in lower:
        return [
            "Which category am I overspending in?",
            "How can I reduce my Food & Dining expenses?",
            "Show me my spending trend this month.",
        ]
    if "recommend" in lower or "suggest" in lower:
        return [
            "Find me cheaper grocery options under ₹500.",
            "What's the best mobile recharge plan for my budget?",
            "Suggest affordable transport alternatives.",
        ]
    if "budget" in lower or "save" in lower:
        return [
            "How much have I spent this month?",
            "Set a budget alert for Food & Dining.",
            "What's my biggest unnecessary expense?",
        ]
    return [
        "How much did I spend this month?",
        "What category should I cut back on?",
        "Recommend affordable alternatives for my top expense.",
    ]


# ── Fallback when no API key ───────────────────────────────────────────────────

def _no_key_response() -> dict:
    return {
        "reply": (
            "⚠️ Gemini API key not configured. "
            "Set the GEMINI_API_KEY environment variable to enable the chatbot."
        ),
        "session_id":          "",
        "actions_taken":       [],
        "suggested_questions": [],
    }