from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.gemini_chat import chat, clear_session

router = APIRouter()


class ChatMessage(BaseModel):
    user_id:    int
    message:    str
    session_id: str = "default"


class ClearSession(BaseModel):
    session_id: str = "default"


@router.post("/")
async def send_message(body: ChatMessage):
    """
    Send a message to SmartSpend AI (TinyLlama local chatbot).

    Capabilities:
    - Spending history Q&A (from live DB context)
    - Budget advice & savings tips
    - Expense categorization (via ML model)
    - Product recommendations (via recommender engine)

    Multi-turn conversation maintained per session_id.
    Falls back to rule-based answers if model is not loaded yet.
    """
    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    return await chat(
        user_id=body.user_id,
        session_id=body.session_id,
        message=body.message.strip(),
    )


@router.delete("/session")
def end_session(session_id: str = "default"):
    """Clear conversation history for a session."""
    cleared = clear_session(session_id)
    return {"cleared": cleared, "session_id": session_id}


@router.get("/starters/{user_id}")
def conversation_starters(user_id: int):
    """Return suggested opening questions."""
    return {
        "starters": [
            "How much have I spent this month?",
            "Which category am I overspending in?",
            "Give me tips to reduce my expenses.",
            "Categorize this: bought coffee and a sandwich.",
            "Recommend affordable groceries under ₹500.",
            "Am I on track with my budget?",
            "What's my biggest expense category?",
            "Where can I cut back this month?",
        ]
    }