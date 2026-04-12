from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import expenses, recommendations, analytics, chat
from app.database import init_db

app = FastAPI(
    title="SmartSpend AI",
    description="AI-powered expense tracking with categorization, pattern analysis, and recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    init_db()

app.include_router(expenses.router,        prefix="/api/expenses",        tags=["Expenses"])
app.include_router(recommendations.router, prefix="/api/recommendations", tags=["Recommendations"])
app.include_router(analytics.router,       prefix="/api/analytics",       tags=["Analytics"])
app.include_router(chat.router,          prefix="/api/chat",          tags=["Chatbot"])

@app.get("/")
def root():
    return {"message": "SmartSpend AI is running", "docs": "/docs"}