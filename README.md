# SmartSpend AI — Backend

FastAPI backend for the Smart Expense Tracker with AI-powered categorization,
spending pattern analysis, and budget-aware recommendations.

## Project Structure

```
smartspend-ai/
├── app/
│   ├── main.py               # FastAPI app + CORS + router registration
│   ├── database.py           # SQLite setup & connection helper
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── routers/
│   │   ├── expenses.py       # CRUD + NLP categorization endpoints
│   │   ├── recommendations.py# Budget-aware product recommendations
│   │   └── analytics.py      # Spending patterns & insights
│   ├── ml/
│   │   ├── categorizer.py    # TF-IDF + Logistic Regression classifier
│   │   ├── pattern_analyser.py # Z-score anomaly + trend detection
│   │   └── recommender.py    # Budget-aware recommendation engine
│   └── services/             # (scraper / Gemini chatbot go here)
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/expenses/` | Add expense (auto-categorized) |
| GET  | `/api/expenses/{user_id}` | List user's expenses |
| POST | `/api/expenses/categorize` | Predict category for text |
| POST | `/api/expenses/retrain` | Retrain ML model with new labels |
| POST | `/api/expenses/budgets/` | Set category budget |
| POST | `/api/recommendations/` | Get products for category + budget |
| GET  | `/api/recommendations/smart/{user_id}` | Cross-category smart recs |
| GET  | `/api/analytics/{user_id}/insights` | Full spending analysis |
| GET  | `/api/analytics/{user_id}/summary` | Dashboard summary |

---

## ML Components

### 1. Expense Categorizer (`ml/categorizer.py`)
- **Algorithm**: TF-IDF (bigrams) → Logistic Regression (multinomial)
- **Categories**: Food & Dining, Transport, Shopping, Utilities, Health, Entertainment, Education, Travel, Other
- **Auto-trains** on a seed dataset at first run, saves model to disk
- **Retrainable** via `/api/expenses/retrain` with user-corrected labels

### 2. Pattern Analyser (`ml/pattern_analyser.py`)
- Per-category breakdown with % share of total spend
- Month-over-month trend detection (increasing / stable / decreasing)
- **Anomaly detection** via Z-score (flags transactions > 2.5σ from mean)
- Auto-generated personalised savings tip

### 3. Recommendation Engine (`ml/recommender.py`)
- Filters product catalogue by category + budget
- Ranks by **savings vs user's average spend** in that category
- `smart_recommendations` endpoint finds overspend categories automatically
- Replace `PRODUCT_CATALOG` with live web-scraped data in production

---

## Next Steps

- [ ] Add Gemini API chatbot (`app/services/gemini_chat.py`)
- [ ] Add web scraper for live product data (`app/services/scraper.py`)
- [ ] Add JWT authentication
- [ ] Connect to React frontend
