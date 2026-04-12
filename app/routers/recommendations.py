from fastapi import APIRouter, Query
from app.models.schemas import RecommendationRequest
from app.ml.recommender import get_recommendations, get_cross_category_recommendations
from app.services.scraper import orchestrator, AmazonScraper, SnapdealScraper
from app.database import get_connection
import httpx, re
from bs4 import BeautifulSoup

router = APIRouter()


@router.get("/debug")
async def debug_scrapers(query: str = Query(default="wireless mouse")):
    """
    Diagnostic endpoint — shows raw HTML structure from Snapdeal and raw hrefs from Amazon.
    Open in browser: http://localhost:8000/api/recommendations/debug?query=wireless+mouse
    """
    out = {"query": query, "snapdeal": {}, "amazon": {}}

    hdrs = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient() as client:
        # ── Snapdeal ──────────────────────────────────────────────────────
        try:
            sd_url  = f"https://www.snapdeal.com/search?keyword={query.replace(' ', '+')}&santizedQuery={query.replace(' ', '+')}"
            sd_resp = await client.get(sd_url, headers=hdrs, timeout=15, follow_redirects=True)
            sd_soup = BeautifulSoup(sd_resp.text, "html.parser")
            sd_cards = (
                sd_soup.select("div.product-tuple-listing") or
                sd_soup.select("div[class*='product-tuple']") or
                []
            )
            out["snapdeal"]["status"]      = sd_resp.status_code
            out["snapdeal"]["cards_found"] = len(sd_cards)
            if sd_cards:
                card = sd_cards[0]
                # Collect all classes containing 'price'
                price_elements = []
                for el in card.find_all(class_=re.compile("price", re.I)):
                    price_elements.append({
                        "tag":   el.name,
                        "class": " ".join(el.get("class", [])),
                        "text":  el.get_text(strip=True)[:80],
                    })
                out["snapdeal"]["first_card_price_elements"] = price_elements
                out["snapdeal"]["first_card_full_text"]      = card.get_text(" ", strip=True)[:400]
                out["snapdeal"]["first_card_html"]           = str(card)[:1500]
            else:
                out["snapdeal"]["page_title"] = sd_soup.title.get_text() if sd_soup.title else "N/A"
                out["snapdeal"]["page_snippet"] = sd_resp.text[:800]
        except Exception as e:
            out["snapdeal"]["error"] = str(e)

        # ── Amazon ────────────────────────────────────────────────────────
        try:
            az_url  = f"https://www.amazon.in/s?k={query.replace(' ', '+')}&i=aps"
            az_resp = await client.get(az_url, headers=hdrs, timeout=15, follow_redirects=True)
            az_soup = BeautifulSoup(az_resp.text, "html.parser")
            az_cards = az_soup.select('[data-component-type="s-search-result"]')
            out["amazon"]["status"]      = az_resp.status_code
            out["amazon"]["cards_found"] = len(az_cards)
            hrefs = []
            for card in az_cards[:5]:
                a = card.select_one("h2 a")
                if a:
                    href = a.get("href", "")
                    is_sspa = "/sspa/" in href
                    sspa_m  = re.search(r"[?&]url=(%2F[^&]+)", href)
                    from urllib.parse import unquote
                    decoded = unquote(sspa_m.group(1)) if sspa_m else ""
                    final   = ("https://www.amazon.in" + decoded) if decoded else (href if href.startswith("http") else "https://www.amazon.in" + href)
                    hrefs.append({"is_sspa": is_sspa, "raw_href": href[:150], "decoded": decoded[:150], "final_url": final[:150]})
            out["amazon"]["first_5_hrefs"] = hrefs
        except Exception as e:
            out["amazon"]["error"] = str(e)

    return out


@router.post("/")
async def recommend_products(req: RecommendationRequest):
    """
    Get product recommendations for a specific category within a budget.
    Tries live scraping first; falls back to static catalogue if scraping yields nothing.
    """
    avg_spend = _avg_spend(req.user_id, req.category)
    query     = orchestrator.category_to_query(req.category)

    live_results = await orchestrator.search(
        query=query,
        category=req.category,
        max_budget=req.max_budget,
        avg_spend=avg_spend,
    )

    if live_results:
        return {"source": "live", "results": live_results}

    # Fallback to static catalogue
    static = get_recommendations(req.user_id, req.category, req.max_budget)
    return {"source": "static", "results": static}


@router.get("/search")
async def search_products(
    query:      str   = Query(..., description="Search keyword e.g. 'rice 5kg'"),
    category:   str   = Query("Other"),
    max_budget: float = Query(..., gt=0),
    user_id:    int   = Query(...),
    min_price:  float = Query(0.0, ge=0),
):
    """
    Free-text product search across all four platforms simultaneously.
    min_price: floor price — used when user has budget to spend on better products.
    """
    avg_spend    = _avg_spend(user_id, category)
    live_results = await orchestrator.search(
        query=query,
        category=category,
        max_budget=max_budget,
        avg_spend=avg_spend,
        min_price=min_price,
    )
    return {
        "query":   query,
        "count":   len(live_results),
        "results": live_results,
    }


@router.get("/smart/{user_id}")
async def smart_recommendations(user_id: int):
    """
    Automatically finds overspend categories and fetches live cheaper alternatives
    from all four platforms.
    """
    conn = get_connection()
    user = conn.execute("SELECT budget FROM users WHERE id = ?", (user_id,)).fetchone()
    rows = conn.execute(
        """
        SELECT category, SUM(amount) AS total
        FROM expenses WHERE user_id = ?
        GROUP BY category ORDER BY total DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    if not user:
        return {"recommendations": [], "message": "User not found"}

    budget      = float(user["budget"])
    per_cat_bud = budget / max(len(rows), 1)

    all_recs = []
    for row in rows:
        cat   = row["category"]
        spent = float(row["total"])
        if spent > per_cat_bud:
            query   = orchestrator.category_to_query(cat)
            results = await orchestrator.search(
                query=query,
                category=cat,
                max_budget=per_cat_bud,
                avg_spend=spent,
            )
            all_recs.extend(results[:3])   # top 3 per overspent category

    all_recs.sort(key=lambda x: -x.get("savings", 0))

    return {
        "source":          "live",
        "recommendations": all_recs[:10],
        "count":           len(all_recs[:10]),
    }


# ── helper ────────────────────────────────────────────────────────────────────

def _avg_spend(user_id: int, category: str) -> float:
    conn = get_connection()
    row  = conn.execute(
        "SELECT AVG(amount) AS avg FROM expenses WHERE user_id = ? AND category = ?",
        (user_id, category),
    ).fetchone()
    conn.close()
    return float(row["avg"]) if row and row["avg"] else 0.0