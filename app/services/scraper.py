"""
Live Product Scraper — Amazon India, Flipkart, Snapdeal, Meesho

Architecture:
  - Each site has its own scraper class inheriting from BaseScraper.
  - ScraperOrchestrator runs all scrapers concurrently (asyncio) and
    merges results, deduplicating by (name, brand).
  - Results are returned live on every request.
  - Rotating User-Agents + randomised delays prevent trivial blocking.
  - If a site is down or blocks us, it fails silently and the others
    still return results.

Usage:
    from app.services.scraper import orchestrator
    products = await orchestrator.search(query="rice 5kg", category="Food & Dining", max_budget=500)
"""

import asyncio
import re
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScrapedProduct:
    name:     str
    brand:    str
    price:    float
    category: str
    source:   str          # "amazon" | "flipkart" | "snapdeal" | "meesho"
    url:      str  = ""
    image:    str  = ""
    rating:   float = 0.0
    savings:  float = 0.0
    reason:   str   = ""


# ── Shared helpers ─────────────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

def _headers() -> dict:
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT":             "1",
        "Connection":      "keep-alive",
    }

def _parse_price(text: str) -> float:
    """Extract numeric price from strings like '₹1,299', 'Rs. 450', '1299.00', 'Rs.1.299'."""
    # Step 1: strip commas used as thousand separators
    text = text.replace(",", "")
    # Step 2: find the first continuous run of digits (with at most one dot)
    # This handles: "₹1299.00 onwards", "Rs.1299", "1 299 ₹" etc.
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 0.0
    try:
        val = float(match.group())
        return val if val > 0 else 0.0
    except ValueError:
        return 0.0

def _extract_brand(name: str) -> str:
    """Heuristically extract brand from product name (first 1–2 words)."""
    words = name.strip().split()
    return words[0] if words else "Unknown"


# ── Base scraper ───────────────────────────────────────────────────────────────

class BaseScraper(ABC):
    SOURCE: str = ""

    async def fetch_html(self, url: str, client: httpx.AsyncClient) -> str | None:
        await asyncio.sleep(random.uniform(0.5, 1.5))   # polite delay
        try:
            resp = await client.get(url, headers=_headers(), timeout=12, follow_redirects=True)
            if resp.status_code == 200:
                return resp.text
            logger.warning(f"[{self.SOURCE}] HTTP {resp.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[{self.SOURCE}] Fetch error: {e}")
        return None

    @abstractmethod
    async def search(
        self,
        query: str,
        category: str,
        max_budget: float,
        client: httpx.AsyncClient,
        min_price: float = 0.0,
    ) -> list[ScrapedProduct]:
        ...


# ── Amazon India ───────────────────────────────────────────────────────────────

class AmazonScraper(BaseScraper):
    SOURCE = "amazon"
    BASE   = "https://www.amazon.in/s"

    async def search(self, query, category, max_budget, client, min_price=0.0):
        url  = f"{self.BASE}?k={query.replace(' ', '+')}&i=aps"
        html = await self.fetch_html(url, client)
        if not html:
            return []

        soup     = BeautifulSoup(html, "html.parser")
        products = []

        for card in soup.select('[data-component-type="s-search-result"]')[:10]:
            try:
                # Name
                name_tag = card.select_one("h2 span")
                if not name_tag:
                    continue
                name = name_tag.get_text(strip=True)

                # Price — try whole + fraction first, fall back to any price span
                whole    = card.select_one(".a-price-whole")
                fraction = card.select_one(".a-price-fraction")
                if whole:
                    price_str = whole.get_text(strip=True).replace(",", "")
                    if fraction:
                        price_str += "." + fraction.get_text(strip=True)
                    price = _parse_price(price_str)
                else:
                    price_tag = card.select_one(".a-price .a-offscreen")
                    price     = _parse_price(price_tag.get_text()) if price_tag else 0.0

                if price <= 0 or price > max_budget or price < min_price:
                    continue

                # URL — Amazon's h2 a often has empty href; the real link is on
                # a-link-normal wrapping the title, or the data-asin attribute
                url_ = ""
                asin = card.get("data-asin", "")

                # Try selectors in order of reliability
                for link_sel in [
                    "a.a-link-normal.s-underline-text",   # standard title link
                    "a.a-link-normal[href*='/dp/']",       # direct product page
                    "a.a-link-normal[href*='/sspa/']",     # sponsored
                    "h2 a[href]",                          # original attempt
                    "a[href*='/dp/']",                     # any dp link
                ]:
                    a_tag = card.select_one(link_sel)
                    if a_tag:
                        href = a_tag.get("href", "").strip()
                        if href:
                            break
                else:
                    href = ""

                if href:
                    # Decode sspa redirect: /sspa/click?...&url=%2Fdp%2FB0xxx
                    sspa_match = re.search(r"[?&]url=(%2F[^&]+)", href)
                    if sspa_match:
                        from urllib.parse import unquote
                        href = unquote(sspa_match.group(1))
                    url_ = href if href.startswith("http") else "https://www.amazon.in" + href
                elif asin:
                    # Last resort: construct URL from ASIN on the card's data attribute
                    url_ = f"https://www.amazon.in/dp/{asin}"

                # Rating
                rating_tag = card.select_one("span.a-icon-alt")
                rating     = float(rating_tag.get_text().split()[0]) if rating_tag else 0.0

                # Image
                img_tag = card.select_one("img.s-image")
                image   = img_tag["src"] if img_tag else ""

                products.append(ScrapedProduct(
                    name=name, brand=_extract_brand(name),
                    price=price, category=category,
                    source=self.SOURCE, url=url_,
                    image=image, rating=rating,
                ))
            except Exception as e:
                logger.debug(f"[amazon] card parse error: {e}")

        return products


# ── Flipkart ───────────────────────────────────────────────────────────────────

class FlipkartScraper(BaseScraper):
    SOURCE = "flipkart"
    BASE   = "https://www.flipkart.com/search"

    async def search(self, query, category, max_budget, client, min_price=0.0):
        url  = f"{self.BASE}?q={query.replace(' ', '%20')}"
        html = await self.fetch_html(url, client)
        if not html:
            return []

        soup     = BeautifulSoup(html, "html.parser")
        products = []

        # Flipkart uses multiple card layouts; try both
        cards = soup.select("div._1AtVbE") or soup.select("div._2kHMtA")

        for card in cards[:12]:
            try:
                name_tag = (
                    card.select_one("div._4rR01T") or
                    card.select_one("a.s1Q9rs")    or
                    card.select_one("div.IRpwTa")
                )
                if not name_tag:
                    continue
                name = name_tag.get_text(strip=True)
                if not name:
                    continue

                price_tag = (
                    card.select_one("div._30jeq3") or
                    card.select_one("div._1_WHN1")
                )
                if not price_tag:
                    continue
                price = _parse_price(price_tag.get_text())

                if price <= 0 or price > max_budget or price < min_price:
                    continue

                a_tag = card.select_one("a[href]")
                url_  = "https://www.flipkart.com" + a_tag["href"] if a_tag else ""

                rating_tag = card.select_one("div._3LWZlK")
                rating     = float(rating_tag.get_text()) if rating_tag else 0.0

                img_tag = card.select_one("img._396cs4") or card.select_one("img._2r_T1I")
                image   = img_tag["src"] if img_tag else ""

                products.append(ScrapedProduct(
                    name=name, brand=_extract_brand(name),
                    price=price, category=category,
                    source=self.SOURCE, url=url_,
                    image=image, rating=rating,
                ))
            except Exception as e:
                logger.debug(f"[flipkart] card parse error: {e}")

        return products


# ── Snapdeal ───────────────────────────────────────────────────────────────────

class SnapdealScraper(BaseScraper):
    SOURCE = "snapdeal"
    BASE   = "https://www.snapdeal.com/search"

    # Price pattern: ₹ or Rs followed by digits (with optional commas/dots)
    _PRICE_RE = re.compile(r"(?:₹|Rs\.?)\s*([\d,]+(?:\.\d+)?)")

    async def search(self, query, category, max_budget, client, min_price=0.0):
        url  = f"{self.BASE}?keyword={query.replace(' ', '+')}&santizedQuery={query.replace(' ', '+')}"
        html = await self.fetch_html(url, client)
        if not html:
            return []

        soup     = BeautifulSoup(html, "html.parser")
        products = []

        # ── Find product cards (try multiple known selectors) ──────────────
        cards = (
            soup.select("div.product-tuple-listing") or
            soup.select("div[class*='product-tuple']") or
            soup.select("li.product") or
            []
        )
        logger.info(f"[snapdeal] found {len(cards)} cards for '{query}'")

        for card in cards[:10]:
            try:
                # ── Name ──────────────────────────────────────────────────
                name_tag = (
                    card.select_one("p.product-title") or
                    card.select_one("[class*='product-title']") or
                    card.select_one("a[title]")
                )
                if not name_tag:
                    continue
                name = (name_tag.get("title") or name_tag.get_text(strip=True)).strip()
                if not name:
                    continue

                # ── Price — read data-price attribute first (most reliable) ──
                price_tag = card.select_one("span[data-price]")
                if price_tag and price_tag.get("data-price"):
                    try:
                        price = float(price_tag["data-price"])
                    except ValueError:
                        price = _parse_price(price_tag.get_text(strip=True))
                else:
                    # Fallback: class-selector cascade
                    for sel in [
                        "span.lfloat.product-price", "span.product-price",
                        "div.product-price", "[class*='product-price']",
                        "span.price", "div.price",
                    ]:
                        tag = card.select_one(sel)
                        if tag:
                            price = _parse_price(tag.get_text(strip=True))
                            if price > 0:
                                break
                    # Last resort: regex scan entire card text
                    if price <= 0:
                        m = self._PRICE_RE.search(card.get_text(" ", strip=True))
                        price = _parse_price(m.group(0)) if m else 0.0

                logger.info(f"[snapdeal] price={price} for '{name[:40]}'")

                if price <= 0 or price > max_budget or price < min_price:
                    continue

                # ── MRP for savings calc ──────────────────────────────────
                mrp_tag = card.select_one("span.product-desc-price") or card.select_one("span.strike")
                mrp = _parse_price(mrp_tag.get_text(strip=True)) if mrp_tag else 0.0

                # ── URL ───────────────────────────────────────────────────
                a_tag = (
                    card.select_one("a.dp-widget-link") or
                    card.select_one("a[href*='snapdeal.com/products']") or
                    card.select_one("a[href*='snapdeal.com']") or
                    card.select_one("a[href]")
                )
                href = a_tag.get("href", "") if a_tag else ""
                if href.startswith("http"):
                    url_ = href
                elif href.startswith("/"):
                    url_ = "https://www.snapdeal.com" + href
                else:
                    url_ = ""

                # ── Image ─────────────────────────────────────────────────
                img_tag = card.select_one("img.product-image") or card.select_one("img[src]")
                image   = ""
                if img_tag:
                    image = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-lazy-src") or ""

                savings = round(mrp - price, 2) if mrp > price else 0.0
                products.append(ScrapedProduct(
                    name=name, brand=_extract_brand(name),
                    price=price, category=category,
                    source=self.SOURCE, url=url_,
                    image=image,
                    savings=savings,
                    reason=f"₹{savings:.0f} off MRP on Snapdeal" if savings > 0 else "Good deal on Snapdeal",
                ))
            except Exception as e:
                logger.warning(f"[snapdeal] card parse error: {e}")

        return products


# ── Meesho ─────────────────────────────────────────────────────────────────────

class MeeshoScraper(BaseScraper):
    """
    Meesho is a React SPA — the HTML response contains a __NEXT_DATA__ JSON
    block with product data; we extract that instead of scraping the DOM.
    """
    SOURCE = "meesho"
    BASE   = "https://www.meesho.com/search"

    async def search(self, query, category, max_budget, client, min_price=0.0):
        url  = f"{self.BASE}?q={query.replace(' ', '%20')}"
        html = await self.fetch_html(url, client)
        if not html:
            return []

        products = []
        soup     = BeautifulSoup(html, "html.parser")

        # Try JSON extraction from __NEXT_DATA__
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if script:
            import json
            try:
                data    = json.loads(script.string)
                # Path may vary; try to locate product listings
                catalog = (
                    data.get("props", {})
                        .get("pageProps", {})
                        .get("searchResultsData", {})
                        .get("products", [])
                )
                for item in catalog[:10]:
                    price = float(item.get("price", {}).get("mrp", 0))
                    if price <= 0 or price > max_budget or price < min_price:
                        continue
                    name = item.get("name", "")
                    products.append(ScrapedProduct(
                        name=name,
                        brand=item.get("supplier_name", _extract_brand(name)),
                        price=price,
                        category=category,
                        source=self.SOURCE,
                        url=f"https://www.meesho.com{item.get('url', '')}",
                        image=item.get("images", [{}])[0].get("url", ""),
                        rating=float(item.get("ratings", {}).get("average", 0)),
                    ))
                return products
            except Exception as e:
                logger.debug(f"[meesho] JSON parse error: {e}")

        # Fallback: DOM scraping
        for card in soup.select("div[class*='ProductCard']")[:10]:
            try:
                name_tag  = card.select_one("p[class*='ProductTitle']")
                price_tag = card.select_one("h5[class*='price']") or card.select_one("span[class*='price']")
                if not name_tag or not price_tag:
                    continue
                name  = name_tag.get_text(strip=True)
                price = _parse_price(price_tag.get_text())
                if price <= 0 or price > max_budget or price < min_price:
                    continue
                img_tag = card.select_one("img")
                image   = img_tag.get("src", "") if img_tag else ""
                products.append(ScrapedProduct(
                    name=name, brand=_extract_brand(name),
                    price=price, category=category,
                    source=self.SOURCE, image=image,
                ))
            except Exception as e:
                logger.debug(f"[meesho] card parse error: {e}")

        return products


# ── Orchestrator ───────────────────────────────────────────────────────────────

class ScraperOrchestrator:
    """Runs all scrapers concurrently and merges deduplicated results."""

    def __init__(self):
        self.scrapers: list[BaseScraper] = [
            AmazonScraper(),
            FlipkartScraper(),
            SnapdealScraper(),
            MeeshoScraper(),
        ]

    async def search(
        self,
        query:      str,
        category:   str,
        max_budget: float,
        avg_spend:  float = 0.0,
        min_price:  float = 0.0,
    ) -> list[dict]:
        """
        Parameters
        ----------
        query      : search keyword derived from category / user input
        category   : expense category for tagging results
        max_budget : upper price limit
        avg_spend  : user's average spend in this category (for savings calc)

        Returns
        -------
        List of product dicts sorted by savings desc, then price asc.
        """
        async with httpx.AsyncClient() as client:
            tasks   = [s.search(query, category, max_budget, client, min_price) for s in self.scrapers]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[ScrapedProduct] = []
        seen:   set[str]             = set()

        for batch in results:
            if isinstance(batch, Exception):
                logger.warning(f"Scraper task failed: {batch}")
                continue
            for p in batch:
                key = f"{p.name[:40].lower()}_{p.price}"
                if key not in seen:
                    seen.add(key)
                    if avg_spend > p.price:
                        p.savings = round(avg_spend - p.price, 2)
                        pct       = round(p.savings / avg_spend * 100)
                        p.reason  = f"Save ₹{p.savings:.0f} ({pct}% less than your usual spend)"
                    else:
                        if min_price > 0:
                            p.reason = f"Premium pick from {p.brand} within your ₹{max_budget:.0f} budget"
                        else:
                            p.reason = f"Affordable option from {p.brand} within your budget"
                    merged.append(p)

        # Sort: if min_price set (user has budget), prefer higher-priced items first
        if min_price > 0:
            merged.sort(key=lambda x: (-x.price, -x.rating))
        else:
            merged.sort(key=lambda x: (-x.savings, x.price))

        return [
            {
                "name":     p.name,
                "brand":    p.brand,
                "price":    p.price,
                "category": p.category,
                "source":   p.source,
                "url":      p.url,
                "image":    p.image,
                "rating":   p.rating,
                "savings":  p.savings,
                "reason":   p.reason,
            }
            for p in merged[:12]   # top 12 across all sites
        ]

    # ── Helper: derive a good search query from category + context ──────────

    @staticmethod
    def category_to_query(category: str, hint: str = "") -> str:
        """
        Convert an expense category into a practical search query.
        e.g. "Food & Dining" → "grocery essentials"
        """
        defaults = {
            "Food & Dining":  "grocery essentials",
            "Transport":      "travel pass commute",
            "Shopping":       "daily essentials",
            "Utilities":      "mobile recharge plan",
            "Health":         "health supplement medicine",
            "Entertainment":  "ott subscription",
            "Education":      "online course",
            "Travel":         "budget hotel travel",
            "Other":          "everyday products",
        }
        base = defaults.get(category, category.lower())
        return f"{hint} {base}".strip() if hint else base


# Singleton
orchestrator = ScraperOrchestrator()