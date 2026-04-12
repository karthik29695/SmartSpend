"""
Run this from inside your SmartSpend/ folder:
    cd SmartSpend
    python debug_scrapers.py

It will:
1. Fetch Snapdeal live and print the first card's HTML + every price-related class found
2. Fetch Amazon live and print the first card's link href so you can see if it's sspa or not
"""

import asyncio
import re
import random
import httpx
from bs4 import BeautifulSoup

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

def headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
    }

QUERY = "wireless mouse"

async def debug_snapdeal(client):
    print("\n" + "="*60)
    print("SNAPDEAL DEBUG")
    print("="*60)
    url = f"https://www.snapdeal.com/search?keyword={QUERY.replace(' ', '+')}&santizedQuery={QUERY.replace(' ', '+')}"
    print(f"URL: {url}")
    try:
        resp = await client.get(url, headers=headers(), timeout=15, follow_redirects=True)
        print(f"Status: {resp.status_code}  |  Content-Length: {len(resp.text)}")
    except Exception as e:
        print(f"FETCH ERROR: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Find product cards ──────────────────────────────────────────────────
    cards = soup.select("div.product-tuple-listing")
    print(f"Cards (div.product-tuple-listing): {len(cards)}")

    if not cards:
        # Try alternate selectors
        for sel in ["div[class*='product-tuple']", "div[class*='product-card']", "li.product", "div.product"]:
            alt = soup.select(sel)
            if alt:
                print(f"  → Alt selector {sel!r} found {len(alt)} cards")
                cards = alt[:3]
                break
        if not cards:
            print("  → No product cards found at all. First 4000 chars of body:")
            print(resp.text[:4000])
            return

    card = cards[0]

    # ── Print all classes that contain 'price' ──────────────────────────────
    print("\n--- All elements with 'price' in class name ---")
    for el in card.find_all(class_=re.compile("price", re.I)):
        cls = " ".join(el.get("class", []))
        txt = el.get_text(strip=True)[:80]
        print(f"  <{el.name} class='{cls}'> → '{txt}'")

    # ── Try every selector we have ──────────────────────────────────────────
    print("\n--- Selector probe ---")
    for sel in [
        "span.product-price", "div.product-price", "span.lfloat.product-price",
        "[class*='product-price']", "span.price", "div.price", "span[class*='price']",
        "div[class*='price']",
    ]:
        tag = card.select_one(sel)
        print(f"  {sel!r:40s} → {repr(tag.get_text(strip=True)[:60]) if tag else None}")

    # ── Print name ──────────────────────────────────────────────────────────
    name_tag = card.select_one("p.product-title")
    print(f"\n--- Product title selector 'p.product-title' → {repr(name_tag.get_text(strip=True)[:80]) if name_tag else None}")

    print("\n--- First card raw HTML (first 2500 chars) ---")
    print(str(card)[:2500])


async def debug_amazon(client):
    print("\n" + "="*60)
    print("AMAZON DEBUG")
    print("="*60)
    url = f"https://www.amazon.in/s?k={QUERY.replace(' ', '+')}&i=aps"
    print(f"URL: {url}")
    try:
        resp = await client.get(url, headers=headers(), timeout=15, follow_redirects=True)
        print(f"Status: {resp.status_code}  |  Content-Length: {len(resp.text)}")
    except Exception as e:
        print(f"FETCH ERROR: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    cards = soup.select('[data-component-type="s-search-result"]')
    print(f"Cards: {len(cards)}")

    for i, card in enumerate(cards[:5]):
        a_tag = card.select_one("h2 a")
        href  = a_tag.get("href", "") if a_tag else ""
        name_tag = card.select_one("h2 span")
        name = name_tag.get_text(strip=True)[:60] if name_tag else "—"

        # Decode sspa
        sspa_match = re.search(r"[?&]url=(%2F[^&]+)", href)
        decoded = ""
        if sspa_match:
            from urllib.parse import unquote
            decoded = unquote(sspa_match.group(1))

        is_sspa = "/sspa/" in href
        print(f"\n  [{i}] {name}")
        print(f"       href type : {'SSPA' if is_sspa else 'normal'}")
        print(f"       raw href  : {href[:120]}")
        if decoded:
            print(f"       decoded   : {decoded[:120]}")
        final_url = ("https://www.amazon.in" + decoded) if decoded else (href if href.startswith("http") else "https://www.amazon.in" + href)
        print(f"       final url : {final_url[:120]}")


async def main():
    async with httpx.AsyncClient() as client:
        await debug_snapdeal(client)
        await debug_amazon(client)

if __name__ == "__main__":
    asyncio.run(main())