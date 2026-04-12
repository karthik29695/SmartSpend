"""
Expense Categorization using NLP + ML.

Pipeline:
  1. Keyword override — unambiguous terms are matched instantly (no ML needed)
  2. TF-IDF vectorizer converts description text → feature vector
  3. Logistic Regression classifier predicts category
  4. If ML confidence < threshold, fall back to "Other"

Categories:
  Food & Dining | Transport | Shopping | Utilities | Health |
  Entertainment | Education | Travel | Personal Care | Electronics | Other
"""

import os
import re
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MODEL_PATH = os.getenv("MODEL_PATH", "expense_classifier.pkl")

# ── Keyword override map ───────────────────────────────────────────────────────
KEYWORD_OVERRIDES: dict[str, str] = {
    # Personal Care
    "face wash": "Personal Care", "facewash": "Personal Care",
    "shampoo": "Personal Care", "conditioner": "Personal Care",
    "moisturizer": "Personal Care", "moisturiser": "Personal Care",
    "sunscreen": "Personal Care", "sunblock": "Personal Care",
    "body wash": "Personal Care", "bodywash": "Personal Care",
    "body lotion": "Personal Care", "lotion": "Personal Care",
    "face cream": "Personal Care", "night cream": "Personal Care",
    "serum": "Personal Care", "toner": "Personal Care",
    "scrub": "Personal Care", "face mask": "Personal Care",
    "lip balm": "Personal Care", "chapstick": "Personal Care",
    "deodorant": "Personal Care", "deo": "Personal Care",
    "perfume": "Personal Care", "cologne": "Personal Care",
    "toothpaste": "Personal Care", "toothbrush": "Personal Care",
    "mouthwash": "Personal Care", "floss": "Personal Care",
    "razor": "Personal Care", "shaving cream": "Personal Care",
    "hair oil": "Personal Care", "hair gel": "Personal Care",
    "hair color": "Personal Care", "hair colour": "Personal Care",
    "hair dye": "Personal Care", "hair serum": "Personal Care",
    "nail polish": "Personal Care", "nail paint": "Personal Care",
    "makeup": "Personal Care", "foundation": "Personal Care",
    "lipstick": "Personal Care", "mascara": "Personal Care",
    "eyeliner": "Personal Care", "kajal": "Personal Care",
    "eyeshadow": "Personal Care", "blush": "Personal Care",
    "compact": "Personal Care",
    "sanitary pad": "Personal Care", "tampon": "Personal Care",
    "menstrual cup": "Personal Care",
    "soap": "Personal Care", "hand wash": "Personal Care",
    "sanitizer": "Personal Care", "wet wipe": "Personal Care",
    "cotton pad": "Personal Care", "q tip": "Personal Care",
    "beard oil": "Personal Care", "beard balm": "Personal Care",
    "salon": "Personal Care", "parlour": "Personal Care",
    "haircut": "Personal Care", "waxing": "Personal Care",
    "threading": "Personal Care", "spa": "Personal Care",
    # Electronics
    "laptop": "Electronics", "macbook": "Electronics",
    "smartphone": "Electronics", "iphone": "Electronics",
    "earbuds": "Electronics", "airpods": "Electronics",
    "headphone": "Electronics", "earphone": "Electronics",
    "smartwatch": "Electronics", "smart watch": "Electronics",
    "power bank": "Electronics", "charger": "Electronics",
    "hard disk": "Electronics", "ssd": "Electronics",
    "pen drive": "Electronics", "usb drive": "Electronics",
    "monitor": "Electronics", "keyboard": "Electronics",
    "webcam": "Electronics", "printer": "Electronics",
    "router": "Electronics",
    "smart tv": "Electronics", "led tv": "Electronics",
    "gaming console": "Electronics", "playstation": "Electronics",
    "refrigerator": "Electronics", "fridge": "Electronics",
    "washing machine": "Electronics", "microwave": "Electronics",
    "mixer grinder": "Electronics", "water purifier": "Electronics",
    "air conditioner": "Electronics",
    "bluetooth speaker": "Electronics",
    "dslr": "Electronics", "mirrorless camera": "Electronics",
    # Shopping — brands & clothing
    "zudio": "Shopping", "westside": "Shopping", "max fashion": "Shopping",
    "h&m": "Shopping", "zara": "Shopping", "mango": "Shopping",
    "uniqlo": "Shopping", "myntra": "Shopping", "ajio": "Shopping",
    "nykaa fashion": "Shopping", "fabindia": "Shopping", "biba": "Shopping",
    "pantaloons": "Shopping", "shoppers stop": "Shopping",
    "lifestyle store": "Shopping", "reliance trends": "Shopping",
    "v mart": "Shopping", "lenskart": "Shopping",
    "clothes": "Shopping", "clothing": "Shopping",
    "shirt": "Shopping", "tshirt": "Shopping", "t-shirt": "Shopping",
    "jeans": "Shopping", "trousers": "Shopping", "kurta": "Shopping",
    "saree": "Shopping", "kurti": "Shopping", "lehenga": "Shopping",
    "dress": "Shopping", "skirt": "Shopping", "jacket": "Shopping",
    "hoodie": "Shopping", "sweater": "Shopping", "leggings": "Shopping",
    "pyjama": "Shopping", "shorts": "Shopping",
    "innerwear": "Shopping", "underwear": "Shopping", "socks": "Shopping",
    "sneakers": "Shopping", "sandals": "Shopping", "chappal": "Shopping",
    "heels": "Shopping", "boots": "Shopping", "footwear": "Shopping",
    "meesho": "Shopping",
    # Food & Dining
    "zomato": "Food & Dining", "swiggy": "Food & Dining",
    "blinkit": "Food & Dining", "zepto": "Food & Dining",
    "restaurant": "Food & Dining", "cafe": "Food & Dining",
    "grocery": "Food & Dining", "groceries": "Food & Dining",
    "supermarket": "Food & Dining", "dmart": "Food & Dining",
    # Transport
    "uber": "Transport", "ola cab": "Transport", "rapido": "Transport",
    "petrol": "Transport", "diesel": "Transport",
    "metro card": "Transport", "bus pass": "Transport",
    "irctc": "Transport", "fastag": "Transport",
    # Utilities
    "electricity bill": "Utilities", "internet bill": "Utilities",
    "mobile recharge": "Utilities", "broadband": "Utilities",
    "netflix": "Utilities", "hotstar": "Utilities",
    "amazon prime": "Utilities", "spotify": "Utilities",
    "youtube premium": "Utilities", "lpg": "Utilities",
    # Health
    "medicine": "Health", "pharmacy": "Health",
    "doctor": "Health", "hospital": "Health",
    "gym membership": "Health", "blood test": "Health",
    "dental": "Health", "physiotherapy": "Health",
    # Entertainment
    "movie ticket": "Entertainment", "pvr": "Entertainment",
    "inox": "Entertainment", "concert": "Entertainment",
    "bookmyshow": "Entertainment",
    # Education
    "udemy": "Education", "coursera": "Education",
    "tuition fee": "Education", "school fee": "Education",
    "college fee": "Education", "exam fee": "Education",
    # Travel
    "hotel booking": "Travel", "oyo": "Travel",
    "flight ticket": "Travel", "airbnb": "Travel",
    "holiday package": "Travel", "visa fee": "Travel",
}

# ── Seed training data ─────────────────────────────────────────────────────────
SEED_DATA = [
    # Food & Dining
    ("bought groceries at dmart", "Food & Dining"),
    ("weekly vegetable shopping", "Food & Dining"),
    ("dinner at mcdonalds", "Food & Dining"),
    ("coffee from starbucks", "Food & Dining"),
    ("pizza order online", "Food & Dining"),
    ("supermarket shopping big bazaar", "Food & Dining"),
    ("restaurant bill with family", "Food & Dining"),
    ("lunch at subway", "Food & Dining"),
    ("zomato food delivery order", "Food & Dining"),
    ("swiggy order chicken biryani", "Food & Dining"),
    ("milk and bread from local store", "Food & Dining"),
    ("fruit and vegetable purchase", "Food & Dining"),
    ("cafe breakfast croissant", "Food & Dining"),
    ("blinkit instant grocery", "Food & Dining"),
    ("zepto grocery delivery", "Food & Dining"),
    ("snacks chips chocolate", "Food & Dining"),
    ("rice dal pulses monthly stock", "Food & Dining"),
    ("cooking oil spices grocery", "Food & Dining"),
    ("street food pani puri chaat", "Food & Dining"),
    ("office canteen lunch", "Food & Dining"),
    # Transport
    ("uber ride to office", "Transport"),
    ("monthly bus pass", "Transport"),
    ("petrol refill at bpcl", "Transport"),
    ("ola cab booking airport", "Transport"),
    ("metro card recharge", "Transport"),
    ("rapido bike taxi", "Transport"),
    ("car service and maintenance", "Transport"),
    ("vehicle insurance renewal", "Transport"),
    ("toll plaza payment fastag", "Transport"),
    ("parking charges", "Transport"),
    ("auto rickshaw fare", "Transport"),
    ("diesel fill up", "Transport"),
    ("tyre puncture repair", "Transport"),
    ("irctc train ticket booking", "Transport"),
    # Shopping
    ("clothes from zara", "Shopping"),
    ("new shoes nike adidas", "Shopping"),
    ("online shopping meesho", "Shopping"),
    ("kurta dress ethnic wear", "Shopping"),
    ("handbag purse purchase", "Shopping"),
    ("watch accessory jewellery", "Shopping"),
    ("home decor cushion curtain", "Shopping"),
    ("kitchenware utensil purchase", "Shopping"),
    ("bedsheet pillow cover", "Shopping"),
    ("toys games for kids", "Shopping"),
    ("sports equipment cricket bat", "Shopping"),
    ("sunglasses cap hat", "Shopping"),
    ("wallet belt leather goods", "Shopping"),
    ("festive shopping diwali", "Shopping"),
    # Utilities
    ("electricity bill payment bescom", "Utilities"),
    ("internet broadband bill airtel", "Utilities"),
    ("water board bill", "Utilities"),
    ("mobile recharge jio", "Utilities"),
    ("gas cylinder lpg booking", "Utilities"),
    ("netflix subscription monthly", "Utilities"),
    ("amazon prime renewal", "Utilities"),
    ("hotstar disney premium", "Utilities"),
    ("spotify music subscription", "Utilities"),
    ("youtube premium subscription", "Utilities"),
    ("dth tata sky recharge", "Utilities"),
    ("postpaid mobile bill airtel", "Utilities"),
    ("wifi monthly plan renewal", "Utilities"),
    # Health
    ("doctor consultation fee", "Health"),
    ("medicine from apollo pharmacy", "Health"),
    ("gym membership monthly fee", "Health"),
    ("hospital bill surgery", "Health"),
    ("health insurance premium", "Health"),
    ("dental checkup teeth cleaning", "Health"),
    ("blood test lab report", "Health"),
    ("physiotherapy session", "Health"),
    ("eye checkup spectacles", "Health"),
    ("nutritional supplements protein", "Health"),
    ("yoga class membership", "Health"),
    ("medplus pharmacy purchase", "Health"),
    # Entertainment
    ("movie tickets pvr inox", "Entertainment"),
    ("gaming purchase steam", "Entertainment"),
    ("concert event tickets", "Entertainment"),
    ("amusement park entry fee", "Entertainment"),
    ("bookmyshow event booking", "Entertainment"),
    ("board game purchase", "Entertainment"),
    ("playstation game purchase", "Entertainment"),
    ("standup comedy show ticket", "Entertainment"),
    ("ipl cricket match ticket", "Entertainment"),
    # Education
    ("udemy course online learning", "Education"),
    ("college tuition fee semester", "Education"),
    ("books and stationery purchase", "Education"),
    ("online certification coursera", "Education"),
    ("school fee monthly", "Education"),
    ("coaching institute fee", "Education"),
    ("exam registration fee", "Education"),
    ("music guitar class fee", "Education"),
    ("coding bootcamp fee", "Education"),
    ("byju subscription learning", "Education"),
    # Travel
    ("hotel booking oyo makemytrip", "Travel"),
    ("holiday trip goa package", "Travel"),
    ("airbnb rental accommodation", "Travel"),
    ("travel insurance policy", "Travel"),
    ("flight ticket indigo spicejet", "Travel"),
    ("visa application fee", "Travel"),
    ("vacation shimla manali trip", "Travel"),
    ("sightseeing tour booking", "Travel"),
    # Personal Care
    ("face wash cetaphil", "Personal Care"),
    ("shampoo head shoulders", "Personal Care"),
    ("hair conditioner treatment", "Personal Care"),
    ("body lotion moisturizer", "Personal Care"),
    ("sunscreen spf lotion", "Personal Care"),
    ("deodorant axe deo spray", "Personal Care"),
    ("toothpaste colgate oral b", "Personal Care"),
    ("toothbrush electric manual", "Personal Care"),
    ("mouthwash listerine", "Personal Care"),
    ("razor shaving cream gillette", "Personal Care"),
    ("perfume cologne fragrance", "Personal Care"),
    ("hair oil coconut amla", "Personal Care"),
    ("hair serum leave in treatment", "Personal Care"),
    ("nail polish remover manicure", "Personal Care"),
    ("lipstick lip gloss", "Personal Care"),
    ("foundation makeup compact", "Personal Care"),
    ("kajal eyeliner mascara", "Personal Care"),
    ("face cream night cream olay", "Personal Care"),
    ("face serum vitamin c skincare", "Personal Care"),
    ("body wash shower gel", "Personal Care"),
    ("bar soap dove pears", "Personal Care"),
    ("hand wash sanitizer liquid", "Personal Care"),
    ("sanitary napkins stayfree", "Personal Care"),
    ("beard oil grooming kit", "Personal Care"),
    ("skincare routine products", "Personal Care"),
    ("face pack clay mask", "Personal Care"),
    ("haircut parlour visit", "Personal Care"),
    ("waxing threading salon", "Personal Care"),
    ("spa massage treatment", "Personal Care"),
    # Electronics
    ("laptop purchase dell hp", "Electronics"),
    ("macbook apple laptop", "Electronics"),
    ("iphone samsung smartphone purchase", "Electronics"),
    ("wireless headphones sony bose", "Electronics"),
    ("bluetooth earbuds boat", "Electronics"),
    ("power bank charger anker", "Electronics"),
    ("laptop charger adapter", "Electronics"),
    ("usb cable type c", "Electronics"),
    ("external hard drive 1tb", "Electronics"),
    ("pen drive 64gb sandisk", "Electronics"),
    ("gaming mouse keyboard rgb", "Electronics"),
    ("monitor 27 inch dell", "Electronics"),
    ("webcam logitech hd", "Electronics"),
    ("smart tv samsung 55 inch", "Electronics"),
    ("air conditioner 1.5 ton installation", "Electronics"),
    ("refrigerator repair service", "Electronics"),
    ("washing machine purchase lg", "Electronics"),
    ("microwave oven cooking", "Electronics"),
    ("mixer grinder kitchen bajaj", "Electronics"),
    ("water purifier kent ro", "Electronics"),
    ("smartwatch fitbit garmin", "Electronics"),
    ("ipad tablet purchase apple", "Electronics"),
    ("printer ink cartridge", "Electronics"),
    ("wifi router tp link", "Electronics"),
    ("phone screen replacement repair", "Electronics"),
    ("camera dslr mirrorless", "Electronics"),
    ("gaming console ps5 xbox", "Electronics"),
    ("bluetooth speaker jbl", "Electronics"),
    # Other
    ("miscellaneous expense", "Other"),
    ("cash withdrawal atm", "Other"),
    ("bank charges fee", "Other"),
    ("donation charity ngo", "Other"),
    ("gift purchase for friend", "Other"),
    ("loan emi payment", "Other"),
    ("rent payment monthly", "Other"),
    ("postal courier service", "Other"),
    ("repair maintenance home", "Other"),
]


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_match(text: str) -> str | None:
    lower = text.lower()
    for kw in sorted(KEYWORD_OVERRIDES, key=len, reverse=True):
        if kw in lower:
            return KEYWORD_OVERRIDES[kw]
    return None


class ExpenseCategorizor:
    CATEGORIES = [
        "Food & Dining", "Transport", "Shopping", "Utilities",
        "Health", "Entertainment", "Education", "Travel",
        "Personal Care", "Electronics", "Other",
    ]

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self._load_or_train()

    def predict(self, description: str) -> dict:
        # 1. Keyword override
        override = _keyword_match(description)
        if override:
            return {"category": override, "confidence": 1.0,
                    "all_scores": {override: 1.0}, "source": "keyword"}
        # 2. ML
        if self.pipeline is None:
            return {"category": "Other", "confidence": 0.0, "source": "fallback"}
        processed = _preprocess(description)
        proba = self.pipeline.predict_proba([processed])[0]
        classes = self.pipeline.classes_
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        category = classes[best_idx] if confidence >= 0.25 else "Other"
        return {"category": category, "confidence": round(confidence, 3),
                "all_scores": {c: round(float(p), 3) for c, p in zip(classes, proba)},
                "source": "ml"}

    def retrain(self, descriptions: list[str], labels: list[str]) -> dict:
        seed_desc, seed_labels = zip(*SEED_DATA)
        all_desc   = list(seed_desc)   + [_preprocess(d) for d in descriptions]
        all_labels = list(seed_labels) + labels
        X_train, X_test, y_train, y_test = train_test_split(
            all_desc, all_labels, test_size=0.2, random_state=42)
        self.pipeline.fit(X_train, y_train)
        report = classification_report(y_test, self.pipeline.predict(X_test), output_dict=True)
        self._save()
        return {"status": "retrained", "metrics": report}

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=8000,
                                      sublinear_tf=True, min_df=1)),
            ("clf",   LogisticRegression(max_iter=2000, C=5.0, solver="lbfgs",
                                         class_weight="balanced")),
        ])

    def _load_or_train(self):
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    self.pipeline = pickle.load(f)
                print(f"[Categorizer] Loaded model from {MODEL_PATH}")
                return
            except Exception:
                pass
        self._train_from_seed()

    def _train_from_seed(self):
        print("[Categorizer] Training model on seed data …")
        descriptions, labels = zip(*SEED_DATA)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit([_preprocess(d) for d in descriptions], list(labels))
        self._save()
        print(f"[Categorizer] Model trained on {len(SEED_DATA)} examples and saved.")

    def _save(self):
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.pipeline, f)


categorizer = ExpenseCategorizor()