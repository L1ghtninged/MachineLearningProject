from playwright.sync_api import sync_playwright
import pandas as pd
import time
import re
from datetime import datetime

cities = [
    "prague","vienna","budapest","berlin","rome","paris"
]

# různé sezóny
dates = [
    ("2026-01-15","2026-01-16"),  # zima
    ("2026-04-10","2026-04-11"),  # jaro
    ("2026-07-10","2026-07-11"),  # léto
    ("2026-10-10","2026-10-11")   # podzim
]

data = []

def extract_number(text):
    if not text:
        return None
    text = text.replace("\xa0", "").replace(",", ".")
    match = re.search(r"\d+(\.\d+)?", text)
    if match:
        return float(match.group())
    return None

def extract_price(text):
    if not text:
        return None

    text = text.replace("\xa0", "").replace(" ", "")

    # např. 3,450 nebo 12.500
    match = re.search(r"\d{1,3}(?:[.,]\d{3})+", text)

    if match:
        num = match.group()
        num = num.replace(",", "").replace(".", "")
        return int(num)

    return None

with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)

    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )

    page = context.new_page()

    for city in cities:
        for checkin, checkout in dates:

            print(f"\n--- {city} | {checkin} ---")

            for page_number in range(8):

                offset = page_number * 25

                url = (
                    f"https://www.booking.com/searchresults.html?"
                    f"ss={city}"
                    f"&checkin_year={checkin[:4]}"
                    f"&checkin_month={checkin[5:7]}"
                    f"&checkin_monthday={checkin[8:]}"
                    f"&checkout_year={checkout[:4]}"
                    f"&checkout_month={checkout[5:7]}"
                    f"&checkout_monthday={checkout[8:]}"
                    f"&group_adults=2"
                    f"&no_rooms=1"
                    f"&offset={offset}"
                    f"&lang=en-us"
                )

                page.goto(url)

                try:
                    page.wait_for_load_state("networkidle")
                    page.wait_for_timeout(3000)
                except:
                    print("Timeout – skipping page")
                    continue

                hotels = page.query_selector_all('[data-testid="property-card"]')

                print("Hotels found:", len(hotels))

                for hotel in hotels:

                    name = None
                    price = None
                    rating = None
                    stars = None
                    distance_km = None
                    breakfast_included = False
                    review_count = None

                    # -------- NAME --------
                    name_el = hotel.query_selector('[data-testid="title"]')
                    if name_el:
                        name = name_el.inner_text().strip()

                    # -------- PRICE --------
                    price_el = hotel.query_selector('[data-testid="price-and-discounted-price"]') \
                               or hotel.query_selector('[data-testid="price"]')

                    if price_el:
                        price = extract_price(price_el.inner_text())

                    # -------- RATING --------
                    rating_el = hotel.query_selector('[data-testid="review-score"]')
                    if rating_el:
                        rating = extract_number(rating_el.inner_text())

                    # -------- STARS --------
                    star_el = hotel.query_selector('[data-testid="rating-stars"]')
                    if star_el:
                        stars = len(star_el.query_selector_all("span"))

                    # -------- DISTANCE --------
                    distance_el = hotel.query_selector('[data-testid="distance"]')

                    if distance_el:
                        text = distance_el.inner_text().lower()
                        number = extract_number(text)

                        if number is not None:
                            if "km" in text:
                                distance_km = number
                            elif "m" in text:
                                distance_km = number / 1000

                    # -------- BREAKFAST --------
                    badges = hotel.query_selector_all("span")

                    for b in badges:
                        txt = b.inner_text().lower()
                        if "breakfast" in txt:
                            breakfast_included = True
                            break

                    # -------- REVIEW COUNT --------
                    review_el = hotel.query_selector('[data-testid="review-score"]')

                    if review_el:
                        parent = review_el.evaluate_handle("el => el.parentElement")
                        text = parent.inner_text().lower()

                        match = re.search(r"(\d[\d,]*)\s+reviews", text)

                        if match:
                            review_count = int(match.group(1).replace(",", ""))

                    # -------- DATE FEATURES --------
                    checkin_dt = datetime.strptime(checkin, "%Y-%m-%d")
                    checkout_dt = datetime.strptime(checkout, "%Y-%m-%d")

                    month = checkin_dt.month
                    day_of_week = checkin_dt.weekday()
                    is_weekend = 1 if day_of_week >= 5 else 0
                    stay_length = (checkout_dt - checkin_dt).days

                    # -------- SAVE --------
                    if price and rating:

                        data.append({
                            "city": city,
                            "hotel_name": name,
                            "price": price,
                            "rating": rating,
                            "stars": stars,
                            "distance_km": distance_km,
                            "breakfast": breakfast_included,
                            "review_count": review_count,
                            "month": month,
                            "day_of_week": day_of_week,
                            "is_weekend": is_weekend,
                            "stay_length": stay_length
                        })

                        print(city, name, price)

                time.sleep(2)

    browser.close()

# -------- SAVE DATA --------
df = pd.DataFrame(data)

# odstranění prázdných hodnot
df = df.dropna()

df.to_csv("hotel_prices_final.csv", index=False)

print("\nSaved records:", len(df))