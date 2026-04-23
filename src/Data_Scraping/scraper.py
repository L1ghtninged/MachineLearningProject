from playwright.sync_api import sync_playwright
import pandas as pd
import time
import re
from datetime import datetime, timedelta

cities = [
    "prague","vienna","budapest","berlin","paris","rome",
    "barcelona","amsterdam","london","madrid"
    "lisbon","copenhagen","stockholm","warsaw",
    "dublin","brussels","oslo","helsinki","zurich"
]
TARGET_PER_CITY = 1000

import random


def generate_dates():
    dates = []
    start = datetime(2026, 4, 24)
    end = datetime(2026, 12, 31)
    current = start

    stay_lengths = [1, 2, 3, 4, 5]

    while current <= end:
        chosen_lengths = random.sample(stay_lengths, 2)

        for stay in chosen_lengths:
            checkout = current + timedelta(days=stay)
            dates.append((current.strftime("%Y-%m-%d"), checkout.strftime("%Y-%m-%d")))

        current += timedelta(days=12)
    return dates

dates = generate_dates()
HOTELS_PER_DATE = TARGET_PER_CITY // len(dates)
print(len(dates))
print(HOTELS_PER_DATE)

data = []

def extract_number(text):
    if not text:
        return None
    text = text.replace("\xa0", "").replace(",", ".")
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else None


def extract_price(text):
    if not text:
        return None

    text = text.replace("\xa0", "").replace(" ", "")

    match = re.search(r"\d{1,3}(?:[.,]\d{3})+", text)

    if match:
        num = match.group().replace(",", "").replace(".", "")
        return int(num)

    return None

all_start_time = time.time()

with sync_playwright() as p:

    browser = p.chromium.launch(headless=False)

    context = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    )

    page = context.new_page()
    city_counts = {city: 0 for city in cities}

    for city in cities:
        city_collected = 0
        for checkin, checkout in dates:
            date_collected = 0

            for page_number in range(2):
                if date_collected >= HOTELS_PER_DATE:
                    break
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

                try:
                    page.goto(url, timeout=25000)
                    page.wait_for_load_state("networkidle")
                    time.sleep(2)
                except:
                    break

                hotels = page.query_selector_all('[data-testid="property-card"]')

                if not hotels:
                    break

                print("Hotels:", len(hotels))

                for hotel in hotels:
                    if date_collected >= HOTELS_PER_DATE:
                        break
                    if city_counts[city] >= TARGET_PER_CITY:
                        break

                    name = None
                    price = None
                    rating = None
                    stars = None
                    distance_km = None
                    breakfast = False
                    review_count = None

                    el = hotel.query_selector('[data-testid="title"]')
                    if el:
                        name = el.inner_text().strip()

                    el = hotel.query_selector('[data-testid="price-and-discounted-price"]') \
                         or hotel.query_selector('[data-testid="price"]')

                    if el:
                        price = extract_price(el.inner_text())

                    rating_el = hotel.query_selector('[data-testid="review-score"]')
                    if rating_el:
                        rating = extract_number(rating_el.inner_text())

                    el = hotel.query_selector('[data-testid="rating-stars"]')
                    if el:
                        stars = len(el.query_selector_all("span"))

                    el = hotel.query_selector('[data-testid="distance"]')
                    if el:
                        txt = el.inner_text().lower()
                        num = extract_number(txt)

                        if num:
                            if "km" in txt:
                                distance_km = num
                            elif "m" in txt:
                                distance_km = num / 1000

                    spans = hotel.query_selector_all("span")
                    for s in spans:
                        if "breakfast" in s.inner_text().lower():
                            breakfast = True
                            break

                    if rating_el:
                        try:
                            parent = rating_el.evaluate_handle("el => el.parentElement")
                            txt = parent.inner_text().lower()
                            match = re.search(r"(\d[\d,]*)\s+reviews", txt)
                            if match:
                                review_count = int(match.group(1).replace(",", ""))
                        except:
                            pass

                    checkin_dt = datetime.strptime(checkin, "%Y-%m-%d")
                    checkout_dt = datetime.strptime(checkout, "%Y-%m-%d")

                    month = checkin_dt.month
                    dow = checkin_dt.weekday()
                    is_weekend = 1 if dow >= 5 else 0
                    stay_length = (checkout_dt - checkin_dt).days

                    if price and rating:
                        data.append({
                            "city": city,
                            "checkin": checkin,
                            "checkout": checkout,
                            "hotel_name": name,
                            "price": price,
                            "rating": rating,
                            "stars": stars,
                            "distance_km": distance_km,
                            "breakfast": breakfast,
                            "review_count": review_count,
                            "month": month,
                            "day_of_week": dow,
                            "is_weekend": is_weekend,
                            "stay_length": stay_length
                        })
                        city_counts[city] += 1
                        date_collected += 1
                time.sleep(1)

    browser.close()
total_end_time = time.time()
total_elapsed = (total_end_time - all_start_time) / 60
print(f"Celkový čas: {total_elapsed:.2f} minut")
print(f"Celkem nasbíráno: {len(data)} řádků")
df = pd.DataFrame(data)
#df = df.dropna()

filename = f"hotel_prices_{cities[0]}.csv"
df.to_csv(filename, index=False)
print(f"\nUloženo do: {filename}")

print("\nSaved:", len(df))
