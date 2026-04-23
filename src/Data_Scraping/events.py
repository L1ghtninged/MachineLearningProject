import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

API_KEY = "API_KEY"
HOTEL_FILE = "hotel_prices.csv"
OUTPUT_FILE = "events.csv"

CITY_COORDS = {
    "prague": {"lat": 50.0755, "lon": 14.4378},
    "vienna": {"lat": 48.2082, "lon": 16.3738},
    "budapest": {"lat": 47.4979, "lon": 19.0402},
    "berlin": {"lat": 52.5200, "lon": 13.4050},
    "paris": {"lat": 48.8566, "lon": 2.3522},
    "rome": {"lat": 41.9028, "lon": 12.4964},
    "barcelona": {"lat": 41.3851, "lon": 2.1734},
    "amsterdam": {"lat": 52.3676, "lon": 4.9041},
    "london": {"lat": 51.5074, "lon": -0.1278},
    "madrid": {"lat": 40.4168, "lon": -3.7038},
    "lisbon": {"lat": 38.7169, "lon": -9.1391},
    "copenhagen": {"lat": 55.6761, "lon": 12.5683},
    "stockholm": {"lat": 59.3293, "lon": 18.0686},
    "warsaw": {"lat": 52.2297, "lon": 21.0122},
    "dublin": {"lat": 53.3331, "lon": -6.2489},
    "brussels": {"lat": 50.8503, "lon": 4.3517},
    "oslo": {"lat": 59.9139, "lon": 10.7522},
    "helsinki": {"lat": 60.1695, "lon": 24.9354},
    "zurich": {"lat": 47.3769, "lon": 8.5417},
}

RADIUS_KM = 30
RATE_LIMIT_DELAY = 0.3


def get_unique_triplets(hotel_csv):
    df = pd.read_csv(hotel_csv)
    subset = df[['city', 'checkin', 'checkout']].drop_duplicates()
    return subset.to_dict('records')


def fetch_event_count(city_name, checkin, checkout):
    city = CITY_COORDS.get(city_name.lower())
    if not city:
        return 0

    start_dt = f"{checkin}T00:00:00Z"
    end_dt = f"{checkout}T23:59:59Z"

    params = {
        "apikey": API_KEY,
        "latlong": f"{city['lat']},{city['lon']}",
        "radius": RADIUS_KM,
        "unit": "km",
        "startDateTime": start_dt,
        "endDateTime": end_dt,
        "size": 1,
    }

    try:
        resp = requests.get("https://app.ticketmaster.com/discovery/v2/events.json", params=params)
        if resp.status_code == 429:
            print("Rate limit, 10 sekund")
            time.sleep(10)
            return fetch_event_count(city_name, checkin, checkout)

        resp.raise_for_status()
        data = resp.json()
        return data.get("page", {}).get("totalElements", 0)
    except Exception as e:
        print(f"Chyba u {city_name} {checkin}: {e}")
        return 0


def main():
    if not os.path.exists(HOTEL_FILE):
        print(f"Soubor {HOTEL_FILE} neexistuje")
        return

    targets = get_unique_triplets(HOTEL_FILE)
    print(f"Nalezeno {len(targets)} unikátních termínů ke stažení")

    results = []
    for i, t in enumerate(targets):
        count = fetch_event_count(t['city'], t['checkin'], t['checkout'])
        print(f"[{i + 1}/{len(targets)}] {t['city']} ({t['checkin']}): {count} eventů")

        results.append({
            "city": t['city'],
            "checkin": t['checkin'],
            "checkout": t['checkout'],
            "event_count": count
        })

        if (i + 1) % 20 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)

        time.sleep(RATE_LIMIT_DELAY)

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()