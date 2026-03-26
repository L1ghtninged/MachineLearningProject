import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = "GWuuGDawS9v78CGGDtURscJ5UotF2LiJ"

cities = [
    {"name": "Prague", "lat": 50.0755, "lon": 14.4378},
    {"name": "Vienna", "lat": 48.2082, "lon": 16.3738},
    {"name": "Budapest", "lat": 47.4979, "lon": 19.0402},
    {"name": "Berlin", "lat": 52.52, "lon": 13.4050},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Barcelona", "lat": 41.3851, "lon": 2.1734},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Milan", "lat": 45.4642, "lon": 9.19},
    {"name": "Munich", "lat": 48.1351, "lon": 11.5820},
    {"name": "Lisbon", "lat": 38.7169, "lon": -9.1391},
    {"name": "Copenhagen", "lat": 55.6761, "lon": 12.5683},
    {"name": "Stockholm", "lat": 59.3293, "lon": 18.0686},
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122},
    {"name": "Dublin", "lat": 53.3331, "lon": -6.2489},
    {"name": "Brussels", "lat": 50.8503, "lon": 4.3517},
    {"name": "Oslo", "lat": 59.9139, "lon": 10.7522},
    {"name": "Helsinki", "lat": 60.1695, "lon": 24.9354},
    {"name": "Zurich", "lat": 47.3769, "lon": 8.5417},
]

start_date = datetime(2026, 6, 1)
end_date = datetime(2026, 9, 30)

data = []
seen = set()
OUTPUT_FILE = "events.csv"

def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(days=n)

def save_progress():
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Průběžně uloženo {len(df)} záznamů do {OUTPUT_FILE}")

for city in cities:
    print(f"\n--- {city['name']} ---")

    for single_date in daterange(start_date, end_date):
        start_dt = single_date.strftime("%Y-%m-%dT00:00:00Z")
        end_dt = (single_date + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

        page = 0
        while True:
            params = {
                "apikey": API_KEY,
                "latlong": f"{city['lat']},{city['lon']}",
                "radius": 50,
                "unit": "km",
                "startDateTime": start_dt,
                "endDateTime": end_dt,
                "size": 50,
                "page": page,
                "sort": "date,asc",
                "locale": "en-us"
            }

            try:
                response = requests.get(
                    "https://app.ticketmaster.com/discovery/v2/events.json",
                    params=params,
                    timeout=20
                )

                if response.status_code == 429:  # rate limit
                    wait_time = 5 + page*2  # exponenciální backoff
                    print(f"Rate limit, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue  # zkus znovu stejnou stránku

                response.raise_for_status()

                data_json = response.json()
                events = data_json.get("_embedded", {}).get("events", [])

                if not events:  # už žádné eventy
                    break

                print(f"{single_date.date()} page {page} -> {len(events)} events")

                for event in events:
                    event_id = event.get("id")
                    if event_id in seen:
                        continue
                    seen.add(event_id)

                    venue = event.get("_embedded", {}).get("venues", [{}])[0]
                    data.append({
                        "city": city["name"],
                        "date": event.get("dates", {}).get("start", {}).get("localDate"),
                        "event_name": event.get("name"),
                        "lat": float(venue.get("location", {}).get("latitude", 0)),
                        "lon": float(venue.get("location", {}).get("longitude", 0))
                    })

                if len(data) % 200 == 0:
                    save_progress()

                page += 1
                time.sleep(0.3)

            except requests.RequestException as e:
                print("Request error:", e)
                time.sleep(5)
                continue
            except Exception as e:
                print("Other error:", e)
                break

save_progress()
print("\nHotovo. Celkem záznamů:", len(data))