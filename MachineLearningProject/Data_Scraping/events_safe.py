import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import itertools

# ================================================================
# KONFIGURACE
# ================================================================

API_KEY = "API_KEY"

# Musí odpovídat názvům v hotel CSV (case-insensitive porovnání je níže)
CITIES = [
    {"name": "Prague", "lat": 50.0755, "lon": 14.4378},
    {"name": "Vienna", "lat": 48.2082, "lon": 16.3738},
    {"name": "Budapest", "lat": 47.4979, "lon": 19.0402},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Barcelona", "lat": 41.3851, "lon": 2.1734},
    {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041},
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Milan", "lat": 45.4642, "lon": 9.1900},
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

OUTPUT_FILE = "../data/event_counts.csv"
RADIUS_KM = 30  # Poloměr hledání eventů od středu města
MAX_PAGES = 5  # Max stránek na jeden API dotaz (1 stránka = 200 eventů)
RATE_LIMIT_DELAY = 0.25  # Sekund mezi requesty (Ticketmaster: 5 req/s limit)


# ================================================================
# GENEROVÁNÍ KOMBINACÍ (city × checkin/checkout)
# Stejná logika jako v tvém hotel scraperů
# ================================================================

def generate_date_pairs():
    """Stejná logika jako v hotel scraperů – každých 10 dní, stay 1–4 noci."""
    pairs = []
    start = datetime(2026, 1, 1)
    end = datetime(2026, 12, 31)
    current = start
    while current <= end:
        for stay in [1, 2, 3, 4]:
            checkout = current + timedelta(days=stay)
            pairs.append((
                current.strftime("%Y-%m-%d"),
                checkout.strftime("%Y-%m-%d"),
                stay
            ))
        current += timedelta(days=10)
    return pairs


DATE_PAIRS = generate_date_pairs()

# ================================================================
# API HELPER
# ================================================================

BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"


def fetch_event_count(city: dict, checkin: str, checkout: str) -> int:
    """
    Vrátí počet unikátních eventů v daném městě a období.
    Stránkuje dokud nejsou všechny eventy načteny nebo nedosáhne MAX_PAGES.
    """
    start_dt = f"{checkin}T00:00:00Z"
    end_dt = f"{checkout}T23:59:59Z"

    seen_ids: set = set()
    page = 0

    while page < MAX_PAGES:
        params = {
            "apikey": API_KEY,
            "latlong": f"{city['lat']},{city['lon']}",
            "radius": RADIUS_KM,
            "unit": "km",
            "startDateTime": start_dt,
            "endDateTime": end_dt,
            "size": 200,  # Maximum povolené Ticketmasterem
            "page": page,
            "sort": "date,asc",
            "locale": "en-us",
        }

        for attempt in range(4):  # Retry logika s exponenciálním backoffem
            try:
                resp = requests.get(BASE_URL, params=params, timeout=20)

                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"    ⏳ Rate limit (page {page}), čekám {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise RuntimeError("Neplatný API klíč. Zkontroluj API_KEY.")

                resp.raise_for_status()
                break

            except requests.Timeout:
                print(f"    ⌛ Timeout (pokus {attempt + 1}/4)")
                time.sleep(5)

        else:
            # Všechny pokusy selhaly – vrátíme co máme
            print(f"    ❌ Přeskakuji page {page} po 4 neúspěšných pokusech")
            break

        payload = resp.json()
        events = payload.get("_embedded", {}).get("events", [])

        if not events:
            break  # Žádné další výsledky

        for ev in events:
            eid = ev.get("id")
            if eid:
                seen_ids.add(eid)

        # Ticketmaster vrací totalPages v page metadata
        page_meta = payload.get("page", {})
        total_pages = page_meta.get("totalPages", 1)

        if page + 1 >= total_pages:
            break  # Načetli jsme všechny dostupné stránky

        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    return len(seen_ids)


# ================================================================
# HLAVNÍ SMYČKA
# ================================================================

def main():
    records = []
    total = len(CITIES) * len(DATE_PAIRS)
    done = 0

    for city in CITIES:
        print(f"\n{'=' * 50}")
        print(f"  {city['name'].upper()}")
        print(f"{'=' * 50}")

        for checkin, checkout, stay in DATE_PAIRS:
            done += 1
            count = fetch_event_count(city, checkin, checkout)

            records.append({
                "city": city["name"],
                "checkin": checkin,
                "checkout": checkout,
                "stay_length": stay,
                "event_count": count,
            })

            # Průběžné ukládání každých 50 záznamů
            if len(records) % 50 == 0:
                _save(records)

            progress = f"[{done}/{total}]"
            print(f"  {progress} {checkin}→{checkout} ({stay}d) → {count} eventů")

            time.sleep(RATE_LIMIT_DELAY)

    _save(records)
    print(f"\n✅ Hotovo. Uloženo {len(records)} záznamů do {OUTPUT_FILE}")
    return records


def _save(records):
    pd.DataFrame(records).to_csv(OUTPUT_FILE, index=False)


# ================================================================
# JOIN HELPER (použij po dokončení scrapování)
# ================================================================

def merge_with_hotels(hotel_csv: str = "hotel_prices_old.csv",
                      event_csv: str = "event_counts.csv") -> pd.DataFrame:
    """
    Mergne hotel data s event counts.

    Příklad použití:
        df = merge_with_hotels()
        df.to_csv("dataset_final.csv", index=False)
    """
    df_h = pd.read_csv(hotel_csv)
    df_e = pd.read_csv(event_csv)

    # Normalizace názvů měst (case-insensitive)
    df_h["city"] = df_h["city"].str.capitalize()
    df_e["city"] = df_e["city"].str.capitalize()

    df = df_h.merge(df_e[["city", "checkin", "checkout", "event_count"]],
                    on=["city", "checkin", "checkout"],
                    how="left")

    df["event_count"] = df["event_count"].fillna(0).astype(int)

    print(f"Hotels celkem:      {len(df_h):>6}")
    print(f"Event rows celkem:  {len(df_e):>6}")
    print(f"Merged rows:        {len(df):>6}")
    print(f"Chybějící event_count: {df['event_count'].isna().sum()}")

    return df


# ================================================================

if __name__ == "__main__":
    main()