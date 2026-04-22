import pandas as pd
import numpy as np
import holidays

df = pd.read_csv("dataset.csv")
df["stars"] = df["stars"] / 2

df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"] = np.log1p(df["distance_km"])

df["price_vs_city_mean"] = df.groupby(["city", "checkin"])["price"].transform(
    lambda x: x / x.mean()
)

df["checkin_dt"] = pd.to_datetime(df["checkin"])

df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)

CITY_COUNTRY = {
    "Prague": "CZ",
    "Vienna": "AT",
    "Budapest": "HU",
    "Berlin": "DE",
    "Paris": "FR",
    "Rome": "IT",
    "Barcelona": "ES",
    "Amsterdam": "NL",
    "London": "GB",
    "Madrid": "ES",
    "Milan": "IT",
    "Munich": "DE",
    "Lisbon": "PT",
    "Copenhagen": "DK",
    "Stockholm": "SE",
    "Warsaw": "PL",
    "Dublin": "IE",
    "Brussels": "BE",
    "Oslo": "NO",
    "Helsinki": "FI",
    "Zurich": "CH",
}

COUNTRY_HOLIDAYS = {
    country: holidays.country_holidays(country, years=2026)
    for country in set(CITY_COUNTRY.values())
}


def is_holiday(row):
    country = CITY_COUNTRY.get(row["city"])
    if not country:
        return 0
    return int(row["checkin_dt"].date() in COUNTRY_HOLIDAYS[country])


df["is_holiday"] = df.apply(is_holiday, axis=1)

city_mean_price = df.groupby("city")["price"].mean()
df["city_encoded"] = df["city"].map(city_mean_price)

df["breakfast"] = df["breakfast"].astype(int)

DROP_COLS = [
    "checkin",
    "checkout",
    "checkin_dt",
    "hotel_name",
    "city",
    "review_count",
    "distance_km",
]

df_ml = df.drop(columns=DROP_COLS)

df_ml.to_csv("dataset_ml.csv", index=False)
