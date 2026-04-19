"""
Dataset Transformation pro ML model
=====================================
Vstup:  dataset_final.csv
Výstup: dataset_ml.csv

Transformace:
  1. Oprava stars (dělit 2)
  2. Logaritmické transformace (review_count, distance_km)
  3. Cenová relativita vůči městu a datu
  4. Sezónní features (week_of_year, is_holiday)
  5. Target encoding měst
  6. Odstranění sloupců nepoužitelných pro ML
"""

import pandas as pd
import numpy as np
import holidays

# ================================================================
# NAČTENÍ
# ================================================================

df = pd.read_csv("dataset.csv")
print(f"Načteno řádků: {len(df)}")
print(f"Sloupce: {df.columns.tolist()}\n")

# ================================================================
# 1. OPRAVA STARS
# Booking scraper počítá <span> elementy → hodnoty 2,4,6,8,10
# Správná škála je 1–5
# ================================================================

df["stars"] = df["stars"] / 2

print("Stars po opravě:")
print(df["stars"].value_counts().sort_index())
print()

# ================================================================
# 2. LOGARITMICKÉ TRANSFORMACE
# ================================================================

df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"] = np.log1p(df["distance_km"])

# ================================================================
# 3. CENOVÁ RELATIVITA
# Poměr ceny hotelu vůči průměru ostatních hotelů
# ve stejném městě a ve stejný checkin den
# ================================================================

df["price_vs_city_mean"] = df.groupby(["city", "checkin"])["price"].transform(
    lambda x: x / x.mean()
)

# ================================================================
# 4. SEZÓNNÍ FEATURES
# ================================================================

df["checkin_dt"] = pd.to_datetime(df["checkin"])

# Týden v roce (1–53) – lepší granularita než měsíc
df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)

# Státní svátky – kombinace svátků pro všechna města v datasetu
# Každé město má svoji zemi
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

# Předpočítat svátky pro každou zemi na rok 2026
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

print("Svátky nalezeny:")
print(df["is_holiday"].value_counts())
print()

# ================================================================
# 5. TARGET ENCODING MĚST
# Průměrná cena za město – zachycuje obecnou cenovou hladinu
# ================================================================

city_mean_price = df.groupby("city")["price"].mean()
df["city_encoded"] = df["city"].map(city_mean_price)

print("City encoding (průměrná cena):")
print(city_mean_price.sort_values(ascending=False).round(0))
print()

# ================================================================
# 6. BREAKFAST → INT
# ================================================================

df["breakfast"] = df["breakfast"].astype(int)

# ================================================================
# 7. ODSTRANĚNÍ SLOUPCŮ NEPOUŽITELNÝCH PRO ML
# ================================================================

DROP_COLS = [
    "checkin",  # informaci nesou month, day_of_week, week_of_year
    "checkout",  # informaci nese stay_length
    "checkin_dt",  # pomocný sloupec
    "hotel_name",  # string, bez embeddings nepoužitelný
    "city",  # nahrazeno city_encoded
    "review_count",  # nahrazeno log_review_count
    "distance_km",  # nahrazeno log_distance
]

df_ml = df.drop(columns=DROP_COLS)

# ================================================================
# PŘEHLED VÝSLEDKU
# ================================================================

print("Finální sloupce:")
for col in df_ml.columns:
    print(f"  {col:<25} dtype={df_ml[col].dtype}  "
          f"null={df_ml[col].isna().sum()}  "
          f"min={df_ml[col].min():.2f}  max={df_ml[col].max():.2f}")

print(f"\nCelkem řádků: {len(df_ml)}")
print(f"Celkem sloupců: {len(df_ml.columns)}")

# ================================================================
# ULOŽENÍ
# ================================================================

df_ml.to_csv("dataset_ml.csv", index=False)
print("\n✅ Uloženo do dataset_ml.csv")
