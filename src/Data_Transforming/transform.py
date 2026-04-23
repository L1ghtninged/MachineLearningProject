import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv")
df["stars"] = df["stars"] / 2

df["log_review_count"] = np.log1p(df["review_count"])
df["log_distance"] = np.log1p(df["distance_km"])

df["price_vs_city_mean"] = df.groupby(["city", "checkin"])["price"].transform(
    lambda x: x / x.mean()
)

df["checkin_dt"] = pd.to_datetime(df["checkin"])

df["week_of_year"] = df["checkin_dt"].dt.isocalendar().week.astype(int)

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
