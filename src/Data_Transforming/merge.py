import pandas as pd

df_h = pd.read_csv("../data/hotel_prices.csv")
df_e = pd.read_csv("../data/events.csv")

print(df_h.head())
print(df_e.head())


df = df_h.merge(df_e[["city", "checkin", "checkout", "event_count"]],
                on=["city", "checkin", "checkout"],
                how="left")

print(df["event_count"].isna().sum())
df.to_csv("dataset.csv", index=False)