import pandas as pd

df_h = pd.read_csv("../data/hotel_prices_final.csv")
df_e = pd.read_csv("../data/event_counts.csv")

df_h["city"] = df_h["city"].str.capitalize()
df_e["city"] = df_e["city"].str.capitalize()

df = df_h.merge(df_e[["city", "checkin", "checkout", "event_count"]],
                on=["city", "checkin", "checkout"],
                how="left")

print(df["event_count"].isna().sum())
df.to_csv("dataset.csv", index=False)