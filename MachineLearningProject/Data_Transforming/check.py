import pandas as pd

df_h = pd.read_csv("../data/hotel_prices_final.csv")
df_e = pd.read_csv("../data/event_counts.csv")

df_h["city"] = df_h["city"].str.capitalize()
df_e["city"] = df_e["city"].str.capitalize()

# Kolik unikátních kombinací je v každém datasetu
hotel_keys = set(zip(df_h["city"], df_h["checkin"], df_h["checkout"]))
event_keys = set(zip(df_e["city"], df_e["checkin"], df_e["checkout"]))

print(f"Unikátní kombinace v hotelech: {len(hotel_keys)}")
print(f"Unikátní kombinace v eventech: {len(event_keys)}")
print(f"Překryv (match):               {len(hotel_keys & event_keys)}")
print(f"Hotely BEZ event záznamu:      {len(hotel_keys - event_keys)}")