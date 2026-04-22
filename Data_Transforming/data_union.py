import pandas as pd

CSV_FILE_1 = "../data/events_clean.csv"
CSV_FILE_2 = "../data/events_filtered.csv"
OUTPUT_FILE = "../data/events_merged.csv"

df1 = pd.read_csv(CSV_FILE_1)
df2 = pd.read_csv(CSV_FILE_2)

df_merged = pd.concat([df1, df2], ignore_index=True)

df_merged = df_merged.drop_duplicates(keep="first")

df_merged.to_csv(OUTPUT_FILE, index=False)
print("Spojený soubor uložen do:", OUTPUT_FILE)