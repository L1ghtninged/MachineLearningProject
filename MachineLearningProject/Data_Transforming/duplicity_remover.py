import pandas as pd

INPUT_FILE = "../data/event_counts.csv"
OUTPUT_FILE = "../data/event_counts.csv"

SUBSET_COLUMNS = None

df = pd.read_csv(INPUT_FILE)

print("Původní počet řádků:", len(df))

df_clean = df.drop_duplicates(subset=SUBSET_COLUMNS, keep="first")

print("Po odstranění duplicit:", len(df_clean))
print("Odstraněno:", len(df) - len(df_clean))

df_clean.to_csv(OUTPUT_FILE, index=False)

print("Uloženo do:", OUTPUT_FILE)