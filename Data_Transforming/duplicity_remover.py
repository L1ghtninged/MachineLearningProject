import pandas as pd

INPUT_FILE = "../data/event_counts.csv"
OUTPUT_FILE = "../data/event_counts.csv"
SUBSET_COLUMNS = None

df = pd.read_csv(INPUT_FILE)
df_clean = df.drop_duplicates(subset=SUBSET_COLUMNS, keep="first")
df_clean.to_csv(OUTPUT_FILE, index=False)
