import pandas as pd
try:
    df = pd.read_csv("../data/hotel_prices.csv")
    df = df.dropna()
    print("Obsazení podle měst")
    city_counts = df['city'].value_counts()

    print(city_counts)

    print("\n-------------------------------")
    print(f"Celkový počet záznamů v datasetu: {len(df)}")
    print(f"Počet unikátních měst: {df['city'].nunique()}")


except FileNotFoundError:
    pass