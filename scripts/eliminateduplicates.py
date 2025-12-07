import pandas as pd

df = pd.read_csv("/Users/nicolasmarkides/Documents/lag-llama-project/lag-llama/data/full_exchange_rates.csv")

# Drop duplicate dates
df = df.drop_duplicates(subset="DATE", keep="first")

# Convert to datetime
df["date"] = pd.to_datetime(df["DATE"])

# Sort and set index
df = df.sort_values("date").set_index("date").asfreq("D")

# Save cleaned CSV if needed
df.to_csv("/Users/nicolasmarkides/Documents/lag-llama-project/lag-llama/data/full_exchange_rates_cleaned.csv", index=True)
print("Duplicates removed and cleaned CSV saved.")