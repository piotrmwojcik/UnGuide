import pandas as pd

# Hardcoded config
CSV_PATH = "data/I2P_prompts_4703.csv"   # <--- change this to your CSV path
NUDITY_FILTER_ENABLED = True         # set to False if you don't want the filter

# Read CSV, using first column as index
df = pd.read_csv(CSV_PATH, index_col=0)

# Check if this is an NSFW dataset with nudity_percentage column
if NUDITY_FILTER_ENABLED and "nudity_percentage" in df.columns:
    # ensure numeric (coerce bad values to NaN)
    df["nudity_percentage"] = pd.to_numeric(df["nudity_percentage"], errors="coerce")
    # keep rows with nudity_percentage > 0
    df = df[df["nudity_percentage"].gt(0)]
    # sort descending
    df = df.sort_values(by="nudity_percentage", ascending=False)

# Do something with the result, e.g. show first rows or save
print(df.head())