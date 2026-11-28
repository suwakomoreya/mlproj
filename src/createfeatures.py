import pandas as pd
import numpy as np
from pathlib import Path
#load hourly consumption
df = pd.read_csv("data/hourly_consumption.csv",index_col=0,parse_dates=True)
power=df["Global_active_power"]
#reconstruct complete hourly timeline
#missing hours = nan
expected = pd.date_range(start=power.index.min(),end=power.index.max(),freq="h")
power = power.reindex(expected)
#create feature df
features = pd.DataFrame(index=power.index)
#historical consumption features
features["current_power"]=power
features["previous_hour_power"]=power.shift(1)
features["same_hour_yesterday"]=power.shift(24)

#average of 24 hours before  the current hour
features["rolling_mean_24h"]=(power.shift(1).rolling(window=24).mean())
#calendar features for the hour we are predicting
target_time = features.index + pd.Timedelta(hours=1)
features["hour"] = target_time.hour
features["day_of_week"] = target_time.day_of_week
features["month"]= target_time.month
#target: consumption during the next hour
features["next_hour_power"] = power.shift(-1)
#remove rows containg missing features or targets
features=features.dropna()
#save features dataset
features.index.name ="datetime"

Path("data").mkdir(exist_ok=True)
features.to_csv("data/features.csv")
print("feature dataset: ")
print(features.head())

print("\nfeature statistics:")
print(features.describe())

print("\nfeaturedataset shape:")
print(features.shape)
print("\nsaved features.csv")