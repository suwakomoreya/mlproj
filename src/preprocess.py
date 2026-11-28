from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

dataset = fetch_ucirepo(id=235)
df = dataset.data.features.copy()

# numerical columns
numeric_cols = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

#missing value check
print("missing vals after  conversion")
print(df.isnull().sum())

# date time combined
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format ="%d/%m/%Y %H:%M:%S", errors = "coerce")

# filter out invalid timestamps
df = df.dropna(subset=["datetime"])

#sort chronologically
df = df.sort_values("datetime")
print(df[["datetime", "Global_active_power"]].head())

#datatime as the df index
power = df.set_index("datetime")["Global_active_power"]

# aggregate min measurement sinto hourly avgs
hourly = power.resample("h").agg(["mean","count"])

hourly = hourly.rename(columns={"mean": "Global_active_power","count": "valid_minutes"})
#keep hours with 45 valid measumrents
hourly = hourly[hourly["valid_minutes"]>=45]

#remove measurecountcolumn
hourly = hourly.drop(columns=["valid_minutes"])

print(hourly.head())
print(hourly.describe())
print("hourly observ:", len(hourly))


#saving processed data
Path("data").mkdir(exist_ok=True)
#save dataset
hourly.to_csv("data/hourly_consumption.csv")
print("dataset saved")