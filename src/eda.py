import pandas as pd
import matplotlib.pyplot as plt
#load data
df = pd.read_csv("data/hourly_consumption.csv", index_col="datetime", parse_dates=True)
power = df["Global_active_power"]

#missing hourly timetsamps
expected = pd.date_range(start=power.index.min(),end=power.index.max(),freq="h")
missing_hours = expected.difference(power.index)
print("expected hourly observations",len(expected))
print("available hourly observations", len(power))
print("missing hours:", len(missing_hours))

#distribution of consumption
plt.figure(figsize=(10,5))
plt.hist(power,bins=50,edgecolor ="black")

plt.title("distribution of electricity consumption")
plt.xlabel("activepower (kw)")
plt.ylabel("frequency")
plt.tight_layout()
plt.show()

#avg consumption by hour
hourly_pattern = power.groupby(power.index.hour).mean()
plt.figure(figsize=(10,5))
plt.plot(hourly_pattern.index,hourly_pattern.values,marker = "o")
plt.title("avg electricity consumption by hour")
plt.xlabel("hour of day")
plt.ylabel("avg active power (kw)")
plt.xticks(range(0,24))
plt.grid()
plt.tight_layout()
plt.show()

# monthly consumption trend
monthly_pattern =power.resample("MS").mean()
plt.figure(figsize=(12,5))
plt.plot(monthly_pattern.index,monthly_pattern.values)
plt.title("Monthly avg electricity consumption")
plt.xlabel("Date")
plt.ylabel("Avg active power(kw)")
plt.grid()
plt.tight_layout()
plt.show()

#restore missing timestamps as nan
complete_power = power.reindex(expected)
# prev hour vs current hour
previous_hour = power.shift(1)
plt.figure(figsize=(8,6))
plt.scatter(previous_hour,power,alpha=0.15,s=5)
plt.title("Prev hour vs current hour consumption")
plt.xlabel("Previous hour (kw)")
plt.ylabel("current hour kw")
plt.grid()
plt.tight_layout()
plt.show()