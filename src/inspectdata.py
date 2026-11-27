from ucimlrepo import fetch_ucirepo
import pandas as pd

dataset = fetch_ucirepo(id=235)

df = dataset.data.features.copy()
print(df.head()) #print first 5 rows