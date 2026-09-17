import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file_path = "prices_2016_2026/AALLON.HE_prices.csv"

data = pd.read_csv(
    file_path,
    header=[0, 1],
    index_col=0,
    parse_dates=True
)

#print(data.columns)
#print(data)
#print(data.index)
#print(data.values)
#print(data["Adj Close"])
#print(data["Adj Close"].values)

plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Adj Close"].values, linewidth=1.5)
plt.title(f"{file_path} - Adjusted Close")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.show()