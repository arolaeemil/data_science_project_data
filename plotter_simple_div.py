import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

file_path = "dividends/UPM.HE_dividends.csv"

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

plt.figure(figsize=(12, 6))
plt.plot(data.index, data.values, linewidth=1.5)
plt.title(f"{file_path}")
plt.xlabel("Date")
plt.ylabel("Dividend")
plt.grid(True, alpha=0.3)
plt.show()