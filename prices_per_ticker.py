import yfinance as yf
import pandas as pd

with open("companies.txt", "r", encoding="utf-8") as f:
    tickers = [
        line.strip().upper()
        for line in f
        if line.strip()
    ]

tickers = [f"{ticker}.HE" for ticker in tickers]

for ticker in tickers:
    data = yf.download(
        ticker,
        start="2016-01-01",
        end="2026-09-17",
        auto_adjust=False,
        group_by="column",
        threads=True
    )
    data.to_csv("prices_2016_2026/" + ticker + "_prices.csv")