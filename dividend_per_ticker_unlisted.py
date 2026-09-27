import yfinance as yf
import pandas as pd

with open("companies_unlisted.txt", "r", encoding="utf-8") as f:
    tickers = [
        line.strip().upper()
        for line in f
        if line.strip()
    ]

tickers = [f"{ticker}.HE" for ticker in tickers]

for ticker in tickers:
    stock = yf.Ticker(ticker)
    data = stock.dividends
    data.to_csv("dividends_unlisted/" + ticker + "_dividends.csv")