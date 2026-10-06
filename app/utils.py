import yfinance as yf
import numpy as np
import json

def write_prices():
    with open("static/companies.txt", "r", encoding="utf-8") as file:
        tickers = [line.strip().upper() + ".HE" for line in file if line.strip()]

    data = {}
    for ticker in tickers:
        ticker_data = yf.download(ticker, auto_adjust=False, start="2016-01-01", end="2026-09-17")
        data[ticker[:-3]] = ticker_data["Adj Close"].squeeze().tolist()

    with open("static/data.json", "w", encoding="utf-8") as file:
        json.dump(data, file)

def normalize_prices():
    data = get_prices()

    for key, values in data.items():
        values = np.asarray(values)
        mean = np.mean(values)
        std = np.std(values)
        normalized = (values - mean) / std
        data[key] = normalized.tolist()

    with open("static/normalized.json", "w", encoding="utf-8") as file:
        json.dump(data, file)

def get_prices():
    with open("static/data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_normalized():
    with open("static/normalized.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data
