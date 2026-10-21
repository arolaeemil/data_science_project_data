import yfinance as yf
import numpy as np
import json

def write_prices_open():
    with open("static/companies.txt", "r", encoding="utf-8") as file:
        tickers = [line.strip().upper() + ".HE" for line in file if line.strip()]

    data = {}
    for ticker in tickers:
        ticker_data = yf.download(ticker, auto_adjust=False, start="2016-01-01", end="2026-09-17")
        data[ticker[:-3]] = ticker_data["Open"].squeeze().tolist()

    with open("static/data_open.json", "w", encoding="utf-8") as file:
        json.dump(data, file)


def write_prices_open_cut(cut_amount):
    with open("static/companies.txt", "r", encoding="utf-8") as file:
        tickers = [line.strip().upper() + ".HE" for line in file if line.strip()]

    data = {}
    for ticker in tickers:
        ticker_data = yf.download(ticker, auto_adjust=False, start="2016-01-01", end="2026-09-17")
        data[ticker[:-3]] = ticker_data["Open"].squeeze().tolist()[:-cut_amount]

    data_path = "static/simulation/data_open_cut_" + str(cut_amount) + ".json"
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(data, file)

def normalize_prices_open():
    data = get_prices_open()
    normalization_data = {}

    for key, values in data.items():
        values = np.asarray(values)
        mean = np.mean(values)
        std = np.std(values)
        normalized = (values - mean) / std
        data[key] = normalized.tolist()
        normalization_data[key] = {"mean": mean, "std": std}

    with open("static/normalized_open.json", "w", encoding="utf-8") as file:
        json.dump(data, file)
    with open("static/normalization_data_open.json", "w", encoding="utf-8") as file:
        json.dump(normalization_data, file)

def get_prices_open():
    with open("static/data_open.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def normalize_prices_open_cut(cut_amount):
    data = get_prices_open_cut(cut_amount)
    normalization_data = {}

    for key, values in data.items():
        values = np.asarray(values)
        mean = np.mean(values)
        std = np.std(values)
        normalized = (values - mean) / std
        data[key] = normalized.tolist()
        normalization_data[key] = {"mean": mean, "std": std}

    data_path = "static/simulation/normalized_open_cut_" + str(cut_amount) + ".json"
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(data, file)
    data_path = "static/simulation/normalization_data_open_cut_" + str(cut_amount) + ".json"    
    with open(data_path, "w", encoding="utf-8") as file:
        json.dump(normalization_data, file)


def get_prices_open_cut(cut_amount):
    data_path = "static/simulation/data_open_cut_" + str(cut_amount) + ".json"
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def companies():
    with open("static/companies.txt", "r", encoding="utf-8") as file:
        return [line.strip().upper() for line in file if line.strip()]

def get_normalized_open():
    with open("static/normalized_open.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_normalization_data_open():
    with open("static/normalization_data_open.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_normalized_open_cut(cut_amount):
    data_path = "static/simulation/normalized_open_cut_" + str(cut_amount) + ".json"
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_normalization_data_open_cut(cut_amount):
    data_path = "static/simulation/normalization_data_open_cut_" + str(cut_amount) + ".json"
    with open(data_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def unnormalize_open(data):
    normalization_data = get_normalization_data_open()
    original = {}

    for key, value in data.items():
        mean = normalization_data[key]["mean"]
        std = normalization_data[key]["std"]
        original[key] = (value * std) + mean
    return original

def unnormalize_open_cut(data, cut_amount):
    normalization_data = get_normalization_data_open_cut(cut_amount)
    original = {}

    for key, value in data.items():
        mean = normalization_data[key]["mean"]
        std = normalization_data[key]["std"]
        original[key] = (value * std) + mean
    return original

if __name__ == "__main__":

    #write_prices_open()
    #normalize_prices_open()

    make_cuts = 1

    if make_cuts == 1:
        limit = 30
        cut = 30
        i = cut
        while i <= limit*cut:
            write_prices_open_cut(i)
            normalize_prices_open_cut(i)
            i = i + cut
