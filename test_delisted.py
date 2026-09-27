import yfinance as yf

ticker = yf.Ticker("OKM1V.HE")

data = ticker.history(
    start="2016-01-01",
    end="2017-01-01",
    auto_adjust=False
)

print(data)