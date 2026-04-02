import yfinance as yf

def load_data(ticker="AAPL"):
    data = yf.download(ticker, start="2018-01-01", end="2023-01-01")
    return data["Close"].dropna()