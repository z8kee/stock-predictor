# app.py
from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# 1. Serve the main HTML website
@app.route('/')
def home():
    return render_template('index.html')

# 2. Create an API endpoint for the frontend to fetch stock data
@app.route('/api/history/<ticker>/<interval>')
def get_history(ticker, interval):
    print(f"Frontend requested data for: {ticker}")
    
    if interval == '1m':
        period = "5d"
    elif interval in ['5m', '15m', '1h']:
        period = "1mo"
    else:
        period = "6mo"

    df = yf.download(ticker, period, interval)
    
    # TradingView requires a very specific JSON format: 
    # {time: 'YYYY-MM-DD', open: O, high: H, low: L, close: C}
    chart_data = []
    for index, row in df.iterrows():
        chart_data.append({
            "time": int(index.timestamp()),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        })
        
    return jsonify(chart_data)

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)