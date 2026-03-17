# app.py
from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
from predictor import SentimentAnalyser

app = Flask(__name__)
stock_cache = {}

analyser = SentimentAnalyser(0.2)
print("analyser ready twin")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/history/<ticker>/<interval>')
def get_history(ticker, interval):
    # Create a unique key for the cache (e.g., "GC=F_5m")
    cache_key = f"{ticker}_{interval}"
    
    if interval == '1m':
        period = "5d"
    elif interval in ['5m', '15m', '1h']:
        period = "1mo"
    else:
        period = "6mo"

    try:
        if cache_key in stock_cache:
            # CACHE HIT
            print(f"Auto-updating {ticker}... Fetching 1d data only.")
            new_df = yf.download(tickers=ticker, period="1d", interval=interval, progress=False)
            cached_df = stock_cache[cache_key]
            combined_df = pd.concat([cached_df, new_df])
        else:
            # CACHE MISS
            print(f"First load for {ticker}... Fetching full {period} history.")
            combined_df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)

        # 1. Drop any missing data points (NaNs)
        combined_df.dropna(inplace=True)
        # 2. Remove any duplicate timestamps (Yahoo Finance sometimes sends two of the same minute)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        # 3. STRICTLY sort chronologically (Lightweight Charts will crash if not ascending)
        combined_df.sort_index(inplace=True)
        
        # Save the perfectly clean data back to cache
        stock_cache[cache_key] = combined_df

        # Format data for TradingView
        chart_data = []
        for index, row in combined_df.iterrows():
            # Quick safety check to skip rows with missing data (NaN)
            if pd.isna(row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close']):
                continue

            open_val = row['Open'].iloc[0] if isinstance(row['Open'], pd.Series) else row['Open']
            high_val = row['High'].iloc[0] if isinstance(row['High'], pd.Series) else row['High']
            low_val = row['Low'].iloc[0] if isinstance(row['Low'], pd.Series) else row['Low']
            close_val = row['Close'].iloc[0] if isinstance(row['Close'], pd.Series) else row['Close']
            
            chart_data.append({
                "time": int(index.timestamp()), 
                "open": float(open_val),
                "high": float(high_val),
                "low": float(low_val),
                "close": float(close_val)
            })
            
        return jsonify(chart_data)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return jsonify([]) # Return empty array if it fails so the frontend doesn't crash
    
@app.route('/api/news/<ticker>')
def get_news_sentiment(ticker):
    print(f"Fetching news and calculating custom sentiment for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        results = []
        
        # Grab the top 5 most recent articles
        for item in news_items[:5]:
            content = item.get('content', item)  # new yfinance API nests data under 'content'
            title = content.get('title', '')
            click_through = content.get('clickThroughUrl') or content.get('canonicalUrl') or {}
            link = click_through.get('url', '')
            
            if not title:
                continue
                
            # 3. Use YOUR custom method to get the final_score
            # Note: You might need to pass the title as a list [title] depending 
            # on how your tokenizer handles single strings vs batches
            score = analyser.get_sentiment_score([title]) 
            
            # Format based on your score calculation (Pos - Neg)
            if score > 0.05:
                sentiment = "good"
            elif score < -0.05:
                sentiment = "bad"
            else:
                sentiment = "ok"
                
            results.append({
                "title": title,
                "link": link,
                "sentiment": sentiment,
                # Convert numpy float to standard Python float for JSON serialization
                "score": float(score) 
            })
            
        return jsonify(results)
    except Exception as e:
        print(f"Error fetching or analyzing news: {e}")
        return jsonify({"error": "Could not fetch or analyze news"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)