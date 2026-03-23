# app.py

from flask import Flask, render_template, jsonify
import yfinance as yf, pandas as pd, numpy as np, pickle, tensorflow as tf, time, requests
from functools import lru_cache
from predictor import SentimentAnalyser
from finance import compute_indicators_and_pct
app = Flask(__name__)
stock_cache = {}

analyser = SentimentAnalyser()
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
            print(f"First load for {ticker}, Fetching full {period} history.")
            combined_df = yf.download(tickers=ticker, period=period, interval=interval, progress=False)

        combined_df.dropna(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        
        stock_cache[cache_key] = combined_df

        chart_data = []
        for index, row in combined_df.iterrows():
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
        return jsonify([])
    
@app.route('/api/news/<ticker>')
def get_news_sentiment(ticker):
    print(f"Fetching news and calculating custom sentiment for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news
        
        results = []
        
        for item in news_items[:5]:
            content = item.get('content', item)
            title = content.get('title', '')
            click_through = content.get('clickThroughUrl') or content.get('canonicalUrl') or {}
            link = click_through.get('url', '')
            
            if not title:
                continue
                
            score = analyser.get_sentiment_score([title]) 
            
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
                "score": float(score) 
            })
            
        return jsonify(results)
    except Exception as e:
        print(f"Error fetching or analyzing news: {e}")
        return jsonify({"error": "Could not fetch or analyze news"}), 500

model_cache = {}

def get_models(timeframe):
    if timeframe not in model_cache:
        custom_objects = {'mse': tf.keras.losses.MeanAbsoluteError, 'mae': tf.keras.losses.MeanAbsoluteError}
        model_cache[timeframe] = {
            'predictor': tf.keras.models.load_model(f'models/predictor_{timeframe}.keras', custom_objects=custom_objects),
            'autoencoder': tf.keras.models.load_model(f'models/autoencoder_{timeframe}.keras', custom_objects=custom_objects),
            'f_scaler': pickle.load(open(f'models/scaler_features_{timeframe}.pkl', 'rb')),
            'anom_scaler': pickle.load(open(f'models/scaler_anom_{timeframe}.pkl', 'rb')),
        }
    return model_cache[timeframe]

@lru_cache(maxsize=1)
def fetch_cached_vix(ttl_hash):
    print("--- Fetching fresh VIX data from Yahoo ---")
    vix_df = yf.download("^VIX", period="5d", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    return vix_df['Close'].rename("VIX")

@app.route('/api/predict/<ticker>/<timeframe>')
def predict(ticker, timeframe):
    try:
        feature_cols = ['Open', 'High', 'Low', 'Close',
                        'VIX', 'EMA_Dist', '50TD', '200TD',
                        'EMA_Spread', 'VWAP_Dist', 'Hour_Sin',
                        'Hour_Cos', 'is_new_york', 'is_london', 'is_asia',
                        'Volatility', 'RSI', 'ROC', 'BB_Position',
                        'Stoch_K', 'Stoch_D']

        period_map = {'1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': 'max'}
        raw_df = yf.download(ticker, interval=timeframe, period=period_map.get(timeframe, '60d'), progress=False)

        curr_hash = round(time.time() / 300)
        vix_series = fetch_cached_vix(curr_hash)

        df = compute_indicators_and_pct(ticker, raw_df, vix_series)
        df.dropna(inplace=True)

        if len(df) < 60:
            return jsonify({"error": "Not enough data for prediction"}), 400

        window_sizes = {'1m': 30, '5m': 48, '15m': 32, '1h': 24, '1d': 20}
        w_size = window_sizes.get(timeframe, 30)

        models = get_models(timeframe)
        window = df[feature_cols].values[-w_size:]  
        window_scaled = models['f_scaler'].transform(window)
        x_input = window_scaled.reshape(1, w_size, 21)

        recon = models['autoencoder'].predict(x_input, verbose=0)
        anomaly_raw = np.mean(np.abs(x_input - recon))
        anomaly_scaled = models['anom_scaler'].transform([[anomaly_raw]])  # (1, 1)

        headlines = []
        for item in yf.Ticker(ticker).news[:5]:
            content = item.get('content', item)
            title = content.get('title', '')
            if title:
                headlines.append(title)
        sentiment_score = float(analyser.get_sentiment_score(headlines)) if headlines else 0.0
        sentiment_input = np.array([[sentiment_score]])

        # 5. Predict
        probs = models['predictor'].predict([x_input, sentiment_input, anomaly_scaled], verbose=0)[0]
        labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        adj_buy, adj_hold, adj_sell = analyser.apply_sentiment_scaling(probs[2], probs[1], probs[0], sentiment_score, 0.30)
        predicted_class = int(np.argmax([adj_sell, adj_hold, adj_buy]))

        return jsonify({
            'signal': labels[predicted_class],
            'probabilities': {
                'sell': float(adj_sell),
                'hold': float(adj_hold),
                'buy':  float(adj_buy)
            },
            'sentiment': sentiment_score,
            'anomaly': float(anomaly_raw)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/api/recommendation/<ticker>')
def get_openai_recommendation(ticker):
    try:
        api_key = "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"
        base_url = "https://api.openai.com/v1/chat/completions"

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        current_price = hist['Close'].iloc[-1] if not hist.empty else "Unknown"

        headlines = []
        for item in stock.news[:3]:
            content = item.get('content', item)
            if content.get('title'):
                headlines.append(content.get('title'))
        news_context = " | ".join(headlines)

        SYSTEM_PROMPT = f"""
        You are an expert quantitative financial analyst. Analyze the current market context for {ticker}.
        - The current live price is: {current_price}
        - The latest news headlines today are: {news_context}
        
        Provide exactly two things:
        1. "recommendation": Strictly choose exactly one of these strings: [Strong Sell, Sell, Hold, Buy, Strong Buy].
        2. "description": A 2-3 sentence summary of today's market structure for this asset based on the price and news provided, justifying your recommendation.
        
        Respond ONLY in valid JSON format.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        messages = [
            {"role": "system", "content": "You are a financial AI agent that outputs strictly in JSON."},
            {"role": "user", "content": SYSTEM_PROMPT}
        ]

        payload = {"model": "gpt-4o-mini", "messages": messages}
        resp = requests.post(base_url, json=payload, headers=headers)
        
        if resp.status_code != 200:
            raise RuntimeError(f"Chat completion failed: {resp.status_code} {resp.text}")
        
        data = resp.json()
        
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    except Exception as e:
        print(f"OpenAI Agent Error: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
