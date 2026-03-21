# app.py

from flask import Flask, render_template, jsonify
import yfinance as yf, pandas as pd, numpy as np, pickle, tensorflow as tf
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

        vix_df = yf.download("^VIX", period="5d", progress=False)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_series = vix_df['Close'].rename("VIX")

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
        predicted_class = int(np.argmax(probs))
        labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        adj_buy, adj_sell = analyser.apply_sentiment_scaling(probs[2], probs[0], sentiment_score)
        
        return jsonify({
            'signal': labels[predicted_class],
            'probabilities': {
                'sell': float(adj_sell),
                'hold': float(probs[1]),
                'buy':  float(adj_buy)
            },
            'sentiment': sentiment_score,
            'anomaly': float(anomaly_raw)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8888, use_reloader=False)
