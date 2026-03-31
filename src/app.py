import yfinance as yf, pandas as pd, numpy as np, pickle, tensorflow as tf
import requests, os, json, time
from datetime import datetime
from flask import Flask, render_template, jsonify
from functools import lru_cache
from dotenv import load_dotenv
from finance import compute_indicators_and_pct
from predictor import SentimentAnalyser
from db import TradeHistoryDB

load_dotenv()
app = Flask(__name__)

stock_cache = {}
model_cache = {}

rr_ratio = 1.5
max_open_trades = 8
risk_per_trade = 150.0 
point_value_per_unit = 1.0

black_swan_timeframes = {'1m': 1.7956, '5m': 2.5867, '15m': 2.7912, '1h': 2.5130, '1d': 0.8591}
barrier_targets = {'1m': 3.5, '5m': 5.0, '15m': 5.5, '1h': 6.0, '1d': 4.5}
timeframe_timeouts = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '1d': 86400}
max_duration = {'1m': 120, '5m': 60, '15m': 32, '1h': 24, '1d': 20} 
labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
period_map = {'1m': '7d', '5m': '60d', '15m': '60d', '1h': '730d', '1d': 'max'}
window_sizes = {'1m': 30, '5m': 48, '15m': 32, '1h': 24, '1d': 20}
recommended_thresholds = {'1m': {'BUY': 0.4496, 'SELL': 0.4404}, '5m': {'BUY': 0.4649, 'SELL': 0.5310},
                          '15m': {'BUY': 0.4448, 'SELL': 0.4240}, '1h': {'BUY': 0.3884, 'SELL': 0.4237},
                          '1d': {'BUY': 0.7132, 'SELL': 0.4974}
                          }

analyser = SentimentAnalyser()
trade_db = TradeHistoryDB()

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/stock')
def home():
    return render_template('index.html')

def robust_download(ticker, interval, period, retries=3):
    for attempt in range(retries):
        try:
            df = yf.download(ticker, interval=interval, period=period, progress=False)
            if not df.empty and len(df) > 10:
                return df
        except Exception as e:
            print(f"Yahoo download failed (Attempt {attempt+1}/{retries}): {e}")
        time.sleep(2)
    return pd.DataFrame()

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

def get_models(timeframe):
    if timeframe not in model_cache:
        model_cache[timeframe] = {
            'predictor': tf.keras.models.load_model(f'models/predictor_{timeframe}.keras'),
            'autoencoder': tf.keras.models.load_model(f'models/autoencoder_{timeframe}.keras'),
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

        cache_key = f"{ticker}_{timeframe}"
        if cache_key in stock_cache and len(stock_cache[cache_key]) > 60:
            raw_df = stock_cache[cache_key].copy()
        else:
            raw_df = robust_download(ticker, timeframe, period_map.get(timeframe, '60d'))
            stock_cache[cache_key] = raw_df

        if raw_df.empty:
            print(f"{ticker}-{timeframe} data collection failed. returning 0s for safety")
            return jsonify({'signal': 'HOLD', 'probabilities': {'buy':0, 'hold':1, 'sell':0}, 'sentiment': 0, 'anomaly': 0})
        
        curr_hash = round(time.time() / 300)
        vix_series = fetch_cached_vix(curr_hash)

        df = compute_indicators_and_pct(ticker, raw_df, vix_series, timeframe)
        df.dropna(inplace=True)

        if len(df) < 60:
            return jsonify({"error": "Not enough data for prediction"}), 400

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
        finbert_sentiment_score = float(analyser.get_sentiment_score(headlines)) if headlines else 0.0
        sentiment_input = np.array([[finbert_sentiment_score]])

        current_price = float(raw_df['Close'].iloc[-1])
        
        high_low = raw_df['High'] - raw_df['Low']
        high_close = np.abs(raw_df['High'] - raw_df['Close'].shift())
        low_close = np.abs(raw_df['Low'] - raw_df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        current_atr = float(tr.rolling(12).mean().iloc[-1])

        
        local_tz = datetime.now().astimezone().tzinfo
        
        if raw_df.index.tz is not None:
            yf_utc_index = raw_df.index.tz_convert('UTC').tz_localize(None)
        else:
            yf_utc_index = raw_df.index
        
        open_trades = trade_db.get_trades_by_status('OPEN')
        #print(f"[TRADE CHECK] Found {len(open_trades)} OPEN trades")
        
        for trade in open_trades:
            if trade['ticker'] == ticker:
                #print(f"[TRADE CHECK] Processing {trade['signal']} trade #{trade['id']}: {trade['entry_price']} | TP: {trade['target_price']} | SL: {trade['stop_loss']}")
                status = 'OPEN'
                
                yf_utc_index_local = yf_utc_index
                
                try:
                    local_trade_time = pd.to_datetime(trade['date_time'])
                    
                    if local_trade_time.tz is not None:
                        trade_time_utc = local_trade_time.tz_convert('UTC').tz_localize(None)

                    else:
                        trade_time_utc = local_trade_time.tz_localize(local_tz).tz_convert('UTC').tz_localize(None)
                
                except Exception as e:
                    print(f"[TRADE CHECK] Timezone error: {e}, using fallback (naive time)")
                    trade_time_utc = pd.to_datetime(trade['date_time'])
                    if trade_time_utc.tz is not None:
                        trade_time_utc = trade_time_utc.tz_localize(None)

                # print(f"[TRADE CHECK] Trade time (UTC naive): {trade_time_utc}")
                # print(f"[TRADE CHECK] Data range: {yf_utc_index_local.min()} to {yf_utc_index_local.max()}")
                
                try:
                    post_trade_df = raw_df[yf_utc_index_local >= trade_time_utc]
                except TypeError as e:
                    print(f"[TRADE CHECK] Index comparison error: {e}. Retrying with naive conversion...")
                    # Force both to tz-naive (use local copy to avoid affecting other trades)
                    if yf_utc_index_local.tz is not None:
                        yf_utc_index_local = yf_utc_index_local.tz_localize(None)
                    if trade_time_utc.tz is not None:
                        trade_time_utc = trade_time_utc.tz_localize(None)
                    post_trade_df = raw_df[yf_utc_index_local >= trade_time_utc]
                
                if not post_trade_df.empty:
                    tp_hit_first = None
                    
                    print(f"[TRADE CHECK #{trade['id']}] {trade['signal']} | Entry: {trade['entry_price']} | TP: {trade['target_price']} | SL: {trade['stop_loss']} | Candles: {len(post_trade_df)}")
                    
                    for candle_count, (idx, row) in enumerate(post_trade_df.iterrows(), 1):
                        high = float(row['High'])
                        low = float(row['Low'])
                        close = float(row['Close'])
                        
                        if trade['signal'] == 'BUY':
                            tp_hit = high >= trade['target_price']
                            sl_hit = low <= trade['stop_loss']
                            
                            if tp_hit or sl_hit:
                                print(f"  [Candle {candle_count}] H:{high:.2f} L:{low:.2f} C:{close:.2f} | TP_hit:{tp_hit} SL_hit:{sl_hit}")
                            
                            if tp_hit and sl_hit:
                                tp_distance = abs(trade['target_price'] - high)
                                sl_distance = abs(trade['stop_loss'] - low)
                                tp_hit_first = (tp_distance <= sl_distance)
                                print(f"  [BOTH HIT] TP_dist:{tp_distance:.4f} SL_dist:{sl_distance:.4f} => TP_first:{tp_hit_first}")
                            elif tp_hit:
                                tp_hit_first = True
                                print(f"  [TP HIT] at {high:.2f}")
                            elif sl_hit:
                                tp_hit_first = False
                                print(f"  [SL HIT] at {low:.2f}")
                        
                        elif trade['signal'] == 'SELL':
                            tp_hit = low <= trade['target_price']
                            sl_hit = high >= trade['stop_loss']
                            
                            if tp_hit or sl_hit:
                                print(f"  [Candle {candle_count}] H:{high:.2f} L:{low:.2f} C:{close:.2f} | TP_hit:{tp_hit} SL_hit:{sl_hit}")
                            
                            if tp_hit and sl_hit:
                                tp_distance = abs(low - trade['target_price'])
                                sl_distance = abs(high - trade['stop_loss'])
                                tp_hit_first = (tp_distance <= sl_distance)
                                print(f"  [BOTH HIT] TP_dist:{tp_distance:.4f} SL_dist:{sl_distance:.4f} => TP_first:{tp_hit_first}")
                            elif tp_hit:
                                tp_hit_first = True
                                print(f"  [TP HIT] at {low:.2f}")
                            elif sl_hit:
                                tp_hit_first = False
                                print(f"  [SL HIT] at {high:.2f}")
                        
                        if tp_hit_first is not None:
                            status = 'SUCCESSFUL' if tp_hit_first else 'FAILED'
                            print(f"[TRADE CHECK #{trade['id']}] CLOSED as {status}")
                            break
                    
                    if status == 'OPEN' and len(post_trade_df) >= max_duration.get(timeframe, 120):
                        curr_price = float(post_trade_df['Close'].iloc[-1])
                        if trade['signal'] == 'BUY':
                            status = 'SUCCESSFUL' if curr_price > trade['entry_price'] else 'FAILED'
                        elif trade['signal'] == 'SELL':
                            status = 'SUCCESSFUL' if curr_price < trade['entry_price'] else 'FAILED'
                        print(f"[TRADE CHECK] Max duration hit, closing with status: {status}")
                    
                    if status != 'OPEN':
                        sl_distance = abs(trade['entry_price'] - trade['stop_loss'])
                        
                        if sl_distance > 0:
                            raw_units = risk_per_trade / (sl_distance * point_value_per_unit)
                            units = int(round(raw_units, 0))
                        else:
                            units = 1

                        if status == 'SUCCESSFUL':
                            exit_price = trade['target_price']
                        elif status == 'FAILED':
                            exit_price = trade['stop_loss']
                        else:
                            exit_price = float(post_trade_df['Close'].iloc[-1])

                        if trade['signal'] == 'BUY':
                            points_captured = exit_price - trade['entry_price']
                        elif trade['signal'] == 'SELL':
                            points_captured = trade['entry_price'] - exit_price

                        strategy = points_captured * units * point_value_per_unit
                        print(f"[TRADE CHECK] Updating trade {trade['id']}: {status}, PnL={strategy}")

                        trade_db.update_trade_status(trade['id'], status, strategy)

        probs = models['predictor'].predict([x_input, sentiment_input, anomaly_scaled], verbose=0)[0]

        curr_shift = 3.5 if float(anomaly_raw) < black_swan_timeframes[timeframe] else 1.5

        try:
            llm_data = fetch_llm_sentiment(ticker, "gpt-5.4-mini")
            openai_score = float(llm_data.get("rating", 0.0))
            print(f"Openai score: {openai_score}, FinBERT score: {finbert_sentiment_score}")
            final_sentiment = (finbert_sentiment_score * 0.3) + (openai_score * 0.7)
        except Exception as e:
            print(f"LLM sentiment failed, defaulting to 0: {e}")
            openai_score = 0.0
            final_sentiment = finbert_sentiment_score

        adj_buy, adj_hold, adj_sell = analyser.apply_sentiment_scaling(probs[2], probs[1], probs[0], final_sentiment, curr_shift)
        prob_list = [adj_sell, adj_hold, adj_buy]
        predicted_class = int(np.argmax([prob_list[0], prob_list[1], prob_list[2]]))
        signal = labels[predicted_class]
        print(f"Buy: {probs[2]:.3f} | Sell: {probs[0]:.3f} | Hold: {probs[1]:.3f} | ATR: {current_atr:.4f}")
        
        if signal in ['BUY', 'SELL'] and prob_list[predicted_class] >= recommended_thresholds.get(timeframe, {}).get(signal.upper(), 0.5):
            # Get ALL trades (not just open) to check for recent duplicates
            all_trades = trade_db.get_all_trades()
            ticker_trades = [t for t in all_trades if t['ticker'] == ticker]
            currently_open = [t for t in ticker_trades if t['status'] == 'OPEN']

            if len(currently_open) <= max_open_trades:
                # Check if a trade was placed too recently (within timeframe window)
                # Use current system time, not candle time
                now = datetime.now()
                recent_trade_exists = False
                
                if ticker_trades:  # If there are any trades for this ticker
                    most_recent_trade = max(ticker_trades, key=lambda t: pd.to_datetime(t['date_time']))
                    last_trade_time = pd.to_datetime(most_recent_trade['date_time'])
                    
                    # Remove timezone info from both for comparison
                    if last_trade_time.tz is not None:
                        last_trade_time = last_trade_time.tz_localize(None)
                    now_naive = now.replace(tzinfo=None)
                    
                    time_since_last_trade = (now_naive - last_trade_time).total_seconds()
                    timeframe_timeout = timeframe_timeouts.get(timeframe, 60)
                    
                    print(f"[DEBUG] Last trade for {ticker}: {last_trade_time}, Time since: {time_since_last_trade:.1f}s, Timeout: {timeframe_timeout}s")
                    
                    if time_since_last_trade < timeframe_timeout:
                        print(f"[SKIP] Recent trade found for {ticker} ({time_since_last_trade:.1f}s ago). Skipping to avoid spam.")
                        recent_trade_exists = True
                
                if not recent_trade_exists:
                    profit_factor = (barrier_targets.get(timeframe, 3.5))
                    
                    profit_dist = profit_factor * current_atr
                    stop_dist = profit_dist / rr_ratio
                    
                    if signal == 'BUY':
                        target_price = current_price + profit_dist
                        stop_loss = current_price - stop_dist
                    else: # SELL
                        target_price = current_price - profit_dist
                        stop_loss = current_price + stop_dist

                    print(f"[INSERT] Placing {signal} trade for {ticker} at {current_price}")
                    trade_db.insert_trade(ticker, signal, prob_list[predicted_class], current_price, target_price, stop_loss, timeframe, 0)

        return jsonify({
            'signal': signal,
            'probabilities': {
                'sell': float(adj_sell),
                'hold': float(adj_hold),
                'buy':  float(adj_buy)
            },
            'winning_prob': round(float(max(adj_buy, adj_hold, adj_sell)), 2),
            'sentiment': final_sentiment,
            'anomaly': float(anomaly_raw)
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/info')
def info():
    return render_template('info.html')

def fetch_llm_sentiment(ticker, model_name):
    api_key = os.getenv('OPENAI_API_KEY', '') if api_key is None or api_key.strip() == "" else api_key
    if not api_key:
        raise RuntimeError("OpenAI API key not provided. Set it as an environment variable or pass it as an argument.")
    base_url = "https://api.openai.com/v1/chat/completions"

    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    current_price = hist['Close'].iloc[-1] if not hist.empty else "Unknown"

    headlines = [item.get('content', item).get('title') for item in stock.news[:3] if item.get('content', item).get('title')]
    news_context = " | ".join(headlines)

    SYSTEM_PROMPT = f"""
    You are an expert quantitative financial analyst. Analyze the market for {ticker}.
    - Live price: {current_price}
    - Latest headlines: {news_context}
    
    Provide exactly 3 things:
    1. "recommendation": Choose one: [Strong Sell, Sell, Neutral, Buy, Strong Buy].
    2. "description": A 2-3 sentence summary.
    3. "rating": A float from -0.1000 to 0.1000 based on the news, be very precise with the numbers (e.g -0.0587).
    Respond ONLY in valid, strict, JSON format.
    """
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    messages = [
        {"role": "system", "content": "You output strict JSON."},
        {"role": "user", "content": SYSTEM_PROMPT}
    ]

    resp = requests.post(base_url, json={"model": model_name, "messages": messages, "response_format": {"type": "json_object"}}, headers=headers, timeout=8)
    
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error: {resp.status_code}")
    
    raw_content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "{}")
    return json.loads(raw_content) # Convert JSON string to Python dictionary

@app.route('/api/recommendation/<ticker>/<gpt_model>')
def frontend_recommendation(ticker, gpt_model):
    try:
        data = fetch_llm_sentiment(ticker, gpt_model)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/trade-history')
def trade_history():
    trades = trade_db.get_all_trades()
    completed_trades = trade_db.get_success_ratio()
    success_count = sum(1 for trade in completed_trades if trade['status'] == 'SUCCESSFUL')
    ratio = success_count / len(completed_trades) if completed_trades else 0
    return render_template('trade_history.html', trades=trades, ratio=ratio)

if __name__ == '__main__':
    api = input("Enter API Key for openai (or press Enter to skip): ").strip()
    if api:
        os.environ['OPENAI_API_KEY'] = api
    app.run(debug=True, port=5000, use_reloader=False)