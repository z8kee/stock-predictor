import yfinance as yf, matplotlib.pyplot as plt, pandas as pd, os, numpy as np

def apply_triple_barrier(df, forward_window, profit_factor, stop_factor):
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    atrs = df['ATR_pct'].values
    n = len(df)
    signals = np.ones(n) # Default to 1 (HOLD)

    # calculate absolute distance for Targets and Stops
    profit_dist = profit_factor * atrs * closes
    stop_dist = stop_factor * atrs * closes

    for i in range(n - forward_window):
        future_highs = highs[i+1 : i+1+forward_window]
        future_lows = lows[i+1 : i+1+forward_window]

        buy_target = closes[i] + profit_dist[i]
        buy_stop = closes[i] - stop_dist[i]

        buy_target_hit = np.argmax(future_highs >= buy_target)
        buy_stop_hit = np.argmax(future_lows <= buy_stop)

        buy_target_triggered = future_highs[buy_target_hit] >= buy_target
        buy_stop_triggered = future_lows[buy_stop_hit] <= buy_stop

        buy_success = False
        if buy_target_triggered:
            if not buy_stop_triggered:
                buy_success = True
            elif buy_target_hit <= buy_stop_hit:
                buy_success = True # target hit before stop loss

        sell_target = closes[i] - profit_dist[i]
        sell_stop = closes[i] + stop_dist[i]

        sell_target_hit = np.argmax(future_lows <= sell_target)
        sell_stop_hit = np.argmax(future_highs >= sell_stop)

        sell_target_triggered = future_lows[sell_target_hit] <= sell_target
        sell_stop_triggered = future_highs[sell_stop_hit] >= sell_stop

        sell_success = False
        if sell_target_triggered:
            if not sell_stop_triggered:
                sell_success = True
            elif sell_target_hit <= sell_stop_hit:
                sell_success = True # target hit before stop loss

        if buy_success and not sell_success:
            signals[i] = 2
        elif sell_success and not buy_success:
            signals[i] = 0
        elif buy_success and sell_success:
            #check which one was hit first
            signals[i] = 2 if buy_target_hit < sell_target_hit else 0

    return signals

def compute_indicators_and_pct(ticker, df, vix_series, interval):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    forward_windows = {'1m': 30, '5m': 48, '15m': 32, '1h': 24, '1d': 20}
    barrier_targets = {'1m': 3.5, '5m': 5.0, '15m': 5.5, '1h': 6.0, '1d': 4.5}
    df.index.name = None 

    
    df['Target_Close'] = df['Close']

    df['HL_Spread'] = (df['High'] - df['Low']) / df['Close']
    
    # technical indicators
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_Dist'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = (100 - (100 / (1 + rs))) / 100

    # stochastic oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    
    # %K line (current close relative to the 14-period range)
    # this natively scales between 0.0 and 1.0
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14)
    
    # %D line (3-period moving average of %K)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # Merge VIX
    df['Date_Join'] = df.index.date
    vix_temp = vix_series.copy()
    vix_temp.index = vix_temp.index.date
    df = df.join(vix_temp, on='Date_Join')
    df['VIX'] = df['VIX'].ffill()
    df.drop(columns=['Date_Join'], inplace=True)
    
    prev_close = df['Close'].shift(1)
    
    # MACD, ROC, Bollinger Bands
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['ROC'] = df['Close'].pct_change(periods=10)
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    df['BB_Std'] = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    # --- FEATURE ENGINEERING (Log Returns for Input) ---
    df_final = pd.DataFrame(index=df.index)
    df_final['Open'] = np.log(df['Open'] / prev_close)
    df_final['High'] = np.log(df['High'] / prev_close)
    df_final['Low'] = np.log(df['Low'] / prev_close)
    df_final['Close'] = np.log(df['Close'] / prev_close)

    # --- CALCULATE ATR (Average True Range) ---
    # 1. True Range components
    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    
    # 2. Find the maximum of the three for each row
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # 3. Calculate the 14-period Average True Range
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 4. Convert to Percentage (so it matches your Log Returns)
    df['ATR_pct'] = df['ATR'] / df['Close']   
    
    #macro trend distance
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Trend_Dist_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    
    # 2. Macro Trend
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Trend_Dist_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    
    # 3. The Reversal Indicator (How far apart are the two EMAs?)
    df['EMA_Spread'] = (df['EMA_50'] - df['EMA_200']) / df['EMA_200']
    
    # cyclical time features
    temp_index = pd.to_datetime(df.index)
    hour = temp_index.hour

    df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24.0)

    # asian settings 12am to 9am
    df['is_asia'] = ((hour >= 0) & (hour < 9)).astype(int)

    # london settings 8am to 4pm
    df['is_london'] = ((hour >= 8) & (hour < 16)).astype(int)

    # nyc settings 1pm to 10pm
    df['is_new_york'] = ((hour >= 13) & (hour < 22)).astype(int)

    # #vwap
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        # Add a tiny epsilon (1e-8) to the denominator to mathematically prevent division by zero
        vol_sum = df['Volume'].rolling(window=24).sum()
        df['Rolling_VWAP'] = (typical_price * df['Volume']).rolling(window=24).sum() / (vol_sum + 1e-8)
    else:
        # FALLBACK: If there is no volume (like in Forex), we just use the moving average of the typical price.
        # This acts identically to VWAP for the neural network's purposes.
        df['Rolling_VWAP'] = typical_price.rolling(window=24).mean()
    df['VWAP_Dist'] = (df['Close'] - df['Rolling_VWAP']) / df['Rolling_VWAP']


    # --- ADD TARGETS & REMAINING FEATURES ---
    df_final['Target_Close'] = df['Target_Close'] # This is the price for the trend line
    df_final['VIX'] = np.log(df['VIX'] / df['VIX'].shift(1).replace(0, np.nan)).fillna(0)
    df_final['RSI'] = df['RSI']
    df_final['EMA_Dist'] = df['EMA_Dist']
    df_final['50TD'] = df['Trend_Dist_50']
    df_final['200TD'] = df['Trend_Dist_200']
    df_final['EMA_Spread'] = df['EMA_Spread']
    df_final['Stoch_K'] = df['Stoch_K']
    df_final['Stoch_D'] = df['Stoch_D']
    df_final['Rolling_WVAP'] = df['Rolling_VWAP']
    df_final['VWAP_Dist'] = df['VWAP_Dist']
    df_final['Volatility'] = df['HL_Spread']
    df_final['Hour_Sin'] = df['Hour_Sin']
    df_final['Hour_Cos'] = df['Hour_Cos']
    df_final['Signal'] = apply_triple_barrier(df, forward_windows[interval], barrier_targets[interval], barrier_targets[interval]/1.5)
    df_final['is_new_york'] = df['is_new_york']
    df_final['is_london'] = df['is_london']
    df_final['is_asia'] = df['is_asia']
    df_final['MACD'] = df['MACD']
    df_final['ROC'] = df['ROC']
    df_final['BB_Position']= df['BB_Position']
    df_final['Ticker'] = ticker
    df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    return df_final.dropna()

if __name__ == "__main__":
    stocks = ["GC=F", "EURUSD=X", "AUDUSD=X", "GBPUSD=X", "USDJPY=X", "^TNX"]

    intervals_config = {
        '1m': '7d',    # max 7 days for 1 minute data
        '5m': '60d',   # max 60 days for 5 minute data
        '15m': '60d',  # max 60 days for 15 minute data
        '1h': '730d',  # max 2 years for 1 hour data
        '1d': 'max'    # max history for daily data
    }

    vix_df = yf.download("^VIX", period="max", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_series = vix_df['Close'].rename("VIX")

    #create a new dir for every timeframe
    base_dir = r"data"
    os.makedirs(base_dir, exist_ok=True)

    for interval, period in intervals_config.items():
        print(f"\nProcessing Interval: {interval}...")
        
        #creating sub folder for each timeframe
        save_dir = os.path.join(base_dir, interval)
        os.makedirs(save_dir, exist_ok=True)
        
        for t in stocks:
            try:
                #download and process data
                raw_df = yf.download(t, interval=interval, period=period, progress=False)
                
                if raw_df.empty:
                    continue
                    
                clean_df = compute_indicators_and_pct(t, raw_df, vix_series, interval)
                
                # drop rows only if vix is null, forward fill vix with 0s instead of deleting it

                if clean_df['VIX'].isnull().all():
                    print(f"  Warning: VIX missing for {t} (filling with 0)")
                    clean_df['VIX'] = 0.0
                else:
                    clean_df['VIX'] = clean_df['VIX'].ffill().bfill()

                #save csvs, clean it once more by replace inf values with nan and dropping it
                safe_ticker = t.replace("=", "")
                save_path = os.path.join(save_dir, f"{safe_ticker}.csv")
                
                clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                clean_df.dropna(inplace=True)
                clean_df.to_csv(save_path)
                print(f"  Saved {t} -> {len(clean_df)} rows")
                    
                df = pd.read_csv(save_path)
                print(df['Signal'].value_counts(normalize=True) * 100)

            except Exception as e:
                print(f"  Error with {t}: {e}")

    print("\nData download complete. Files saved in separate folders.")
