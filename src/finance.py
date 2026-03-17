import yfinance as yf, matplotlib.pyplot as plt, pandas as pd, os, numpy as np

def apply_triple_barrier(df, forward_window, profit_factor, stop_factor):
    signals = np.ones(len(df))
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    atrs = df['ATR_pct'].values 

    for i in range(len(df) - forward_window):
        current_close = closes[i]
        
        # multiply current price by atr factor
        profit_target = current_close * (1 + (profit_factor * atrs[i]))
        stop_loss = current_close * (1 - (stop_factor * atrs[i]))

        hit = False
        for j in range(1, forward_window + 1):
            if highs[i+j] >= profit_target:
                signals[i] = 2 # buy
                hit = True
                break
            elif lows[i+j] <= stop_loss:
                signals[i] = 0 # sell
                hit = True
                break

        if not hit:
            signals[i] = 1 # hold

    return signals

def compute_indicators_and_pct(ticker, df, vix_series):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.index.name = None 

    df['Target_Close'] = df['Close']
    
    #high-low spread

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
    
    #gives current close relative to the 14 day period, giving a scale between 0 and 1
    df['Stoch_K'] = (df['Close'] - low_14) / (high_14 - low_14)
    
    # 3 day moving average of stoch k
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # merging vix together
    df['Date_Join'] = df.index.date
    vix_temp = vix_series.copy()
    vix_temp.index = vix_temp.index.date
    df = df.join(vix_temp, on='Date_Join')
    df['VIX'] = df['VIX'].ffill()
    df.drop(columns=['Date_Join'], inplace=True)
    
    prev_close = df['Close'].shift(1)
    
    # macd, bollinger bands, roc
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

    # log returns instead of target price so model doesnt echo output
    df_final = pd.DataFrame(index=df.index)
    df_final['Open'] = np.log(df['Open'] / prev_close)
    df_final['High'] = np.log(df['High'] / prev_close)
    df_final['Low'] = np.log(df['Low'] / prev_close)
    df_final['Close'] = np.log(df['Close'] / prev_close)

    # calculate atr
    high_low = df['High'] - df['Low']
    high_prev_close = np.abs(df['High'] - df['Close'].shift(1))
    low_prev_close = np.abs(df['Low'] - df['Close'].shift(1))
    
    # for each row, find the maximum out of the 3
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # calculate 14 day average
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # convert to pct to match log returns
    df['ATR_pct'] = df['ATR'] / df['Close']   
    
    #macro trend distance
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Trend_Dist_50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
 
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Trend_Dist_200'] = (df['Close'] - df['EMA_200']) / df['EMA_200']
    
    # checks how far apart the 2 emas are
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
        # check for rolling vwap, add a small value to stop div by 0
        vol_sum = df['Volume'].rolling(window=24).sum()
        df['Rolling_VWAP'] = (typical_price * df['Volume']).rolling(window=24).sum() / (vol_sum + 1e-8)
    else:
        #if there is no volume we just use the moving average of the price
        #this acts identically to VWAP for the neural network's purposes.
        df['Rolling_VWAP'] = typical_price.rolling(window=24).mean()
    df['VWAP_Dist'] = (df['Close'] - df['Rolling_VWAP']) / df['Rolling_VWAP']


    # --- ADD TARGETS & REMAINING FEATURES ---
    df_final['Target_Close'] = df['Target_Close']
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
    df_final['Signal'] = apply_triple_barrier(df, 60, 7.0, 6.7)
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
    base_dir = r"C:\Users\supah\OneDrive\Documents\School\Uni Projects\StockPrediction\data"
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
                    
                clean_df = compute_indicators_and_pct(t, raw_df, vix_series)
                
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
                    
                df = pd.read_csv(f'C:/Users/supah/OneDrive/Documents/School/Uni Projects/StockPrediction/data/1h/{safe_ticker}.csv')
                print(df['Signal'].value_counts(normalize=True) * 100)

            except Exception as e:
                print(f"  Error with {t}: {e}")

    print("\nData download complete. Files saved in separate folders.")