import numpy as np
import tensorflow as tf
from predictor import prepare_data

# Set the timeframe you want to test (e.g., '1m', '5m', '1h')
timeframess = ['1m', '5m', '15m', '1h', '1d']
def find_thresholds(timeframe):
    print(f"Loading Autoencoder for {timeframe}...")
    custom_objects = {'mse': tf.keras.losses.MeanAbsoluteError, 'mae': tf.keras.losses.MeanAbsoluteError}
    ae = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    print(f"Loading Training Data from {data_folder}...")
    data = prepare_data(data_folder, timeframe)
    
    if data is None:
        print("Error: Could not load data. Check your folder path!")
        return
        
    x_train = data[0] 
    
    print(f"Calculating anomaly scores for {len(x_train)} historical windows...")
    reconstructions = ae.predict(x_train, batch_size=128)
    
    train_anom_scores = np.mean(np.abs(x_train - reconstructions), axis=(1, 2))
    
    p90 = np.percentile(train_anom_scores, 90)
    p95 = np.percentile(train_anom_scores, 95)
    p99 = np.percentile(train_anom_scores, 99)
    max_val = np.max(train_anom_scores)
    
    print("\n" + "="*50)
    print(f"  ANOMALY THRESHOLDS FOR {timeframe.upper()} TIMEFRAME")
    print("="*50)
    print(f"  90th Percentile (Elevated Volatility) : {p90:.4f}")
    print(f"  95th Percentile (Warning Zone)        : {p95:.4f}")
    print(f"  99th Percentile (BLACK SWAN)          : {p99:.4f}")
    print(f"  Absolute Maximum in Training          : {max_val:.4f}")
    print("="*50)
    print(f"Recommendation: Set your UI threshold to {p99:.4f}")

if __name__ == "__main__":
    for t in timeframess:
        data_folder = rf'data\{t}' 
        model_path = rf'src\models\autoencoder_{t}.keras'
        print(f"timeframe {t}")
        find_thresholds(t)

#used ai to find the 99th percentile of autoencoder prediction values, this allows me to look for a black swan event
#unfortunately used it because i am tired
black_swan_timeframes = {'1m': 1.7956, '5m': 2.5867, '15m': 2.7912, '1h': 2.5130, '1d': 0.8591}
