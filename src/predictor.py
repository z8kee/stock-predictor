import tensorflow as tf, numpy as np, pandas as pd, pickle, glob, os, gc
from keras.models import Model
from keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Dropout, Concatenate, Input, RepeatVector, TimeDistributed, LayerNormalization, LeakyReLU
from transformers import BertTokenizer, TFBertForSequenceClassification
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

class SentimentAnalyser:
    def __init__(self, max_weight):
        print("Loading finbert")

        self.tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = TFBertForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.max_weight = max_weight

    def get_sentiment_score(self, headlines):
        if not headlines:
            return 0.0

        tokenised_input = self.tokenizer(headlines, padding=True, truncation=True, return_tensors='tf')

        output = self.model(tokenised_input)
        probs = tf.nn.softmax(output.logits, axis=-1)

        pos_score = probs[0][0].numpy()
        neg_score = probs[0][1].numpy()

        final_score = pos_score - neg_score

        return final_score

    def apply_sentiment_scaling(self, prob_buy, prob_sell, sentiment_score):

      # start with default multipliers which has no change
      buy_multiplier = 1.0
      sell_multiplier = 1.0

       # good news means boost buy, reduce sell, and vice versa for bad news
      if sentiment_score > 0:
        buy_multiplier = 1.0 + (sentiment_score * self.max_weight)
        sell_multiplier = 1.0 - (sentiment_score * self.max_weight)

      elif sentiment_score < 0:
        sell_multiplier = 1.0 + (abs(sentiment_score) * self.max_weight)
        buy_multiplier = 1.0 - (abs(sentiment_score) * self.max_weight)

      # apply multipliers
      adjusted_buy = prob_buy * buy_multiplier
      adjusted_sell = prob_sell * sell_multiplier

      # ensure probabilities cannot exceed 0.99
      adjusted_buy = min(0.99, adjusted_buy)
      adjusted_sell = min(0.99, adjusted_sell)

      return adjusted_buy, adjusted_sell
    

def prepare_data(interval_folder, timeframe_name, window_size=60, forecast_horizon=1):
    files = glob.glob(os.path.join(interval_folder, "*.csv"))

    if not files:
      return None

    feature_cols = ['Open', 'High', 'Low', 'Close',
                    'VIX', 'EMA_Dist', '50TD', '200TD',
                    'EMA_Spread', "Rolling_WVAP", "VWAP_Dist", "Hour_Sin",
                    "Hour_Cos", "is_new_york", "is_london", "is_asia",
                    'Volatility', 'RSI', 'ROC', 'BB_Position',
                    "Stoch_K", "Stoch_D"]

    x_train_list, y_trend_train, y_signal_train = [], [], []
    x_test_list, y_trend_test, y_signal_test = [], [], []

    # initialize scalers
    f_scaler = RobustScaler()
    t_scaler = RobustScaler()
    scalers_fitted = False
    all_feats = []
    target = []

    # scale all files together
    for f in files:
      temp_df = pd.read_csv(f)
      all_feats.append(temp_df[feature_cols].values.astype('float32'))
      target.append(temp_df[['Close']].values.astype('float32'))

    f_scaler.fit(np.concatenate(all_feats, axis=0))
    t_scaler.fit(np.concatenate(target, axis=0))

    for f in files:
      temp_df = pd.read_csv(f)

      feats = temp_df[feature_cols].values.astype('float32')
      target_price = temp_df[['Close']].values.astype('float32') # regression
      target_signal = temp_df['Signal'].values # classification

      x_scaled = f_scaler.transform(feats)
      y_trend_scaled = t_scaler.transform(target_price)

      x_file, y_t_file, y_s_file = [], [], []
      for i in range(window_size, len(x_scaled) - forecast_horizon):
          x_file.append(x_scaled[i - window_size : i])
          y_t_file.append(y_trend_scaled[i : i + forecast_horizon])
          y_s_file.append(target_signal[i : i + forecast_horizon])

      split = int(len(x_file) * 0.8)
      x_train_list.append(np.array(x_file[:split]))
      y_trend_train.append(np.array(y_t_file[:split]))
      y_signal_train.append(np.array(y_s_file[:split]))

      x_test_list.append(np.array(x_file[split:]))
      y_trend_test.append(np.array(y_t_file[split:]))
      y_signal_test.append(np.array(y_s_file[split:]))

    X_train = np.concatenate(x_train_list)
    Y_trend_train = np.concatenate(y_trend_train)
    Y_signal_train = np.concatenate(y_signal_train)

    X_test = np.concatenate(x_test_list)
    Y_trend_test = np.concatenate(y_trend_test)
    Y_signal_test = np.concatenate(y_signal_test)

    with open(f'scaler_features_{timeframe_name}.pkl', 'wb') as f:
        pickle.dump(f_scaler, f)

    with open(f'scaler_target_{timeframe_name}.pkl', 'wb') as f:
        pickle.dump(t_scaler, f)

    print(f"\n{timeframe_name} Data Prepared:")
    print(f"  X shape: {X_train.shape}")
    print(f"  Y trend shape: {Y_trend_train.shape}")
    print(f"  Y signal shape: {Y_signal_train.shape}")

    return X_train, Y_trend_train, Y_signal_train, X_test, Y_trend_test, Y_signal_test

def build_model(n_timesteps, n_features):
    input_layer = Input(shape=(n_timesteps, n_features), name="input_layer")
    
    # cnn-lstm neural network
    x = Conv1D(filters=16, kernel_size=3, padding ='same')(input_layer)
    x = LeakyReLU(negative_slope=0.05)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = LSTM(32, return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.005))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    sentiment_input = Input(shape=(1,), name='sentiment_input')
    anomaly_input = Input(shape=(1,), name='anomaly_input')

    merged = Concatenate()([x, sentiment_input, anomaly_input])

    merged = Dense(32)(merged)
    merged = LeakyReLU(negative_slope=0.05)(merged)
    merged = Dropout(0.1)(merged)

    signal_output = Dense(3, activation='softmax', name='signal_output')(merged)

    model = Model(inputs=[input_layer, sentiment_input, anomaly_input],
                  outputs=signal_output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model

def prepare_inputs(x_data, sentiment_data, anomaly_data):
    return [x_data, sentiment_data.reshape(-1, 1), anomaly_data.reshape(-1, 1)]

def build_autoencoder(n_timesteps, n_features):
    # input: (60, 11)
    input_layer = Input(shape=(n_timesteps, n_features))

    # we compress the 60 steps of data into a single vector
    encoder = LSTM(64, return_sequences=True)(input_layer)
    encoder = LSTM(32, return_sequences=False)(encoder)

    # we take that 32-unit vector and try to expand it back to (60, 11)
    decoder = RepeatVector(n_timesteps)(encoder)
    decoder = LSTM(32, return_sequences=True)(decoder)
    decoder = LSTM(64, return_sequences=True)(decoder)

    # the output must match the exact shape of the input
    output_layer = TimeDistributed(Dense(n_features))(decoder)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mae')
    return model

def prepare_anomaly_data(df, window_size=30):
    feat_cols = ['Open', 'High', 'Low', 'Close', 'VIX', 'RSI', 'EMA_Dist', 'Volatility']
    data = df[feat_cols].values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.isnull().values.any():
        print("Warning: NaNs still present in the data!")
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i: i + window_size])

    return np.array(sequences)

def black_swan(model, current_window, threshold=0.15):
    prediction = model.predict(current_window, verbose=0)

    loss = np.mean(np.abs(current_window-prediction))
    is_anomaly = loss > threshold

    return is_anomaly, loss

def train():
  base_path = '/content/sample_data/data'
  timeframes = ['1m', '5m', '15m', '1h', '1d']

  for timeframe in timeframes:
      print(f"\n--- Processing Timeframe: {timeframe} ---")
      folder_path = os.path.join(base_path, timeframe)

      data = prepare_data(folder_path, timeframe, forecast_horizon=1)
      if data is None: continue
      x_train, y_train_trend, y_train_signal, x_test, y_test_trend, y_test_signal = data

      print(f"Training Autoencoder for {timeframe}...")
      ae = build_autoencoder(x_train.shape[1], x_train.shape[2])
      ae.fit(x_train,
          x_train,
          epochs=20,
          batch_size=128,
          validation_split=0.1,
          shuffle=False,
          verbose=0)
      ae.save(f'autoencoder_{timeframe}.h5')

      train_recon = ae.predict(x_train)
      train_anom = np.mean(np.abs(x_train - train_recon), axis=(1, 2)).reshape(-1, 1)

      test_recon = ae.predict(x_test)
      test_anom = np.mean(np.abs(x_test - test_recon), axis=(1, 2)).reshape(-1, 1)

      anom_scaler = MinMaxScaler()
      train_anom_scaled = anom_scaler.fit_transform(train_anom)
      test_anom_scaled = anom_scaler.transform(test_anom)

      with open(f'scaler_anom_{timeframe}.pkl', 'wb') as f:
          pickle.dump(anom_scaler, f)

      train_sent = np.zeros((len(x_train), 1))

      y_sig_flat = y_train_signal.flatten()

      class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_sig_flat),
        y=y_sig_flat
      )

      # create the dictionary from the sklearn output
      class_weights_dict = dict(enumerate(class_weights))
      print(f"Sell: {class_weights_dict.get(0, 1.0):.2f}")
      print(f"Hold: {class_weights_dict.get(1, 1.0):.2f}")
      print(f"Buy:  {class_weights_dict.get(2, 1.0):.2f}\n")

      # map the weights to every single sample for the classification head
      signal_sample_weights = np.array([class_weights_dict.get(int(y), 1.0) for y in y_sig_flat])

      print(f"Training Predictor for {timeframe} timeframe...")

      predictor = build_model(x_train.shape[1], x_train.shape[2])
      predictor.fit(
          x=[x_train, train_sent, train_anom_scaled],
          y=y_sig_flat,
          sample_weight=signal_sample_weights,
          validation_split=0.1,
          epochs=30,
          batch_size=64,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
          shuffle=True
      )
      predictor.save(f'predictor_{timeframe}.h5')

      del x_train, y_train_trend, y_train_signal, x_test, y_test_trend, y_test_signal, predictor
      gc.collect()
      tf.keras.backend.clear_session()

      print(f"saved predictor_{timeframe}.h5, autoencoder_{timeframe}.h5")
      print(f"saved scaler_features_{timeframe}.pkl, scaler_target_{timeframe}.pkl, scaler_anom_{timeframe}.pkl")


def find_optimal_thresholds(probabilities, y_true):
    print("Starting PRO-TRADER Threshold Optimization...")
    thresholds = np.arange(0.00, 0.60, 0.02)

    best_trade_score = 0
    best_buy_thresh = 0.33
    best_sell_thresh = 0.33

    # pre-calculate the AIs best choices to speed up the loop
    max_classes = np.argmax(probabilities, axis=1)
    max_probs = np.max(probabilities, axis=1)

    for buy_t in thresholds:
        for sell_t in thresholds:
            simulated_predictions = []
            for i in range(len(probabilities)):
                # must be the best choice and pass threshold
                if max_classes[i] == 2 and max_probs[i] >= buy_t:
                    simulated_predictions.append(2)
                elif max_classes[i] == 0 and max_probs[i] >= sell_t:
                    simulated_predictions.append(0)
                else:
                    simulated_predictions.append(1)

            scores = f1_score(y_true, simulated_predictions, average=None, labels=[0, 1, 2])
            trade_f1 = (scores[0] + scores[2]) / 2.0

            if trade_f1 > best_trade_score:
                best_trade_score = trade_f1
                best_buy_thresh = buy_t + 0.01
                best_sell_thresh = sell_t + 0.01

    print("-" * 30)
    print(f"max f1-score: {best_trade_score:.4f}")
    print(f"best buy threshold:  {best_buy_thresh:.2f}")
    print(f"best sell threshold: {best_sell_thresh:.2f}")
    print("-" * 30)

    return best_buy_thresh, best_sell_thresh


def apply_signal_gap(predictions, min_gap=5):
    filtered = predictions.copy()
    last_signal_idx = -min_gap
    for i in range(len(filtered)):
        if filtered[i] != 1:  # buy or sell
            if i - last_signal_idx < min_gap:
                filtered[i] = 1  # suppress to Hold
            else:
                last_signal_idx = i
    return filtered

def predict_and_visualise(timeframe):
    # 1. Load Model & Scalers
    custom_objects = {'mse': tf.keras.losses.MeanAbsoluteError, 'mae': tf.keras.losses.MeanAbsoluteError}
    model = tf.keras.models.load_model(f'predictor_{timeframe}.h5', custom_objects=custom_objects)
    ae_model = tf.keras.models.load_model(f'autoencoder_{timeframe}.h5', custom_objects=custom_objects)

    with open(f'scaler_target_{timeframe}.pkl', 'rb') as f:
        target_scaler = pickle.load(f)
    with open(f'scaler_anom_{timeframe}.pkl', 'rb') as f:
        anom_scaler = pickle.load(f)

    # unpack testing data
    _, _, _, X_te, Y_te_trend, Y_te_signal = prepare_data('/content/sample_data/data/1h', timeframe)

    # get anomaly scores
    test_ae_preds = ae_model.predict(X_te, verbose=0)
    test_anomaly_raw = np.mean(np.abs(X_te - test_ae_preds), axis=(1, 2)).reshape(-1, 1)
    test_anomaly = anom_scaler.transform(test_anomaly_raw)
    test_sentiment = np.zeros((len(test_anomaly), 1))

    # predict log returns
    signal_output = model.predict([X_te, test_sentiment, test_anomaly])

    # unscale returns
    actual_returns = target_scaler.inverse_transform(Y_te_trend.reshape(-1, 1))

    end_display = 2000
    y_true_price = 100 * np.exp(np.cumsum(actual_returns[:end_display]))

    actual_classes_full = Y_te_signal.flatten()

    probabilities = signal_output
    raw_predictions = np.argmax(probabilities, axis=1)

    # find the best thresholds
    best_buy, best_sell = find_optimal_thresholds(probabilities, actual_classes_full)

    #create predictions from threshold
    max_classes = np.argmax(probabilities, axis=1)
    max_probs = np.max(probabilities, axis=1)

    production_predictions = []
    for i in range(len(probabilities)):
        if max_classes[i] == 2 and max_probs[i] >= 0.45:
            production_predictions.append(2) # Buy
        elif max_classes[i] == 0 and max_probs[i] >= 0.45:
            production_predictions.append(0) # Sell
        else:
            production_predictions.append(1) # Hold

    production_predictions = np.array(production_predictions)

    # visualization
    plt.figure(figsize=(16, 8))
    plt.plot(y_true_price, label='Actual Price (Normalized)', color='#2c3e50', linewidth=2)

    colors = ['#c0392b', '#bdc3c7', '#27ae60']
    labels = ['Sell', 'Hold', 'Buy']

    current_position = 1

    for i in range(end_display):
        sig = production_predictions[i]

        if sig != 1 and sig != current_position:
            plt.scatter(i, y_true_price[i], color=colors[sig], s=100, edgecolors='black', zorder=5)
            current_position = sig

        elif sig == 1:
            current_position = 1

    plt.title(f"Hybrid Strategy: {timeframe} Momentum Forecast (Base 100)")
    plt.legend()
    plt.show()

    # print("\n--- PURE AI PREDICTIONS (No Thresholds) ---")
    # print(classification_report(actual_classes_full, raw_predictions, target_names=labels))

    # print(f"\n--- PRODUCTION PERFORMANCE (Optimized Thresholds: Buy={best_buy:.2f}, Sell={best_sell:.2f}) ---")
    # print(classification_report(actual_classes_full, production_predictions, target_names=labels))


if __name__ == "__main__":
    train()
    predict_and_visualise()