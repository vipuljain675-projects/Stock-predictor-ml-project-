# =====================================================================
# 🛡️ AEGIS MARKETS — BiLSTM Training Pipeline (Version 2 - Stationary)
# =====================================================================
# INSTRUCTIONS FOR GOOGLE COLAB:
# 1. Open Google Colab (https://colab.research.google.com)
# 2. Create a new notebook, change runtime type to T4 GPU.
# 3. Copy this entire script, paste it into the first code cell, and RUN.
# =====================================================================

import os
import sys

# 1. Install Libraries
print("Installing required libraries...")
os.system("pip install -q yfinance scikit-learn pandas numpy matplotlib tensorflow")

import pickle
import json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# =====================================================================
# ⚙️ CONFIGURATION
# =====================================================================
TICKER = "^NSEI"   # Use "^NSEI" for Nifty 50, or "BZ=F" for Brent Crude Oil
# =====================================================================

YEARS_DATA   = 8
LOOKBACK     = 30
EPOCHS       = 120
BATCH_SIZE   = 32
TEST_SPLIT   = 0.15

CLEAN_TICKER = TICKER.replace('^', '').replace('=', '')
ASSET_NAME   = 'Nifty 50' if TICKER == '^NSEI' else 'Brent Crude Oil'
MODEL_FILE   = f'lstm_model_{CLEAN_TICKER}.h5'
SCALER_FILE  = f'scaler_{CLEAN_TICKER}.pkl'
TARGET_SCALER_FILE = f'target_scaler_{CLEAN_TICKER}.pkl'
CONFIG_FILE  = f'feature_config_{CLEAN_TICKER}.json'

START_DATE = (datetime.now() - timedelta(days=YEARS_DATA * 365)).strftime('%Y-%m-%d')
END_DATE   = datetime.now().strftime('%Y-%m-%d')

print(f'\n📈 Training Asset  : {ASSET_NAME} ({TICKER})')
print(f'📅 Historical Range: {START_DATE} to {END_DATE}')
print(f'👁️ Window Lookback  : {LOOKBACK} days')

# =====================================================================
# 📥 DATA DOWNLOAD
# =====================================================================
def safe_download(ticker, start, end, name):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            print(f"  ⚠️  {name}: Empty data")
            return None
        print(f"  ✅ {name:15s}: {len(df)} days fetched")
        return df
    except Exception as e:
        print(f"  ❌ {name} failed: {e}")
        return None

print("\n📥 Fetching asset price records...")
stock_raw = safe_download(TICKER, START_DATE, END_DATE, ASSET_NAME)
if stock_raw is None:
    raise ValueError(f"Failed to download data for {TICKER}")

print("\n📥 Fetching macroeconomic indices...")
if TICKER == '^NSEI':
    macro_tickers = {
        'USDINR=X': 'USD_INR',
        '^INDIAVIX': 'India_VIX',
        'GC=F':     'Gold',
        'BZ=F':     'Crude_Oil',
        '^GSPC':    'SP500',
    }
else:
    macro_tickers = {
        'DX-Y.NYB': 'DXY',
        'GC=F':     'Gold',
        '^GSPC':    'SP500',
        'NG=F':     'NatGas',
        '^VIX':     'VIX',
    }

macro_series_dict = {}
for t, name in macro_tickers.items():
    m_df = safe_download(t, START_DATE, END_DATE, name)
    if m_df is not None:
        close_series = m_df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        macro_series_dict[name] = close_series

# =====================================================================
# 🛠️ FEATURE ENGINEERING (STATIONARY V2 PIPELINE)
# =====================================================================
print("\n🛠️ Processing stationary feature transformations...")
df = stock_raw.copy()

# 1. Price Returns (Stationary relative percentage shifts)
df['Ret_Close'] = df['Close'].pct_change(1)
df['Ret_Open']  = df['Open'].pct_change(1)
df['Ret_High']  = df['High'].pct_change(1)
df['Ret_Low']   = df['Low'].pct_change(1)
df['Ret_Volume'] = df['Volume'].pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

# 2. Moving Averages Ratios (Normalized by price to be stationary)
df['SMA_10'] = df['Close'].rolling(10).mean()
df['SMA_20'] = df['Close'].rolling(20).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

df['Close_to_SMA10'] = df['Close'] / (df['SMA_10'] + 1e-9)
df['Close_to_SMA20'] = df['Close'] / (df['SMA_20'] + 1e-9)
df['Close_to_EMA20'] = df['Close'] / (df['EMA_20'] + 1e-9)
df['Close_to_EMA50'] = df['Close'] / (df['EMA_50'] + 1e-9)

# 3. Normalized Oscillators
delta = df['Close'].diff()
gain  = delta.where(delta > 0, 0).rolling(14).mean()
loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
df['RSI_Scaled'] = df['RSI'] / 100.0  # Normalized between 0 and 1

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Price_Ratio'] = df['MACD'] / (df['Close'] + 1e-9)
df['Momentum_5'] = df['Close'].pct_change(5)

# 4. Volatility & Volume Shifts
std20 = df['Close'].rolling(20).std()
df['BB_Width'] = (4 * std20) / (df['SMA_20'] + 1e-9)
df['Rolling_Vol'] = df['Close'].pct_change().rolling(10).std()

obv = [0]
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
        obv.append(obv[-1] + df['Volume'].iloc[i])
    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
        obv.append(obv[-1] - df['Volume'].iloc[i])
    else:
        obv.append(obv[-1])
df['OBV'] = np.array(obv)
df['OBV_pct'] = df['OBV'].pct_change(1).replace([np.inf, -np.inf], 0.0).fillna(0.0)

# 5. Macro Returns
macro_feature_names = []
for name, series in macro_series_dict.items():
    col_name = f'Macro_{name}_5d'
    pct = series.pct_change(5).rename(col_name)
    df = df.merge(pct, left_index=True, right_index=True, how='left')
    df[col_name] = df[col_name].fillna(0.0)
    macro_feature_names.append(col_name)

df.dropna(inplace=True)

FEATURE_COLS = [
    'Ret_Close', 'Ret_Open', 'Ret_High', 'Ret_Low', 'Ret_Volume',
    'Close_to_SMA10', 'Close_to_SMA20', 'Close_to_EMA20', 'Close_to_EMA50',
    'RSI_Scaled', 'MACD_Price_Ratio', 'Momentum_5', 'BB_Width', 'Rolling_Vol', 'OBV_pct'
] + macro_feature_names

print(f"✅ Preprocessing done. Total Stationary Features: {len(FEATURE_COLS)}")
print(f"📌 Features: {FEATURE_COLS}")

# =====================================================================
# 📊 TRAIN/TEST SPLIT & SCALING
# =====================================================================
data_matrix = df[FEATURE_COLS].values.astype(np.float32)
n_features  = data_matrix.shape[1]

split = int(len(data_matrix) * (1 - TEST_SPLIT))
train_raw = data_matrix[:split]
test_raw  = data_matrix[split:]

# Scale inputs between -1 and 1 since returns are centered around 0
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data = scaler.fit_transform(train_raw)
test_data  = scaler.transform(test_raw)

with open(SCALER_FILE, 'wb') as f:
    pickle.dump(scaler, f)

print(f'\n✅ Scaler fit completed and saved to: {SCALER_FILE}')
print(f'   Train Records: {len(train_data)} | Val Records: {len(test_data)}')

# =====================================================================
# 🔄 CREATE SEQUENCES & TARGET SCALING
# =====================================================================
def create_sequences(scaled_data, raw_close, lookback):
    X, y, prev_close = [], [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, :])
        # Next-day return is target
        ret = (raw_close[i] - raw_close[i - 1]) / (raw_close[i - 1] + 1e-9)
        y.append(ret)
        prev_close.append(raw_close[i - 1])
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(prev_close, dtype=np.float32))

X_train, y_train_ret, prev_close_train = create_sequences(train_data, df['Close'].values[:split], LOOKBACK)
X_test,  y_test_ret,  prev_close_test  = create_sequences(test_data,  df['Close'].values[split:], LOOKBACK)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train_ret.reshape(-1, 1)).flatten().astype(np.float32)
y_test  = target_scaler.transform(y_test_ret.reshape(-1, 1)).flatten().astype(np.float32)

with open(TARGET_SCALER_FILE, 'wb') as f:
    pickle.dump(target_scaler, f)

print(f'✅ Sequence tensors loaded. Shape: {X_train.shape}')
print(f'✅ Target scaling applied. Saved to: {TARGET_SCALER_FILE}')

# =====================================================================
# 🧠 BIDIRECTIONAL LSTM + SELF-ATTENTION MODEL
# =====================================================================
@tf.keras.utils.register_keras_serializable()
class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attn_weight', shape=(input_shape[-1], 1),
            initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(
            name='attn_bias', shape=(input_shape[1], 1),
            initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_model(lookback, n_features):
    inp = Input(shape=(lookback, n_features), name='price_input')
    
    x = Bidirectional(LSTM(64, return_sequences=True), name='bilstm_1')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Bidirectional(LSTM(32, return_sequences=True), name='bilstm_2')(x)
    x = BatchNormalization()(x)
    x = SelfAttention(name='self_attention')(x)
    x = Dropout(0.20)(x)

    x = Dense(32, activation='relu', name='dense_1')(x)
    x = Dropout(0.15)(x)
    out = Dense(1, name='output')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=0.0005), # Slightly lower learning rate for stable convergence
        loss='huber',
        metrics=['mae']
    )
    return model

model = build_model(LOOKBACK, n_features)
model.summary()

# =====================================================================
# 🚀 MODEL TRAINING
# =====================================================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

print('\n🚀 Starting training loop...')
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =====================================================================
# 📊 TRAINING CURVES VISUALIZATION
# =====================================================================
print('\n📈 Visualizing training curves...')
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Huber Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    print("✅ Saved training curves plot as 'training_curves.png'")
except Exception as e:
    print(f"⚠️ Could not plot graphs: {e}")

# =====================================================================
# 📈 EVALUATION & ACCURACY TESTS
# =====================================================================
print('\n📊 Testing model accuracy on validation dataset...')
pred_ret_scaled = model.predict(X_test, verbose=0)
pred_return = target_scaler.inverse_transform(pred_ret_scaled).flatten()

# Check raw variance to ensure it's not dead/flat
pred_std = np.std(pred_return * 100)
print(f"📉 Standard deviation of predictions: {pred_std:.4f}% (Should be > 0.05% for active predictions)")

# Directional Accuracy
dir_correct = np.sign(pred_return) == np.sign(y_test_ret)
directional_acc = np.mean(dir_correct) * 100

# MAPE on reconstructed prices
pred_price = prev_close_test * (1.0 + pred_return)
actual_price = prev_close_test * (1.0 + y_test_ret)
mape = np.mean(np.abs((actual_price - pred_price) / actual_price)) * 100

print(f'📈 Directional Accuracy: {directional_acc:.2f}%')
print(f'📉 MAPE (Mean Abs Err): {mape:.4f}%')

# =====================================================================
# 💾 SAVE ARTIFACTS
# =====================================================================
model.save(MODEL_FILE)
print(f'\n💾 Saved model weights to: {MODEL_FILE}')

config = {
    'ticker': TICKER,
    'lookback': LOOKBACK,
    'features': FEATURE_COLS,
    'n_features': len(FEATURE_COLS),
    'directional_accuracy': float(round(directional_acc, 2)),
    'mape': float(round(mape, 4))
}
with open(CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)
print(f'💾 Saved feature config metadata to: {CONFIG_FILE}')

# =====================================================================
# ⬇️ DOWNLOAD FILES
# =====================================================================
try:
    from google.colab import files
    print('\n📥 Downloading deployment files to local directory...')
    files.download(MODEL_FILE)
    files.download(SCALER_FILE)
    files.download(TARGET_SCALER_FILE)
    files.download(CONFIG_FILE)
    print('🎉 Training complete. Move all 4 files to your project backend!')
except ImportError:
    print('\n⚠️ Local environment detected. Files saved in root directory.')
