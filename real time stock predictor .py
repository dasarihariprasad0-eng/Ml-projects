!pip install -q yfinance tensorflow scikit-learn streamlit

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---- Settings ----
STOCKS = {
    'AAPL': 'Apple Inc.',
    'TSLA': 'Tesla Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms'
}

LOOKBACK = 60
EPOCHS = 30
BATCH_SIZE = 32
FEATURES = [
    'Close','RSI','EMA20','EMA50',
    'MACD','MACD_signal','BB_upper',
    'BB_lower','Volume_change','Price_change'
]

# ---- Indicators ----
def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    loss = loss.replace(0, 1e-10)
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['EMA20'] = close.ewm(span=20).mean()
    df['EMA50'] = close.ewm(span=50).mean()
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['BB_upper'] = sma20 + (2 * std20)
    df['BB_lower'] = sma20 - (2 * std20)
    df['Volume_change'] = volume.pct_change()
    df['Price_change'] = close.pct_change()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# ---- Model ----
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True,
             input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    return model

# ---- Train & Predict ----
def train_and_predict(symbol):
    print(f"\n🔄 Processing {symbol}...")
    df = yf.download(
        symbol,
        period="60d",
        interval="1h",
        auto_adjust=True,
        progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if len(df) < 100:
        print(f"❌ Not enough data for {symbol}")
        return None
    df = add_indicators(df)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=FEATURES, inplace=True)
    data = df[FEATURES].astype(float).values
    if not np.isfinite(data).all():
        print(f"❌ {symbol} has bad values")
        return None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)-1):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i, 0])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if len(X) < 10:
        print(f"❌ Not enough sequences for {symbol}")
        return None
    split = int(len(X) * 0.85)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(
        X[:split], y[:split],
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X[split:], y[split:]),
        verbose=0
    )
    last_seq = scaled[-LOOKBACK:].reshape(
        1, LOOKBACK, len(FEATURES)
    ).astype(np.float32)
    pred_scaled = model.predict(last_seq, verbose=0)
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pred_scaled[0, 0]
    pred_price = scaler.inverse_transform(dummy)[0, 0]
    current = float(df['Close'].iloc[-1])
    predicted = round(float(pred_price), 2)
    change = round(
        ((predicted - current) / current) * 100, 2
    )
    signal = (
        "BUY" if change > 0.5
        else "SELL" if change < -0.5
        else "HOLD"
    )
    rsi = round(float(df['RSI'].iloc[-1]), 2)
    macd = round(float(df['MACD'].iloc[-1]), 4)
    ema20 = round(float(df['EMA20'].iloc[-1]), 2)
    high30 = round(float(df['Close'].max()), 2)
    low30 = round(float(df['Close'].min()), 2)
    prices = df['Close'].values.tolist()
    return {
        'symbol': symbol,
        'name': STOCKS[symbol],
        'current': round(current, 2),
        'predicted': predicted,
        'change': change,
        'signal': signal,
        'rsi': rsi,
        'macd': macd,
        'ema20': ema20,
        'high30': high30,
        'low30': low30,
        'prices': prices
    }

# ---- Run All Stocks ----
results = {}
for stock in STOCKS:
    try:
        r = train_and_predict(stock)
        if r:
            results[stock] = r
    except Exception as e:
        print(f"❌ {stock} failed: {e}")

# ---- Print Results ----
print("\n")
print("=" * 50)
print("        QUANTIX — AI STOCK PREDICTOR")
print("=" * 50)

for s, r in results.items():
    arrow = "▲" if r['change'] > 0 else "▼"
    sig_icon = (
        "🟢" if r['signal'] == 'BUY'
        else "🔴" if r['signal'] == 'SELL'
        else "🟡"
    )
    print(f"""
┌─────────────────────────────────┐
  📊 {r['symbol']} — {r['name']}
  Current Price : ${r['current']}
  Next Hour     : ${r['predicted']}
  Change        : {arrow} {r['change']}%
  Signal        : {sig_icon} {r['signal']}
  RSI           : {r['rsi']}
  MACD          : {r['macd']}
  EMA 20        : ${r['ema20']}
  30D High      : ${r['high30']}
  30D Low       : ${r['low30']}
└─────────────────────────────────┘""")

# ---- Plot All Charts ----
print("\n📊 Generating charts...")
fig, axes = plt.subplots(
    len(results), 1,
    figsize=(14, 4 * len(results))
)
fig.patch.set_facecolor('#05070d')

if len(results) == 1:
    axes = [axes]

for ax, (symbol, r) in zip(axes, results.items()):
    prices = r['prices']
    color = '#00cc66' if r['change'] > 0 else '#ff3d5a'
    ax.set_facecolor('#090d18')
    ax.plot(prices, color=color, linewidth=1.5,
            label=f"{symbol} — ${r['current']}")
    ax.fill_between(
        range(len(prices)),
        prices,
        min(prices),
        alpha=0.15,
        color=color
    )
    # Mark predicted price
    ax.axhline(
        y=r['predicted'],
        color='#4a9aef',
        linewidth=1,
        linestyle='--',
        label=f"Predicted: ${r['predicted']}"
    )
    ax.set_title(
        f"{symbol} | Signal: {r['signal']} | "
        f"Change: {r['change']}%",
        color='#b8cce0',
        fontsize=10,
        pad=8
    )
    ax.tick_params(colors='#2a5080', labelsize=7)
    for spine in ax.spines.values():
        spine.set_color('#0c1e35')
    ax.grid(True, color='#0c1e35', linewidth=0.5)
    ax.legend(
        fontsize=8,
        facecolor='#090d18',
        labelcolor='#4a7090'
    )

plt.tight_layout(pad=2.0)
plt.savefig('quantix_predictions.png',
            dpi=150,
            facecolor='#05070d')
plt.show()
print("\n✅ Chart saved as quantix_predictions.png")
print("✅ All done!")