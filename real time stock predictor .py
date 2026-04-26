import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import warnings

# Page Config
st.set_page_config(page_title="Quantix AI", layout="wide")
warnings.filterwarnings('ignore')

# ---- Settings ----
STOCKS = {
    'AAPL': 'Apple Inc.', 'TSLA': 'Tesla Inc.', 'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corp.', 'AMZN': 'Amazon.com Inc.', 
    'NVDA': 'NVIDIA Corp.', 'META': 'Meta Platforms'
}

LOOKBACK = 60
EPOCHS = 15 # Reduced for faster web demo
BATCH_SIZE = 32
FEATURES = ['Close','RSI','EMA20','EMA50','MACD','MACD_signal','BB_upper','BB_lower','Volume_change','Price_change']

# ---- Functions ----
def add_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean().replace(0, 1e-10)
    
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
    return df.replace([np.inf, -np.inf], np.nan).dropna()

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@st.cache_data(show_spinner=False)
def train_and_predict(symbol):
    df = yf.download(symbol, period="60d", interval="1h", auto_adjust=True, progress=False)
    if len(df) < 100: return None
    
    df = add_indicators(df)
    data = df[FEATURES].astype(float).values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)-1):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i, 0])
    
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    
    last_seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(FEATURES)).astype(np.float32)
    pred_scaled = model.predict(last_seq, verbose=0)
    
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pred_scaled[0, 0]
    predicted = round(float(scaler.inverse_transform(dummy)[0, 0]), 2)
    current = round(float(df['Close'].iloc[-1]), 2)
    change = round(((predicted - current) / current) * 100, 2)
    
    return {
        'symbol': symbol, 'name': STOCKS[symbol], 'current': current,
        'predicted': predicted, 'change': change, 'prices': df['Close'].tolist()
    }

# ---- Streamlit UI ----
st.title("📊 Quantix — AI Stock Predictor")
st.markdown("---")

if st.button('🚀 Run Analysis'):
    results = {}
    progress_bar = st.progress(0)
    
    for i, stock in enumerate(STOCKS):
        with st.spinner(f"Analyzing {stock}..."):
            r = train_and_predict(stock)
            if r: results[stock] = r
        progress_bar.progress((i + 1) / len(STOCKS))

    # Display Metrics in Columns
    cols = st.columns(len(results))
    for col, (s, r) in zip(cols, results.items()):
        col.metric(label=s, value=f"${r['current']}", delta=f"{r['change']}%")

    # Generate Unified Chart
    st.subheader("Price Trends & Predictions")
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)))
    fig.patch.set_facecolor('#0e1117')
    
    if len(results) == 1: axes = [axes]
    
    for ax, (symbol, r) in zip(axes, results.items()):
        color = '#00cc66' if r['change'] > 0 else '#ff3d5a'
        ax.set_facecolor('#0e1117')
        ax.plot(r['prices'], color=color, label=f"Current: ${r['current']}")
        ax.axhline(y=r['predicted'], color='#4a9aef', linestyle='--', label=f"Target: ${r['predicted']}")
        ax.set_title(f"{symbol} Prediction", color='white')
        ax.legend()
        plt.setp(ax.get_xticklabels(), color='#555')
        plt.setp(ax.get_yticklabels(), color='#555')

    st.pyplot(fig)
else:
    st.info("Click the button above to start the AI analysis.")
