"""
Tesla Stock Price Prediction — Streamlit Deployment App
Loads the trained LSTM model and TSLA.csv for interactive forecasting.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Price Predictor",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 0.85rem; opacity: 0.8; margin-bottom: 0.2rem; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-sub   { font-size: 0.8rem; opacity: 0.7; }
    .pred-box {
        background: #0e3d2f;
        border-left: 4px solid #00c878;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.4rem 0;
        color: #e0ffe8;
    }
    .pred-day   { font-size: 0.75rem; opacity: 0.75; }
    .pred-price { font-size: 1.4rem; font-weight: 600; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
LOOKBACK = 60

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data(path="TSLA.csv"):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

@st.cache_resource
def build_and_train_model(scaled_data, units=64, dropout=0.25):
    """Train LSTM on the fly (no saved .h5 required for cloud deployment)."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.random.set_seed(42)
    np.random.seed(42)

    X, y = [], []
    for i in range(LOOKBACK, len(scaled_data)):
        X.append(scaled_data[i - LOOKBACK:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(-1, LOOKBACK, 1)

    split = int(len(X) * 0.80)
    X_train, y_train = X[:split], y[:split]

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=40, batch_size=32,
              validation_split=0.1, callbacks=[es], verbose=0)
    return model, X, y, split

def predict_n_days(model, last_sequence, n_days, scaler):
    predictions = []
    current_seq = last_sequence.copy()
    for _ in range(n_days):
        inp = current_seq.reshape(1, LOOKBACK, 1)
        nxt = model.predict(inp, verbose=0)[0][0]
        predictions.append(nxt)
        current_seq = np.append(current_seq[1:], [[nxt]], axis=0)
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/320px-Tesla_Motors.svg.png", width=120)
    st.title("TSLA Predictor")
    st.markdown("---")
    horizon = st.selectbox("Forecast Horizon", ["1 Day", "5 Days", "10 Days"], index=0)
    n_forecast = int(horizon.split()[0])
    st.markdown("---")
    show_ma = st.checkbox("Show Moving Averages", value=True)
    show_vol = st.checkbox("Show Volume Chart", value=True)
    chart_days = st.slider("Days of history to display", 90, 500, 180, step=30)
    st.markdown("---")
    st.caption("Model: Stacked LSTM  |  Lookback: 60 days  |  Split: 80/20 (temporal)")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("📈 Tesla (TSLA) Stock Price Prediction")
st.caption("Deep Learning LSTM model trained on historical OHLCV data")

# Load data
with st.spinner("Loading TSLA data..."):
    try:
        df = load_data("TSLA.csv")
    except FileNotFoundError:
        st.error("TSLA.csv not found. Please place TSLA.csv in the same directory as this app.")
        st.stop()

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[["Adj Close"]])

# Train model
with st.spinner("Training LSTM model… this may take 1–2 minutes on first load."):
    model, X_all, y_all, split = build_and_train_model(scaled_data)

st.success("Model ready!")

# ── KPI Row ──────────────────────────────────────────────────────────────────
latest_price    = df["Adj Close"].iloc[-1]
prev_price      = df["Adj Close"].iloc[-2]
daily_chg       = ((latest_price - prev_price) / prev_price) * 100
all_time_high   = df["Adj Close"].max()
all_time_low    = df["Adj Close"].min()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Latest Adj Close</div>
        <div class='metric-value'>${latest_price:.2f}</div>
        <div class='metric-sub'>{df.index[-1].strftime('%b %d, %Y')}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    color = "#00ff88" if daily_chg >= 0 else "#ff6b6b"
    sign  = "+" if daily_chg >= 0 else ""
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Daily Change</div>
        <div class='metric-value' style='color:{color}'>{sign}{daily_chg:.2f}%</div>
        <div class='metric-sub'>vs previous close</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>All-Time High</div>
        <div class='metric-value'>${all_time_high:.2f}</div>
        <div class='metric-sub'>{df['Adj Close'].idxmax().strftime('%b %d, %Y')}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    years = (df.index[-1] - df.index[0]).days / 365.25
    total_return = ((latest_price - df["Adj Close"].iloc[0]) / df["Adj Close"].iloc[0]) * 100
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Total Return (IPO)</div>
        <div class='metric-value'>+{total_return:.0f}%</div>
        <div class='metric-sub'>{years:.1f} years since IPO</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Price Chart ───────────────────────────────────────────────────────────────
col_chart, col_pred = st.columns([2, 1])

with col_chart:
    st.subheader(f"Price History — Last {chart_days} Days")
    df_plot = df.tail(chart_days).copy()

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(df_plot.index, df_plot["Adj Close"], color="#00c8ff", linewidth=1.4, label="Adj Close")

    if show_ma and len(df_plot) >= 60:
        ma30  = df_plot["Adj Close"].rolling(30).mean()
        ma60  = df_plot["Adj Close"].rolling(60).mean()
        ax.plot(df_plot.index, ma30, color="#ff9f40", linewidth=1.1, linestyle="--", label="MA-30")
        ax.plot(df_plot.index, ma60, color="#ff6b9d", linewidth=1.1, linestyle=":",  label="MA-60")

    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.yaxis.label.set_color("white")
    ax.set_ylabel("Price (USD)", color="white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=8)
    ax.grid(alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if show_vol:
        st.subheader("Trading Volume")
        fig2, ax2 = plt.subplots(figsize=(10, 2), facecolor="#0e1117")
        ax2.set_facecolor("#0e1117")
        ax2.bar(df_plot.index, df_plot["Volume"], color="#4a90d9", alpha=0.6, width=1)
        ax2.tick_params(colors="white", labelsize=7)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
        for spine in ax2.spines.values():
            spine.set_color("#333333")
        ax2.set_ylabel("Volume", color="white", fontsize=8)
        ax2.grid(alpha=0.1, color="white")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

with col_pred:
    st.subheader(f"🔮 {horizon} Forecast")

    # Run prediction
    last_seq = scaled_data[-LOOKBACK:].reshape(LOOKBACK, 1)
    future_prices = predict_n_days(model, last_seq, n_forecast, scaler)
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=n_forecast + 1)[1:]

    for i, (date, price) in enumerate(zip(future_dates, future_prices)):
        chg = ((price - latest_price) / latest_price) * 100
        sign = "+" if chg >= 0 else ""
        chg_color = "#00ff88" if chg >= 0 else "#ff6b6b"
        st.markdown(f"""<div class='pred-box'>
            <div class='pred-day'>Day {i+1} — {date.strftime('%b %d, %Y')}</div>
            <div class='pred-price'>${price:.2f}
                <span style='font-size:0.9rem; color:{chg_color}'>({sign}{chg:.1f}%)</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Mini forecast chart
    st.markdown("**Forecast Trend**")
    fig3, ax3 = plt.subplots(figsize=(5, 3), facecolor="#0e1117")
    ax3.set_facecolor("#0e1117")
    hist_prices = df["Adj Close"].tail(30).values
    hist_days = list(range(-30, 0))
    fut_days  = list(range(0, n_forecast))
    ax3.plot(hist_days, hist_prices, color="#00c8ff", linewidth=1.5, label="Recent")
    ax3.plot(fut_days, future_prices, color="#00c878", linewidth=2,
             marker="o", markersize=5, label="Forecast")
    ax3.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.tick_params(colors="white", labelsize=7)
    for spine in ax3.spines.values():
        spine.set_color("#333333")
    ax3.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=7)
    ax3.set_xlabel("Days (0 = Today)", color="white", fontsize=8)
    ax3.set_ylabel("Price (USD)", color="white", fontsize=8)
    ax3.grid(alpha=0.15, color="white")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Test Set Performance ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Model Performance on Test Set")

X_test  = X_all[split:]
y_test  = y_all[split:]
pred_sc = model.predict(X_test, verbose=0)
pred    = scaler.inverse_transform(pred_sc)
actual  = scaler.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_squared_error, mean_absolute_error
mse  = mean_squared_error(actual, pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(actual, pred)

m1, m2, m3 = st.columns(3)
m1.metric("MSE",  f"{mse:.2f}",  help="Mean Squared Error (lower = better)")
m2.metric("RMSE", f"${rmse:.2f}", help="Root MSE in USD — average prediction error")
m3.metric("MAE",  f"${mae:.2f}",  help="Mean Absolute Error in USD")

fig4, ax4 = plt.subplots(figsize=(14, 4), facecolor="#0e1117")
ax4.set_facecolor("#0e1117")
ax4.plot(actual.flatten(), color="#00c8ff", linewidth=0.9, label="Actual",    alpha=0.9)
ax4.plot(pred.flatten(),   color="#ff9f40", linewidth=0.9, label="Predicted", alpha=0.9, linestyle="--")
ax4.set_title("LSTM Predictions vs Actual (Test Set)", color="white", fontsize=12)
ax4.tick_params(colors="white", labelsize=8)
ax4.set_xlabel("Test Day Index", color="white")
ax4.set_ylabel("Adj Close (USD)", color="white")
ax4.legend(facecolor="#1e1e2e", labelcolor="white", fontsize=9)
for spine in ax4.spines.values():
    spine.set_color("#333333")
ax4.grid(alpha=0.15, color="white")
plt.tight_layout()
st.pyplot(fig4)
plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This app is for educational purposes only. Predictions are based on historical patterns and do not constitute financial advice. Stock markets are subject to unpredictable events.")
