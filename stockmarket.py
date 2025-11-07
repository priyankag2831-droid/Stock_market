import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from io import BytesIO



def adjust_ticker(ticker):
    ticker = ticker.upper().strip()
    indian_bse = ["TATAPOWER", "TCS", "TATASTEEL", "TATAMOTORS"]
    global_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "V", "KO", "NFLX"]

    if ticker in indian_bse:
        return ticker + ".BO"
    elif ticker not in global_tickers and '.' not in ticker:
        return ticker + ".NS"
    else:
        return ticker


def get_usd_inr_rate():
    try:
        fx_data = yf.download("USDINR=X", period="1d", interval="1d", progress=False)
        if not fx_data.empty:
            return float(fx_data['Close'].iloc[-1])
        else:
            return 83.0
    except Exception:
        return 83.0


def convert_to_inr(price, ticker, usd_inr_rate):
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return float(price)
    else:
        return float(price) * usd_inr_rate


@st.cache_data
def download_stock_data(ticker, start, end):
    ticker = adjust_ticker(ticker)
    stock = yf.Ticker(ticker)
    data = stock.history(start=start, end=end)
    if data.empty:
        return None
    return data.dropna()


def prepare_prediction_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i])
        y.append(scaled_data[i][0])
    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def estimate_training_time(data_length, batch_size, epochs):
    per_epoch_time = (data_length / batch_size) * 0.005
    total_time_sec = per_epoch_time * epochs
    minutes, seconds = divmod(total_time_sec, 60)
    return f"Approx. Training Time: {int(minutes)} min {int(seconds)} sec"



st.set_page_config(page_title="Stock Market Analysis", layout="wide")
st.title("📈 Multi-Stock Market Prediction using LSTM")

tickers_input = st.text_input("Enter Stock Tickers (comma separated):", value="RELIANCE.NS")
start_date = st.date_input("Select Start Date:", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("Select End Date:", value=pd.to_datetime("2024-01-01"))
predict_days = st.slider("How many Future Days to Predict?", min_value=1, max_value=30, value=7)


st.subheader("⚙️ LSTM Parameters")
epochs = st.slider("Epochs", 1, 100, 30)
batch_size = st.slider("Batch Size", 8, 128, 32, step=8)
time_step = st.slider("Time Step (Days Lookback)", 10, 90, 60)

company_groups = {
    "RELIANCE.NS": "Large-cap",
    "TCS.NS": "Large-cap",
    "INFY.NS": "Large-cap",
    "WIPRO.NS": "Mid-cap",
    "TATAMOTORS.NS": "Mid-cap",
    "TSLA": "Global",
    "AAPL": "Global",
    "MSFT": "Global",
    "GOOG": "Global",
    "AMZN": "Global"
}


if st.button("Analyze Stocks") and tickers_input:
    tickers = [t.strip() for t in tickers_input.split(",")]
    all_results = []

    usd_inr_rate = get_usd_inr_rate()
    st.info(f"💱 Current USD → INR rate: {usd_inr_rate:.2f}")

    for orig_ticker in tickers:
        adj_ticker = adjust_ticker(orig_ticker)
        stock_data = download_stock_data(adj_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if stock_data is None or len(stock_data) < time_step + 1:
            st.warning(f"Not enough data for {orig_ticker} (Adjusted: {adj_ticker})")
            continue

       
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        fig.update_layout(title=f"Candlestick Chart - {adj_ticker}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

       
        X, y, scaler = prepare_prediction_data(stock_data, time_step)
        if len(X) == 0:
            continue

        est_time = estimate_training_time(len(X), batch_size, epochs)
        st.info(f"⏱ Estimated Training Time for {adj_ticker}: {est_time}")

     
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

       
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,  
            callbacks=[early_stop]
        )

        
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        
        y_pred_inr = [convert_to_inr(val, adj_ticker, usd_inr_rate) for val in y_pred.flatten()]
        y_actual_inr = [convert_to_inr(val, adj_ticker, usd_inr_rate) for val in y_actual.flatten()]

        
        mae = mean_absolute_error(y_actual_inr, y_pred_inr)
        rmse = np.sqrt(mean_squared_error(y_actual_inr, y_pred_inr))
        st.metric(label=f"{adj_ticker} - MAE (₹)", value=f"{mae:.2f}")
        st.metric(label=f"{adj_ticker} - RMSE (₹)", value=f"{rmse:.2f}")

        
        compare_df = pd.DataFrame({
        "Date": stock_data.index[-len(y_actual_inr):],
        "Actual": y_actual_inr,
        "Predicted": y_pred_inr
        })
        compare_df["Error"] = abs(compare_df["Actual"] - compare_df["Predicted"])
        compare_df["Ticker"] = adj_ticker
        compare_df["Group"] = company_groups.get(adj_ticker, "Other")
        compare_df["Currency"] = "INR"

      
        display_df = compare_df.tail(15).reset_index(drop=True)
        display_df.index = display_df.index + 1  # Start from 1

        st.subheader(f"📊 Actual vs Predicted - {adj_ticker}")
        st.dataframe(display_df)

       
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=compare_df["Date"],
            y=compare_df["Actual"],
            mode='lines',
            name="Actual",
            line=dict(color='#ffff00', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=compare_df["Date"],
            y=compare_df["Predicted"],
            mode='lines',
            name="Predicted",
            line=dict(color='#1f77b4', width=2)
        ))
        fig2.update_layout(
            title=f"Actual vs Predicted ({adj_ticker})",
            xaxis_title="Date",
            yaxis_title="Price (INR)",
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig2, use_container_width=True)

     
        future_input = X[-1].copy()
        future_scaled = []
        for _ in range(predict_days):
            pred_scaled = model.predict(future_input.reshape(1, time_step, 1), verbose=0)[0][0]
            future_scaled.append(pred_scaled)
            future_input = np.roll(future_input, -1)
            future_input[-1] = pred_scaled

        future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.bdate_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=predict_days)
        future_inr = [convert_to_inr(val, adj_ticker, usd_inr_rate) for val in future_prices]

        future_df = pd.DataFrame({
            "Ticker": adj_ticker,
            "Date": future_dates,
            "Predicted": future_inr,
            "Currency": "INR"
        })

        st.subheader(f"🔮 {predict_days}-Day Future Forecast - {adj_ticker}")
        st.dataframe(future_df)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=stock_data.index,
            y=[convert_to_inr(v, adj_ticker, usd_inr_rate) for v in stock_data["Close"]],
            mode='lines',
            name="Historical"
        ))
        fig3.add_trace(go.Scatter(
            x=future_df["Date"],
            y=future_df["Predicted"],
            mode='lines+markers',
            name="Forecast"
        ))
        fig3.update_layout(
            title=f"Future Price Forecast ({adj_ticker})",
            xaxis_title="Date",
            yaxis_title="Price (INR)"
        )
        st.plotly_chart(fig3, use_container_width=True)

        all_results.append(compare_df)
        all_results.append(future_df)

   
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        
        if 'Date' in final_df.columns:
            final_df['Date'] = pd.to_datetime(final_df['Date']).dt.tz_localize(None)

        excel_buffer = BytesIO()
        final_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_buffer,
            file_name="Stock_Prediction_Results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

st.markdown("---")
st.caption("LSTM-based Multi-Stock Prediction App")