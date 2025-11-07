
# 📈 Multi-Stock Market Prediction App (LSTM)

A deep learning web app that predicts and visualizes stock market trends for Indian and global stocks using **LSTM neural networks**.  
Built with **Python**, **Streamlit**, **TensorFlow**, and **Plotly**.

---

## 🚀 Features

- Real-time stock data download using **yfinance**
- Predicts stock closing prices using **LSTM (Long Short-Term Memory)** models
- Interactive **candlestick** and **forecast** charts
- Multi-stock comparison and forecast
- INR conversion for international tickers (via USD-INR rate)
- Excel export of predictions and metrics
- Adjustable **epochs**, **batch size**, and **lookback window**

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|--------|
| **Frontend** | Streamlit |
| **Machine Learning** | TensorFlow, scikit-learn |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Plotly |
| **APIs** | Yahoo Finance (via yfinance) |
| **File Export** | openpyxl |

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Stock_market.git
cd Stock_market
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate it

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run stockmarket.py
```

Then open the local URL provided by Streamlit (usually `http://localhost:8501`).

---

## 🧩 Parameters & Controls

| Parameter        | Description                                       |
| ---------------- | ------------------------------------------------- |
| **Epochs**       | Number of training iterations                     |
| **Batch Size**   | Data samples processed per training step          |
| **Time Step**    | Number of previous days considered for prediction |
| **Predict Days** | Future days to forecast                           |

---

## 📊 Output Examples

* **Candlestick chart** for each ticker
* **Actual vs Predicted** stock prices
* **Future forecast** for the next N days
* **Downloadable Excel report** with all metrics and predictions

---

## 🧾 .gitignore Recommendation

```
venv/
__pycache__/
*.h5
*.hdf5
*.pkl
*.ckpt
*.npy
*.npz
*.csv
*.xlsx
.DS_Store
```

---

## 🤝 Contributing

Contributions, ideas, and bug reports are welcome.
Feel free to fork this repo, open issues, or submit pull requests.

---

Developed by:** Priyanka Gond.
