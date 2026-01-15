# 📈 Stock Market Prediction App

An AI-powered stock price prediction web application using **LightGBM** machine learning and real-time **Yahoo Finance** data.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **Real-time Stock Data** - Live prices from Yahoo Finance API
- **AI Price Predictions** - Next-day price predictions using LightGBM
- **Multi-Stock Support** - Track 10+ popular stocks (AAPL, GOOGL, MSFT, TSLA, etc.)
- **Technical Indicators** - RSI, MACD, ATR, Moving Averages, Volatility
- **Smart Caching** - Built-in rate limiting and caching to prevent API blocks
- **Modern UI** - Clean, responsive web interface

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/muhammadtihame/StockMarketPredictionApp.git
cd StockMarketPredictionApp

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

## 🧠 Model Architecture

### Training Details

The model was trained using **LightGBM Regressor** on historical AAPL stock data.

| Parameter | Value |
|-----------|-------|
| Algorithm | LightGBM Regressor |
| Estimators | 3,000 |
| Learning Rate | 0.01 |
| Max Depth | 6 |
| Num Leaves | 48 |
| Subsample | 0.75 |
| Column Sample | 0.75 |
| Early Stopping | 100 rounds |

### Feature Engineering (16 Features)

| Category | Features |
|----------|----------|
| **Price Lags** | `close_lag_1`, `close_lag_2`, `close_lag_3`, `close_lag_5` |
| **Volume** | `volume_lag_1` |
| **Moving Averages** | `ma7`, `ma21`, `ma50` |
| **Price/MA Ratios** | `price_ma7`, `price_ma21` |
| **Returns** | `return_1`, `return_5` |
| **Volatility** | `volatility_20` |
| **Technical Indicators** | `rsi_14`, `macd`, `atr_14` |

### Model Performance

- **Directional Accuracy**: ~53-55% (predicting up/down movement)
- **Target**: Next-day percentage return prediction

## 📁 Project Structure

```
StockMarketPredictionApp/
├── app.py                  # Flask backend
├── lgbm_aapl_model.pkl     # Trained LightGBM model
├── target_scaler.pkl       # RobustScaler for target
├── requirements.txt        # Python dependencies
├── static/
│   └── style.css           # CSS styling
└── templates/
    └── index.html          # Frontend template
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/stocks` | GET | Get all popular stock quotes |
| `/api/predict/<symbol>` | GET | Get prediction for a stock |
| `/api/health` | GET | Health check endpoint |
| `/api/clear-cache` | GET | Clear cached data |

### Example API Response

```json
{
  "success": true,
  "data": {
    "symbol": "AAPL",
    "company_name": "Apple Inc.",
    "current_price": 185.50,
    "predicted_price": 186.25,
    "price_change": 0.75,
    "price_change_pct": 0.40,
    "prediction_date": "2026-01-16",
    "indicators": {
      "rsi_14": 55.32,
      "macd": 0.0012,
      "ma7": 184.20,
      "ma21": 182.50,
      "ma50": 180.10,
      "volatility": 1.25,
      "atr": 2.85
    }
  }
}
```

## 🛠️ Tech Stack

- **Backend**: Flask 3.0
- **ML Framework**: LightGBM 4.2
- **Data Processing**: Pandas, NumPy
- **Data Source**: Yahoo Finance API
- **Scaling**: Scikit-learn RobustScaler

## ⚠️ Disclaimer

This application is for **educational purposes only**. Stock market predictions are inherently uncertain, and this tool should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment choices.

## 📄 License

MIT License - feel free to use and modify for your own projects.

## 👨‍💻 Author

**Muhammad Tihame**

---

⭐ Star this repo if you found it useful!
