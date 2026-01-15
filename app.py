"""
Multi-Stock Price Prediction Web App
Flask backend that uses a pre-trained LightGBM model
Supports multiple stocks with real-time Yahoo Finance data
Includes caching to prevent rate limiting
"""

import pickle
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import json
import time
import threading

app = Flask(__name__)
CORS(app)

# Global variables for model and scaler
model = None
scaler = None

# Cache for stock data to prevent rate limiting
stock_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 300  # 5 minutes cache

# Rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = 0.5  # 500ms between requests

# Popular stocks to display
POPULAR_STOCKS = [
    {'symbol': 'AAPL', 'name': 'Apple Inc.'},
    {'symbol': 'GOOGL', 'name': 'Alphabet Inc.'},
    {'symbol': 'MSFT', 'name': 'Microsoft Corp.'},
    {'symbol': 'AMZN', 'name': 'Amazon.com Inc.'},
    {'symbol': 'TSLA', 'name': 'Tesla Inc.'},
    {'symbol': 'META', 'name': 'Meta Platforms Inc.'},
    {'symbol': 'NVDA', 'name': 'NVIDIA Corp.'},
    {'symbol': 'JPM', 'name': 'JPMorgan Chase'},
    {'symbol': 'V', 'name': 'Visa Inc.'},
    {'symbol': 'WMT', 'name': 'Walmart Inc.'},
]

def load_model():
    """Load the LightGBM model and scaler on startup"""
    global model, scaler
    
    with open('lgbm_aapl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('target_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully!")

def rate_limit():
    """Enforce rate limiting between requests"""
    global last_request_time
    current_time = time.time()
    elapsed = current_time - last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    last_request_time = time.time()

def get_cached_data(cache_key):
    """Get data from cache if valid"""
    with cache_lock:
        if cache_key in stock_cache:
            data, timestamp = stock_cache[cache_key]
            if time.time() - timestamp < CACHE_DURATION:
                return data
    return None

def set_cached_data(cache_key, data):
    """Store data in cache"""
    with cache_lock:
        stock_cache[cache_key] = (data, time.time())

def fetch_yahoo_data(symbol, period='6mo'):
    """Fetch stock data directly from Yahoo Finance API with caching and retry logic"""
    cache_key = f"data_{symbol}_{period}"
    
    # Check cache first
    cached = get_cached_data(cache_key)
    if cached:
        print(f"📦 Using cached data for {symbol}")
        return cached
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            rate_limit()  # Enforce rate limiting
            
            # Calculate date range
            end_date = datetime.now()
            if period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            else:
                start_date = end_date - timedelta(days=180)
            
            # Convert to Unix timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            
            # Yahoo Finance API URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'period1': start_ts,
                'period2': end_ts,
                'interval': '1d',
                'events': 'history'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = 5 * (attempt + 1)
                print(f"⚠️ Rate limited for {symbol}, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
        
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                raise ValueError(f"No data returned for {symbol}")
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Open': quotes['open'],
                'High': quotes['high'],
                'Low': quotes['low'],
                'Close': quotes['close'],
                'Volume': quotes['volume']
            }, index=pd.to_datetime(timestamps, unit='s'))
            
            # Remove any rows with NaN
            df = df.dropna()
            
            if df.empty:
                raise ValueError(f"Empty data for {symbol}")
            
            # Get additional stock info
            meta = result.get('meta', {})
            info = {
                'longName': meta.get('longName', symbol),
                'regularMarketPrice': meta.get('regularMarketPrice', df['Close'].iloc[-1]),
                'previousClose': meta.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]),
                'currency': meta.get('currency', 'USD')
            }
            
            result_data = (df, info, None)
            set_cached_data(cache_key, result_data)
            return result_data
            
        except requests.exceptions.Timeout as e:
            print(f"Timeout fetching {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None, None, f"Timeout after {max_retries} attempts"
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            if "429" in str(e):
                wait_time = 10 * (attempt + 1)
                print(f"⚠️ Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)
                if attempt < max_retries - 1:
                    continue
            return None, None, str(e)
    
    return None, None, "Max retries exceeded"

def get_stock_quote(symbol):
    """Get current stock quote with caching and retry logic"""
    cache_key = f"quote_{symbol}"
    
    # Check cache first
    cached = get_cached_data(cache_key)
    if cached:
        return cached
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            rate_limit()  # Enforce rate limiting
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {'interval': '1d', 'range': '5d'}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=20)
            
            # Handle rate limiting
            if response.status_code == 429:
                wait_time = 3 * (attempt + 1)
                print(f"⚠️ Rate limited for quote {symbol}, waiting {wait_time}s")
                time.sleep(wait_time)
                continue
                
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result.get('meta', {})
                quotes = result['indicators']['quote'][0]
                
                current_price = meta.get('regularMarketPrice', quotes['close'][-1])
                prev_close = meta.get('previousClose', quotes['close'][-2] if len(quotes['close']) > 1 else current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close else 0
                
                quote_data = {
                    'symbol': symbol,
                    'name': meta.get('longName', symbol),
                    'price': round(current_price, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2),
                    'currency': meta.get('currency', 'USD')
                }
                set_cached_data(cache_key, quote_data)
                return quote_data
                
        except requests.exceptions.Timeout as e:
            print(f"Timeout getting quote for {symbol} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            if "429" in str(e):
                time.sleep(5 * (attempt + 1))
                if attempt < max_retries - 1:
                    continue
            break
    
    return None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_features(df):
    """Calculate all 16 features required by the model"""
    data = df.copy()
    
    # Price lags
    data['close_lag_1'] = data['Close'].shift(1)
    data['close_lag_2'] = data['Close'].shift(2)
    data['close_lag_3'] = data['Close'].shift(3)
    data['close_lag_5'] = data['Close'].shift(5)
    
    # Volume lag
    data['volume_lag_1'] = data['Volume'].shift(1)
    
    # Moving averages
    data['ma7'] = data['Close'].rolling(window=7).mean()
    data['ma21'] = data['Close'].rolling(window=21).mean()
    data['ma50'] = data['Close'].rolling(window=50).mean()
    
    # Price to MA ratios
    data['price_ma7'] = data['Close'] / data['ma7']
    data['price_ma21'] = data['Close'] / data['ma21']
    
    # Returns
    data['return_1'] = data['Close'].pct_change(1)
    data['return_5'] = data['Close'].pct_change(5)
    
    # Volatility
    data['volatility_20'] = data['Close'].pct_change().rolling(window=20).std()
    
    # Technical indicators
    data['rsi_14'] = calculate_rsi(data['Close'], 14)
    data['macd'] = calculate_macd(data['Close'])
    data['atr_14'] = calculate_atr(data['High'], data['Low'], data['Close'], 14)
    
    return data

def get_prediction(symbol='AAPL'):
    """Get prediction for tomorrow's price"""
    # Fetch stock data
    df, info, error = fetch_yahoo_data(symbol, '6mo')
    
    if error:
        raise ValueError(f"Could not fetch data for {symbol}: {error}")
    
    # Calculate all features
    data = calculate_features(df)
    
    feature_columns = [
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
        'volume_lag_1', 'ma7', 'ma21', 'ma50', 'price_ma7', 'price_ma21',
        'return_1', 'return_5', 'volatility_20', 'rsi_14', 'macd', 'atr_14'
    ]
    
    # Get the latest complete row
    latest = data[feature_columns].dropna().iloc[-1:]
    
    if latest.empty:
        raise ValueError("Not enough data to calculate features")
    
    # Make prediction
    prediction_scaled = model.predict(latest.values)
    predicted_return = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    
    # Get current price
    current_price = float(df['Close'].iloc[-1])
    
    # Apply prediction
    if abs(predicted_return) < 1:
        predicted_price = current_price * (1 + predicted_return)
    else:
        predicted_price = current_price + predicted_return
    
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    return {
        'symbol': symbol,
        'company_name': info.get('longName', symbol),
        'current_price': round(current_price, 2),
        'predicted_price': round(float(predicted_price), 2),
        'price_change': round(float(price_change), 2),
        'price_change_pct': round(float(price_change_pct), 2),
        'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'volume': int(df['Volume'].iloc[-1]),
        'high_52w': round(float(df['High'].max()), 2),
        'low_52w': round(float(df['Low'].min()), 2),
        'indicators': {
            'rsi_14': round(float(data['rsi_14'].iloc[-1]), 2),
            'macd': round(float(data['macd'].iloc[-1]), 4),
            'ma7': round(float(data['ma7'].iloc[-1]), 2),
            'ma21': round(float(data['ma21'].iloc[-1]), 2),
            'ma50': round(float(data['ma50'].iloc[-1]), 2),
            'volatility': round(float(data['volatility_20'].iloc[-1]) * 100, 2),
            'atr': round(float(data['atr_14'].iloc[-1]), 2)
        }
    }

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/stocks')
def get_stocks():
    """Get all popular stock quotes"""
    stocks = []
    for stock in POPULAR_STOCKS:
        quote = get_stock_quote(stock['symbol'])
        if quote:
            stocks.append(quote)
        else:
            stocks.append({
                'symbol': stock['symbol'],
                'name': stock['name'],
                'price': 0,
                'change': 0,
                'change_pct': 0,
                'error': True
            })
    return jsonify({'success': True, 'stocks': stocks})

@app.route('/api/predict')
@app.route('/api/predict/<symbol>')
def predict(symbol='AAPL'):
    """API endpoint for stock prediction"""
    try:
        result = get_prediction(symbol.upper())
        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'cache_size': len(stock_cache)
    })

@app.route('/api/clear-cache')
def clear_cache():
    """Clear the stock data cache"""
    global stock_cache
    with cache_lock:
        stock_cache = {}
    return jsonify({'success': True, 'message': 'Cache cleared'})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
