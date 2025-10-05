# Standard library imports
import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import datetime
import time
from datetime import UTC
# Third-party imports
from jwt import encode, decode
import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import aiohttp
import asyncio
import nest_asyncio
import jwt
import bcrypt
import extra_streamlit_components as stx

nest_asyncio.apply()

# Initialize session state for authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# Simulated user database - In production, use a real database
USERS = {
    "user1": bcrypt.hashpw("user123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
}

def verify_password(username: str, password: str) -> bool:
    """Verify password against stored hash"""
    if username not in USERS:
        return False
    stored_hash = USERS[username].encode('utf-8')
    return bcrypt.checkpw(password.encode('utf-8'), stored_hash)

def create_jwt_token(username: str) -> str:
    """Create a JWT token for the user"""
    expiration = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=24)
    return encode(
        {"user": username, "exp": expiration},
        "your-secret-key",  # In production, use a secure secret key
        algorithm="HS256"
    )

def verify_jwt_token(token: str) -> bool:
    """Verify JWT token"""
    try:
        decode(token, "your-secret-key", algorithms=["HS256"])
        return True
    except:
        return False

def get_manager():
    """Get cookie manager with unique key"""
    if not hasattr(st.session_state, 'cookie_manager'):
        st.session_state.cookie_manager = stx.CookieManager(key="crypto_assistant_auth")
    return st.session_state.cookie_manager

def check_authentication():
    """Check if user is authenticated"""
    cookie_manager = get_manager()
    if st.session_state.authenticated:
        return True
        
    # Then check token
    token = cookie_manager.get("auth_token")
    if token and verify_jwt_token(token):
        st.session_state.authenticated = True
        return True
        
    return False

def login_page():
    """Display login page"""
    # Add custom CSS for center alignment
    st.markdown("""
        <style>
        .login-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1rem;
            margin-top: 1rem;
        }
        .logo-image {
            max-width: 100px;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Create centered container with smaller logo
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image("logo-v3.png", width=100, use_container_width=True)  # Changed to use_container_width
        st.markdown("<h2 style='text-align: center'>Crypto Assistant</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Advanced Crypto Analysis Platform</p>", unsafe_allow_html=True)

    # Check if already authenticated
    if st.session_state.authenticated:
        return
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if verify_password(username, password):
                token = create_jwt_token(username)
                cookie_manager = get_manager()
                cookie_manager.set("auth_token", token, expires_at=datetime.datetime.now() + datetime.timedelta(days=1))
                st.session_state.authenticated = True
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions
class CryptoAssistantError(Exception):
    """Base exception class for Crypto Assistant."""
    pass

class DataFetchError(CryptoAssistantError):
    """Raised when data fetching fails."""
    pass

class AnalysisError(CryptoAssistantError):
    """Raised when analysis operations fail."""
    pass

class ValidationError(CryptoAssistantError):
    """Raised when data validation fails."""
    pass

class ConfigurationError(CryptoAssistantError):
    """Raised when configuration is invalid."""
    pass

# Configuration dataclass
@dataclass
class Config:
    """Application configuration settings."""
    cache_duration: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay: int = 2
    api_rate_limit: int = 30
    default_timeframe: str = '1d'
    max_lookback_days: int = 365
    min_data_points: int = 30
    sentiment_update_interval: int = 3600  # 1 hour
    technical_indicators: List[str] = None
    agent_weights: Dict[str, float] = None
    
    def __post_init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize technical indicators
        if self.technical_indicators is None:
            self.technical_indicators = [
                'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR'
            ]
        
        # Initialize agent weights
        if self.agent_weights is None:
            self.agent_weights = {
                'TechnicianAgent': 0.35,
                'FundamentalAgent': 0.25,
                'RiskAgent': 0.25,
                'SentimentAgent': 0.15
            }
        
        # Load API keys from environment
        self.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET')
        self.kucoin_api_key = os.getenv('KUCOIN_API_KEY')
        self.kucoin_secret = os.getenv('KUCOIN_SECRET')
        self.huobi_api_key = os.getenv('HUOBI_API_KEY')
        self.huobi_secret = os.getenv('HUOBI_SECRET')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

# Constants
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
EXCHANGES = ['binance', 'kucoin', 'huobi']
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
VOLUME_SIGNIFICANCE_THRESHOLD = 2.0  # Standard deviations
TREND_CHANGE_THRESHOLD = 0.02  # 2%

class OllamaModelManager:
    """Manages interaction with local Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        # Initialize event loop
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        
    async def _make_request(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        """Make an HTTP request to Ollama API"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/{endpoint}"
            try:
                if method == "GET":
                    async with session.get(url) as response:
                        return await response.json()
                elif method == "POST":
                    async with session.post(url, json=data) as response:
                        return await response.json()
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Failed to connect to Ollama: {str(e)}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid response from Ollama: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            # Run async function in the event loop
            response = self.loop.run_until_complete(self._make_request("api/tags"))
            return [model['name'] for model in response.get('models', [])]
        except Exception as e:
            raise ConnectionError(f"Failed to fetch models: {str(e)}")
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if a model is available and working"""
        try:
            # Run async function in the event loop
            response = self.loop.run_until_complete(self._async_validate_model(model_name))
            return response
        except Exception:
            return False
            
    async def _async_validate_model(self, model_name: str) -> bool:
        """Async implementation of model validation"""
        try:
            response = await self._make_request(
                "api/generate",
                method="POST",
                data={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False
                }
            )
            return 'response' in response
        except Exception:
            return False

class DataFetcher:
    """Handles data fetching from various sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.exchanges = {}
        self._init_exchanges()
    
    def _init_exchanges(self):
        """Initialize exchange connections with fallback to public API"""
        try:
            # Initialize Binance
            binance_config = {
                'apiKey': self.config.binance_api_key,
                'secret': self.config.binance_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            self.exchanges['binance'] = ccxt.binance(binance_config)
            
            # Initialize KuCoin
            kucoin_config = {
                'apiKey': self.config.kucoin_api_key,
                'secret': self.config.kucoin_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            self.exchanges['kucoin'] = ccxt.kucoin(kucoin_config)
            
            # Initialize Huobi
            huobi_config = {
                'apiKey': self.config.huobi_api_key,
                'secret': self.config.huobi_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            }
            self.exchanges['huobi'] = ccxt.huobi(huobi_config)
            
        except Exception as e:
            logger.warning(f"Failed to initialize exchanges with API keys: {str(e)}")
            logger.info("Falling back to public API access")
            
            # Initialize exchanges without API keys
            self.exchanges = {
                'binance': ccxt.binance({'enableRateLimit': True}),
                'kucoin': ccxt.kucoin({'enableRateLimit': True}),
                'huobi': ccxt.huobi({'enableRateLimit': True})
            }
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize trading symbol to proper format"""
        if not symbol:
            raise ValidationError("Symbol cannot be empty")
            
        # Convert to uppercase
        symbol = symbol.upper()
        
        # Add slash if missing
        if '/' not in symbol:
            if 'USDT' in symbol:
                base = symbol.replace('USDT', '')
                symbol = f"{base}/USDT"
            else:
                raise ValidationError("Invalid symbol format. Must be like 'BTC/USDT'")
        
        return symbol
    
    def fetch_price_data(self, symbol: str, timeframe: str = '1d', limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data for a given symbol"""
        try:
            # Normalize symbol
            symbol = self._normalize_symbol(symbol)
            logger.info(f"Fetching data for {symbol}")
            
            errors = []
            for exchange_id, exchange in self.exchanges.items():
                try:
                    # Load markets first
                    exchange.load_markets()
                    
                    # Check if symbol exists on this exchange
                    if symbol not in exchange.symbols:
                        alt_symbol = symbol.replace('/', '')  # Try without slash
                        if alt_symbol in exchange.symbols:
                            symbol = alt_symbol
                        else:
                            error_msg = f"{symbol} not found on {exchange_id}"
                            logger.warning(error_msg)
                            errors.append(error_msg)
                            continue
                    
                    # Fetch OHLCV data
                    ohlcv = exchange.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=limit
                    )
                    
                    if not ohlcv:
                        error_msg = f"No data returned from {exchange_id} for {symbol}"
                        logger.warning(error_msg)
                        errors.append(error_msg)
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Process timestamp
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    logger.info(f"Successfully fetched data from {exchange_id} for {symbol}")
                    return df
                    
                except Exception as e:
                    error_msg = f"Error fetching from {exchange_id}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # If we get here, all exchanges failed
            error_details = "\n".join(errors)
            raise DataFetchError(
                f"Could not fetch data for {symbol} from any exchange.\nErrors:\n{error_details}"
            )
            
        except ValidationError as e:
            raise e
        except Exception as e:
            raise DataFetchError(f"Error fetching price data: {str(e)}")
    
    def fetch_market_depth(self, symbol: str, limit: int = 20) -> Dict:
        """Fetch order book data"""
        try:
            symbol = self._normalize_symbol(symbol)
            
            for exchange_id, exchange in self.exchanges.items():
                try:
                    orderbook = exchange.fetch_order_book(symbol, limit=limit)
                    return {
                        'bids': orderbook['bids'],
                        'asks': orderbook['asks'],
                        'timestamp': orderbook['timestamp'],
                        'exchange': exchange_id
                    }
                except Exception as e:
                    logger.warning(f"Error fetching order book from {exchange_id}: {str(e)}")
                    continue
            
            raise DataFetchError(f"Could not fetch order book for {symbol}")
            
        except ValidationError as e:
            raise e
        except Exception as e:
            raise DataFetchError(f"Error fetching market depth: {str(e)}")
    
    def fetch_ticker(self, symbol: str) -> Dict:
        """Fetch current ticker data"""
        try:
            symbol = self._normalize_symbol(symbol)
            
            for exchange_id, exchange in self.exchanges.items():
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    return {
                        'last': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'volume': ticker['baseVolume'],
                        'change_24h': ticker['percentage'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low'],
                        'exchange': exchange_id
                    }
                except Exception as e:
                    logger.warning(f"Error fetching ticker from {exchange_id}: {str(e)}")
                    continue
            
            raise DataFetchError(f"Could not fetch ticker for {symbol}")
            
        except ValidationError as e:
            raise e
        except Exception as e:
            raise DataFetchError(f"Error fetching ticker: {str(e)}")
    
    def close_connections(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            try:
                exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange connection: {str(e)}")

class DataValidator:
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Input must be a pandas DataFrame")
        if not required_columns.issubset(df.columns):
            raise ValidationError(f"Missing required columns: {required_columns - set(df.columns)}")
        if df[list(required_columns)].isnull().any().any():
            raise ValidationError("Data contains missing values")
        if not all(df[col].dtype.kind in 'fc' for col in required_columns):
            raise ValidationError("Numeric columns contain non-numeric data")
        if not ((df['high'] >= df['low']).all() and 
                (df['high'] >= df['open']).all() and 
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and
                (df['low'] <= df['close']).all()):
            raise ValidationError("Invalid price relationships in data")
        return True

    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        if not isinstance(symbol, str):
            raise ValidationError("Symbol must be a string")
        if '/' not in symbol:
            raise ValidationError("Symbol must contain '/' (e.g., 'BTC/USDT')")
        base, quote = symbol.split('/')
        if not (base and quote):
            raise ValidationError("Invalid symbol format")
        if not all(c.isalnum() for c in base + quote):
            raise ValidationError("Symbol contains invalid characters")
        return True

    @staticmethod
    def validate_timeframe(timeframe: str) -> bool:
        valid_timeframes = {'1m', '5m', '15m', '1h', '4h', '1d', '1w'}
        if timeframe not in valid_timeframes:
            raise ValidationError(f"Invalid timeframe. Must be one of {valid_timeframes}")
        return True

    @staticmethod
    def validate_analysis_results(analysis: Dict) -> bool:
        required_keys = {
            'trend', 'momentum', 'volatility', 
            'support_resistance', 'volume_analysis', 'indicators', 
            'sentiment', 'fear_greed_index', 'social_media_sentiment', 'market_data'
        }
        if not required_keys.issubset(analysis.keys()):
            raise ValidationError(f"Missing required analysis sections: {required_keys - set(analysis.keys())}")
        return True

class BaseAgent:
    """Base class for all analysis agents."""
    
    def __init__(self, weight: float):
        self.name = self.__class__.__name__
        self.weight = weight
        self.reasoning = []  # Initialize reasoning list
        
    def _normalize_score(self, score: float) -> float:
        """Normalize score to range [-1, 1]"""
        return max(min(score, 1.0), -1.0)
        
    def _calculate_confidence(self, score: float) -> float:
        """Convert normalized score to confidence percentage"""
        return abs(score) * 100

    def _add_reasoning(self, new_reason: str, importance: str = "normal"):
        """Add reasoning with proper formatting based on importance"""
        prefix = {
            "high": "❗ ",
            "normal": "• ",
            "low": "◦ "
        }.get(importance, "• ")
        self.reasoning.append(f"{prefix}{new_reason}")
    
    def clear_reasoning(self):
        """Clear reasoning list before new analysis"""
        self.reasoning = []

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period).mean()

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series) -> tuple:
    """Calculate MACD, Signal, and Histogram"""
    exp1 = data.ewm(span=12, adjust=False).mean()
    exp2 = data.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger_bands(data: pd.Series, period: int = 20, std: int = 2) -> tuple:
    """Calculate Bollinger Bands"""
    ma = data.rolling(window=period).mean()
    std_dev = data.rolling(window=period).std()
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, ma, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

class MarketAnalyzer:
    """Analyzes market data and generates indicators"""
    
    def __init__(self, config: Config):
        """Initialize MarketAnalyzer with configuration"""
        self.config = config
        self.indicators: Dict = {}
        self.current_symbol: str = ""
        
    def analyze_price(self, data: pd.DataFrame) -> Dict:
            """Analyze price data and generate indicators"""
            try:
                analysis = {
                    'indicators': {},
                    'trend': {},
                    'momentum': {},
                    'volatility': {},
                    'market_data': {},
                    'volume_analysis': {},
                    'support_resistance': {}
                }
                
                # Calculate basic indicators first
                analysis['indicators'] = self._calculate_indicators(data)
                
                # Calculate all analyses
                analysis['trend'] = self._analyze_trend(data, analysis['indicators'])
                analysis['momentum'] = self._analyze_momentum(data, analysis['indicators'])
                analysis['volatility'] = self._analyze_volatility(data, analysis['indicators'])
                analysis['market_data'] = self._get_market_data(data)
                analysis['volume_analysis'] = self._analyze_volume(data)
                analysis['support_resistance'] = self._calculate_support_resistance(data)
                analysis['sentiment'] = self._calculate_sentiment(data, analysis)
                
                # Calculate risk assessment
                analysis['risk_assessment'] = self._calculate_risk_assessment(data, analysis)
                
                # Add market health metrics
                analysis['market_health'] = self._calculate_market_health(data, analysis)
                
                return analysis
                
            except Exception as e:
                logger.error(f"Error in price analysis: {str(e)}")
                raise AnalysisError(f"Failed to analyze price data: {str(e)}")

    def _calculate_sentiment(self, data: pd.DataFrame, analysis: Dict) -> Dict:
            """Calculate market sentiment"""
            try:
                sentiment_score = 0.0
                desc = []
                
                # Technical contribution (40%)
                if analysis['trend']['trend'] in ['Strongly Bullish', 'Bullish']:
                    sentiment_score += 0.4
                    desc.append("Strong bullish trend detected.")
                elif analysis['trend']['trend'] == 'Bearish':
                    sentiment_score -= 0.4
                    desc.append("Bearish trend detected.")
                    
                # Momentum contribution (30%)
                rsi = analysis['momentum']['rsi_value']
                if rsi > 70:
                    sentiment_score -= 0.3
                    desc.append("Overbought conditions (RSI).")
                elif rsi < 30:
                    sentiment_score += 0.3
                    desc.append("Oversold conditions (RSI).")
                    
                # Volume contribution (30%)
                volume_trend = analysis['volume_analysis']['volume_trend']
                volume_intensity = analysis['volume_analysis']['volume_intensity']
                
                if volume_trend == 'Increasing':
                    sentiment_score += 0.3
                    desc.append("Increasing volume trend.")
                elif volume_trend == 'Decreasing':
                    sentiment_score -= 0.2
                    desc.append("Decreasing volume trend.")
                    
                if volume_intensity in ['Very High', 'High']:
                    sentiment_score *= 1.2
                    desc.append(f"High volume intensity ({volume_intensity}).")
                    
                # Normalize score
                sentiment_score = max(min(sentiment_score, 1.0), -1.0)
                
                # Determine sentiment label
                if sentiment_score > 0.3:
                    sentiment = 'Bullish'
                elif sentiment_score < -0.3:
                    sentiment = 'Bearish'
                else:
                    sentiment = 'Neutral'
                    
                # Add market health contribution
                if 'market_health' in analysis:
                    health = analysis['market_health']
                    health_score = (
                        health['liquidity_score'] +
                        health['efficiency'] +
                        health['stability']
                    ) / 30.0  # Normalize to [-1, 1]
                    
                    sentiment_score = (sentiment_score + health_score) / 2
                    
                    if health['liquidity_score'] > 7:
                        desc.append("Strong market liquidity.")
                    elif health['liquidity_score'] < 3:
                        desc.append("Poor market liquidity.")
                
                # Add external sentiment data
                external_data = fetch_external_sentiment_data()
                
                return {
                    'score': round(sentiment_score, 2),
                    'sentiment': sentiment,
                    'description': " ".join(desc),
                    'fear_greed_index': external_data['fear_greed_index'],
                    'social_media_sentiment': external_data['social_media_sentiment']
                }
                
            except Exception as e:
                logger.error(f"Error calculating sentiment: {e}", exc_info=True)
                return {
                    'score': 0,
                    'sentiment': 'Neutral',
                    'description': 'Error calculating sentiment',
                    'fear_greed_index': 50,
                    'social_media_sentiment': 0
                }

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
            """Calculate technical indicators"""
            try:
                indicators = {}
                
                close = data['close']
                high = data['high']
                low = data['low']
                
                # Moving Averages
                for period in [20, 50, 200]:
                    indicators[f'sma_{period}'] = calculate_sma(close, period)
                    indicators[f'ema_{period}'] = calculate_ema(close, period)
                
                # RSI
                indicators['rsi'] = calculate_rsi(close)
                
                # MACD
                macd, signal, hist = calculate_macd(close)
                indicators['macd'] = macd
                indicators['macd_signal'] = signal
                indicators['macd_hist'] = hist
                
                # Bollinger Bands
                upper, middle, lower = calculate_bollinger_bands(close)
                indicators['bbands_upper'] = upper
                indicators['bbands_middle'] = middle
                indicators['bbands_lower'] = lower
                
                # ATR
                indicators['atr'] = calculate_atr(high, low, close)

                # Add Stochastic Oscillator calculation
                high_14 = data['high'].rolling(window=14).max()
                low_14 = data['low'].rolling(window=14).min()
                indicators['slowk'] = ((data['close'] - low_14) / (high_14 - low_14)) * 100
                indicators['slowd'] = indicators['slowk'].rolling(window=3).mean()

                return indicators
                
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
                raise AnalysisError(f"Failed to calculate indicators: {str(e)}")

    def _analyze_trend(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze price trend using multiple indicators"""
        try:
            result = {}
            
            last_close = data['close'].iloc[-1]
            sma20 = indicators['sma_20'].iloc[-1]
            sma50 = indicators['sma_50'].iloc[-1]
            sma200 = indicators['sma_200'].iloc[-1]
            
            # Calculate percentage differences
            result['sma20_diff'] = ((last_close - sma20) / sma20) * 100
            result['sma50_diff'] = ((last_close - sma50) / sma50) * 100
            result['sma200_diff'] = ((last_close - sma200) / sma200) * 100
            
            # Calculate trend strength
            strength = 0
            if last_close > sma20:
                strength += 1
            if last_close > sma50:
                strength += 1
            if last_close > sma200:
                strength += 1
            
            result['strength'] = strength
            
            # Determine trend direction
            if strength == 3 and sma20 > sma50 > sma200:
                result['trend'] = "Strongly Bullish"
            elif strength >= 2:
                result['trend'] = "Bullish"
            elif strength == 1:
                result['trend'] = "Slightly Bullish"
            else:
                result['trend'] = "Bearish"
                
            # Add slope analysis
            result['slope_20'] = self._calculate_slope(indicators['sma_20'].tail(5))
            result['slope_50'] = self._calculate_slope(indicators['sma_50'].tail(5))
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}", exc_info=True)
            return {
                'trend': 'Unknown',
                'strength': 0,
                'sma20_diff': 0,
                'sma50_diff': 0,
                'sma200_diff': 0,
                'slope_20': 0,
                'slope_50': 0
            }
    
    def _analyze_momentum(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze momentum indicators"""
        try:
            result = {}
            
            # RSI Analysis
            rsi = indicators['rsi'].iloc[-1]
            result['rsi_value'] = rsi
            
            if rsi > 70:
                result['rsi_condition'] = 'Overbought'
            elif rsi < 30:
                result['rsi_condition'] = 'Oversold'
            else:
                result['rsi_condition'] = 'Neutral'
            
            # MACD Analysis
            macd = indicators['macd'].iloc[-1]
            signal = indicators['macd_signal'].iloc[-1]
            hist = indicators['macd_hist'].iloc[-1]
            
            result['macd_trend'] = 'Bullish' if macd > signal else 'Bearish'
            result['macd_strength'] = abs(macd - signal)
            result['macd_histogram'] = hist
            
            # Stochastic Analysis
            slowk = indicators['slowk'].iloc[-1]
            slowd = indicators['slowd'].iloc[-1]
            
            result['stoch_k'] = slowk
            result['stoch_d'] = slowd
            
            if slowk > 80 and slowd > 80:
                result['stoch_condition'] = 'Overbought'
            elif slowk < 20 and slowd < 20:
                result['stoch_condition'] = 'Oversold'
            else:
                result['stoch_condition'] = 'Neutral'
            
            # Overall Momentum
            momentum_score = 0
            
            # RSI contribution
            if rsi > 50:
                momentum_score += (rsi - 50) / 20
            else:
                momentum_score -= (50 - rsi) / 20
            
            # MACD contribution
            if macd > signal:
                momentum_score += 0.5
            else:
                momentum_score -= 0.5
            
            # Stochastic contribution
            if slowk > slowd:
                momentum_score += 0.3
            else:
                momentum_score -= 0.3
            
            result['momentum_score'] = momentum_score
            
            if momentum_score > 0.5:
                result['overall_momentum'] = 'Strong Bullish'
            elif momentum_score > 0:
                result['overall_momentum'] = 'Bullish'
            elif momentum_score > -0.5:
                result['overall_momentum'] = 'Bearish'
            else:
                result['overall_momentum'] = 'Strong Bearish'
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}", exc_info=True)
            return {
                'rsi_value': 50,
                'rsi_condition': 'Neutral',
                'macd_trend': 'Neutral',
                'macd_strength': 0,
                'overall_momentum': 'Neutral'
            }
    
    def _analyze_volatility(self, data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze volatility indicators"""
        try:
            result = {}
            
            # ATR Analysis
            atr = indicators['atr'].iloc[-1]
            atr_pct = atr / data['close'].iloc[-1] * 100
            
            result['atr'] = atr
            result['atr_percentage'] = atr_pct
            
            # Bollinger Bands Analysis
            bb_upper = indicators['bbands_upper'].iloc[-1]
            bb_lower = indicators['bbands_lower'].iloc[-1]
            bb_middle = indicators['bbands_middle'].iloc[-1]
            
            current_price = data['close'].iloc[-1]
            
            # Calculate BB width and position
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            
            result['bollinger_width'] = bb_width
            result['bollinger_position'] = bb_position
            
            # Historical Volatility
            returns = data['close'].pct_change()
            vol = returns.std() * np.sqrt(252)  # Annualized volatility
            
            result['annual_volatility'] = vol
            
            # Determine volatility level
            if vol > 0.5:  # 50% annual volatility
                result['volatility_level'] = 'High'
            elif vol > 0.3:
                result['volatility_level'] = 'Medium'
            else:
                result['volatility_level'] = 'Low'
            
            # Price channel analysis
            window = 20
            rolling_high = data['high'].rolling(window=window).max()
            rolling_low = data['low'].rolling(window=window).min()
            channel_width = (rolling_high - rolling_low) / rolling_low * 100
            
            result['channel_width'] = channel_width.iloc[-1]
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}", exc_info=True)
            return {
                'atr': 0,
                'volatility_level': 'Unknown',
                'annual_volatility': 0,
                'bollinger_width': 0
            }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            result = {}
            
            # Basic volume metrics
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
            
            result['volume_change'] = ((current_volume - avg_volume) / avg_volume) * 100
            
            # Volume trend
            recent_volume = data['volume'].tail(5)
            if recent_volume.is_monotonic_increasing:
                result['volume_trend'] = 'Increasing'
            elif recent_volume.is_monotonic_decreasing:
                result['volume_trend'] = 'Decreasing'
            else:
                result['volume_trend'] = 'Mixed'
            
            # Volume intensity
            volume_std = data['volume'].std()
            volume_zscore = (current_volume - avg_volume) / volume_std
            
            if volume_zscore > 2:
                result['volume_intensity'] = 'Very High'
            elif volume_zscore > 1:
                result['volume_intensity'] = 'High'
            elif volume_zscore < -2:
                result['volume_intensity'] = 'Very Low'
            elif volume_zscore < -1:
                result['volume_intensity'] = 'Low'
            else:
                result['volume_intensity'] = 'Normal'
            
            # Price-volume relationship
            price_changes = data['close'].pct_change()
            volume_changes = data['volume'].pct_change()
            
            correlation = price_changes.corr(volume_changes)
            result['price_volume_correlation'] = correlation
            
            # OBV trend
            obv = self.indicators.get('obv', pd.Series())
            if not obv.empty:
                obv_sma = obv.rolling(window=20).mean()
                if obv.iloc[-1] > obv_sma.iloc[-1]:
                    result['obv_trend'] = 'Bullish'
                else:
                    result['obv_trend'] = 'Bearish'
            else:
                result['obv_trend'] = 'Unknown'
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}", exc_info=True)
            return {
                'volume_change': 0,
                'volume_trend': 'Unknown',
                'volume_intensity': 'Normal',
                'price_volume_correlation': 0
            }

    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
            """Calculate support and resistance levels"""
            try:
                result = {
                    'support_levels': [],
                    'resistance_levels': [],
                    'pivot_point': 0
                }
                
                # Calculate recent high/low/close
                high = data['high'].iloc[-1]
                low = data['low'].iloc[-1]
                close = data['close'].iloc[-1]
                
                # Calculate Pivot Point
                pp = (high + low + close) / 3
                result['pivot_point'] = pp
                
                # Calculate Support Levels
                s1 = pp * 2 - high  # First support
                s2 = pp - (high - low)  # Second support
                s3 = low - 2 * (high - pp)  # Third support
                
                result['support_levels'] = [
                    round(level, 2) for level in [s1, s2, s3]
                    if level > 0  # Filter out negative values
                ]
                
                # Calculate Resistance Levels
                r1 = pp * 2 - low  # First resistance
                r2 = pp + (high - low)  # Second resistance
                r3 = high + 2 * (pp - low)  # Third resistance
                
                result['resistance_levels'] = [
                    round(level, 2) for level in [r1, r2, r3]
                ]
                
                # Calculate Fibonacci Retracement Levels
                period_high = data['high'].max()
                period_low = data['low'].min()
                range_size = period_high - period_low
                
                result['fibonacci_levels'] = {
                    '0.236': round(period_high - 0.236 * range_size, 2),
                    '0.382': round(period_high - 0.382 * range_size, 2),
                    '0.500': round(period_high - 0.500 * range_size, 2),
                    '0.618': round(period_high - 0.618 * range_size, 2),
                    '0.786': round(period_high - 0.786 * range_size, 2)
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error calculating support/resistance: {str(e)}", exc_info=True)
                return {
                    'support_levels': [],
                    'resistance_levels': [],
                    'pivot_point': 0,
                    'fibonacci_levels': {}
                }
        
    def _calculate_market_health(self, data: pd.DataFrame, analysis: Dict) -> Dict:
        """Calculate overall market health metrics"""
        try:
            result = {}
            
            # Liquidity Score (based on volume and spread)
            avg_volume = data['volume'].mean()
            current_volume = data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            # Normalize to 0-10 scale
            liquidity_score = min(10, volume_ratio * 5)
            result['liquidity_score'] = round(liquidity_score, 2)
            
            # Market Efficiency Ratio (MER)
            price_movement = abs(data['close'].diff()).sum()
            price_range = data['high'].max() - data['low'].min()
            efficiency_ratio = price_range / price_movement if price_movement else 0
            
            # Normalize to 0-10 scale
            efficiency_score = min(10, efficiency_ratio * 10)
            result['efficiency'] = round(efficiency_score, 2)
            
            # Stability Score
            volatility = analysis['volatility']['annual_volatility']
            stability_score = max(0, 10 - (volatility * 10))
            result['stability'] = round(stability_score, 2)
            
            # Market Quality
            result['market_quality'] = round(
                (liquidity_score + efficiency_score + stability_score) / 3,
                2
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating market health: {str(e)}", exc_info=True)
            return {
                'liquidity_score': 5,
                'efficiency': 5,
                'stability': 5,
                'market_quality': 5
            }
    
    def _get_market_data(self, data: pd.DataFrame) -> Dict:
        try:
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            
            # Calculate 24h volume properly
            volume_24h = data['volume'].tail(24).sum() * current_price  # Multiply by price to get USD volume
            
            result = {
                'last_price': round(current_price, 8),
                'change_24h': round(((current_price - prev_price) / prev_price) * 100, 2),
                'volume_24h': round(volume_24h, 2),  # This will now show proper USD volume
                'high_24h': round(data['high'].tail(24).max(), 8),
                'low_24h': round(data['low'].tail(24).min(), 8),
                'price_range_24h': round(
                    data['high'].tail(24).max() - data['low'].tail(24).min(),
                    8
                ),
                'bid_ask_spread': 0.001,  # Placeholder - replace with actual spread data
                'vwap_24h': self._calculate_vwap(data.tail(24))
            }
            
            # Add additional statistics
            result['avg_price_24h'] = round(data['close'].tail(24).mean(), 8)
            result['price_std_24h'] = round(data['close'].tail(24).std(), 8)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}", exc_info=True)
            return {
                'last_price': 0,
                'change_24h': 0,
                'volume_24h': 0,
                'high_24h': 0,
                'low_24h': 0,
                'bid_ask_spread': 0,
                'vwap_24h': 0
            }
    
    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            volume = data['volume']
            
            return round(
                (typical_price * volume).sum() / volume.sum(),
                8
            )
        except Exception:
            return 0.0
    
    def _calculate_slope(self, prices: pd.Series) -> float:
        """Calculate the slope of a price series"""
        try:
            if len(prices) < 2:
                return 0
                
            # Convert to numpy array and create x values
            y = prices.values
            x = np.arange(len(y))
            
            # Calculate slope using polyfit
            slope, _ = np.polyfit(x, y, 1)
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating slope: {str(e)}", exc_info=True)
            return 0.0
    
    def get_analysis_summary(self, analysis: Dict) -> str:
        """Generate a human-readable summary of the analysis"""
        try:
            lines = [
                "Market Analysis Summary:",
                f"Trend: {analysis['trend']['trend']} (Strength: {analysis['trend']['strength']}/3)",
                f"Momentum: {analysis['momentum']['overall_momentum']} (RSI: {analysis['momentum']['rsi_value']:.1f})",
                f"Volatility: {analysis['volatility']['volatility_level']} (ATR: {analysis['volatility']['atr']:.2f})",
                f"Volume: {analysis['volume_analysis']['volume_trend']} ({analysis['volume_analysis']['volume_intensity']})",
                "",
                "Key Levels:",
                f"Support: {', '.join(map(str, analysis['support_resistance']['support_levels']))}",
                f"Resistance: {', '.join(map(str, analysis['support_resistance']['resistance_levels']))}",
                f"Pivot Point: {analysis['support_resistance']['pivot_point']:.2f}",
                "",
                "Market Health:",
                f"Liquidity: {analysis['market_health']['liquidity_score']}/10",
                f"Efficiency: {analysis['market_health']['efficiency']}/10",
                f"Stability: {analysis['market_health']['stability']}/10"
            ]
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {str(e)}", exc_info=True)
            return "Error generating analysis summary"

    def _calculate_risk_assessment(self, data: pd.DataFrame, analysis: Dict) -> Dict:
        """Calculate comprehensive risk assessment"""
        try:
            risk = {}
            
            # Get volatility data
            volatility = analysis['volatility']
            vol_level = volatility['volatility_level']
            annual_vol = volatility['annual_volatility']
            
            # Calculate base risk score (0-1)
            risk_score = 0.0
            
            # Volatility contribution (40%)
            if vol_level == 'High':
                risk_score += 0.4
            elif vol_level == 'Medium':
                risk_score += 0.2
            
            # Volume risk contribution (30%)
            volume_analysis = analysis['volume_analysis']
            volume_intensity = volume_analysis['volume_intensity']
            
            if volume_intensity in ['Very High', 'Very Low']:
                risk_score += 0.3
            elif volume_intensity in ['High', 'Low']:
                risk_score += 0.15
            
            # Market structure risk (30%)
            market_data = analysis['market_data']
            spread = market_data.get('bid_ask_spread', 0)
            
            if spread > 0.005:  # High spread
                risk_score += 0.3
            elif spread > 0.002:  # Medium spread
                risk_score += 0.15
            
            # Determine risk level
            if risk_score > 0.6:
                risk_level = 'High'
            elif risk_score > 0.3:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Create risk factors breakdown
            risk_factors = {
                'Volatility Risk': float(annual_vol),
                'Volume Risk': float(abs(volume_analysis['volume_change']) / 100),
                'Liquidity Risk': float(spread * 100),
                'Market Risk': float(risk_score)
            }
            
            # Generate risk alerts
            risk_alerts = []
            
            if vol_level == 'High':
                risk_alerts.append({
                    'message': f"High volatility detected ({annual_vol*100:.1f}% annual)",
                    'severity': 'high'
                })
            
            if volume_intensity in ['Very High', 'Very Low']:
                risk_alerts.append({
                    'message': f"Abnormal volume intensity: {volume_intensity}",
                    'severity': 'high'
                })
            
            if spread > 0.005:
                risk_alerts.append({
                    'message': f"Wide bid-ask spread ({spread*100:.3f}%)",
                    'severity': 'high'
                })
            
            # Compile final risk assessment
            risk.update({
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'risk_alerts': risk_alerts,
                'highest_risk': max(risk_factors.items(), key=lambda x: x[1])[0]
            })
            
            return risk
            
        except Exception as e:
            logger.error(f"Error calculating risk assessment: {str(e)}", exc_info=True)
            return {
                'risk_level': 'Unknown',
                'risk_score': 0.5,
                'risk_factors': {},
                'risk_alerts': []
            }

class MarketSentimentAnalyzer:
    def analyze_sentiment(self, data: pd.DataFrame, indicators: Dict[str, np.ndarray]) -> Dict:
        close = data['close'].values
        if len(close) < 30:
            return {"score": 0.0, "sentiment": "Neutral", "description": "Insufficient price data for robust sentiment."}

        recent_returns = np.diff(np.log(close[-30:]))
        avg_return = np.mean(recent_returns)
        rsi = indicators['rsi'][-1]
        sma50 = indicators['sma_50'][-1]
        sma200 = indicators['sma_200'][-1]
        current_price = close[-1]

        score = 0.0
        desc = []

        if avg_return > 0:
            score += 0.3
            desc.append("Recent returns slightly positive.")
        else:
            score -= 0.3
            desc.append("Recent returns slightly negative.")

        # RSI influence more:
        if rsi > 60:
            score += 0.5
            desc.append("RSI leaning bullish (>60).")
        elif rsi < 40:
            score -= 0.5
            desc.append("RSI leaning bearish (<40).")
        else:
            desc.append("RSI neutral range (40-60).")

        if current_price > sma50:
            score += 0.3
            desc.append("Above SMA50 (bullish sign).")
        else:
            score -= 0.3
            desc.append("Below SMA50 (bearish sign).")

        if current_price > sma200:
            score += 0.3
            desc.append("Above SMA200 (long-term bullish).")
        else:
            score -= 0.3
            desc.append("Below SMA200 (long-term cautious).")

        if score > 0.5:
            sentiment = "Bullish"
        elif score < -0.5:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        return {
            "score": round(score, 2),
            "sentiment": sentiment,
            "description": " ".join(desc)
        }

class TechnicianAgent(BaseAgent):
    """Technical Analysis Expert - Pattern recognition and indicator analysis"""
    
    def __init__(self):
        super().__init__(weight=0.35)
    
    def analyze(self, analysis: Dict) -> Dict:
        logger.info(f"[{self.name}] Starting technical analysis...")
        score = 0.0
        reasoning = []

        # Get key technical data
        trend_data = analysis['trend']
        momentum_data = analysis['momentum']
        volatility_data = analysis['volatility']
        volume_data = analysis['volume_analysis']

        # Trend Analysis (40%)
        trend_score = self._analyze_trend(trend_data)
        score += trend_score * 0.4

        # Momentum Analysis (30%)
        momentum_score = self._analyze_momentum(momentum_data)
        score += momentum_score * 0.3

        # Volatility & Volume Impact (30%)
        risk_score = self._analyze_risk(volatility_data, volume_data)
        score += risk_score * 0.3

        # Normalize and determine action
        final_score = self._normalize_score(score)
        confidence = self._calculate_confidence(final_score)
        action = self._determine_action(final_score)

        return {
            "agent": self.name,
            "action": action,
            "confidence": confidence,
            "score": final_score,
            "weight": self.weight,
            "reasoning": reasoning
        }

    def _analyze_trend(self, trend_data: Dict) -> float:
        trend = trend_data['trend']
        trend_mapping = {
            "Strongly Bullish": 1.0,
            "Bullish": 0.6,
            "Slightly Bullish": 0.3,
            "Bearish": -0.6
        }
        score = trend_mapping.get(trend, 0) * trend_data['strength'] / 3
        self._add_reasoning(
            f"Trend: {trend} (Strength: {trend_data['strength']}/3)",
            "high" if abs(score) > 0.5 else "normal"
        )
        return score

    def _analyze_momentum(self, momentum_data: Dict) -> float:
        score = 0.0
        rsi = momentum_data['rsi_value']
        macd_trend = momentum_data['macd_trend']
        macd_strength = momentum_data['macd_strength']

        # RSI Analysis
        if rsi > 70:
            score -= 0.8
            self._add_reasoning(f"RSI Overbought ({rsi:.1f})", "high")
        elif rsi < 30:
            score += 0.8
            self._add_reasoning( f"RSI Oversold ({rsi:.1f})", "high")
        else:
            score += (rsi - 50) / 50
            self._add_reasoning( f"RSI Neutral ({rsi:.1f})", "low")

        # MACD Analysis
        if macd_trend == "Bullish":
            score += macd_strength * 0.5
            self._add_reasoning( f"MACD Bullish ({macd_strength:.2f})", "normal")
        else:
            score -= macd_strength * 0.5
            self._add_reasoning( f"MACD Bearish ({macd_strength:.2f})", "normal")

        return score

    def _analyze_risk(self, volatility_data: Dict, volume_data: Dict) -> float:
        score = 0.0
        vol_level = volatility_data['volatility_level']
        bb_width = volatility_data['bollinger_width']
        volume_change = volume_data['volume_change']

        # Volatility Impact
        if vol_level == "High":
            score -= 0.5
            self._add_reasoning( "High volatility - caution advised", "high")
        elif vol_level == "Low":
            score += 0.3
            self._add_reasoning( "Low volatility - stable conditions", "normal")

        # Volume Confirmation
        if volume_change > 50:
            score *= 1.2
            self._add_reasoning( "Strong volume confirmation", "high")
        elif volume_change < -50:
            score *= 0.8
            self._add_reasoning( "Weak volume support", "normal")

        return score

    def _determine_action(self, score: float) -> str:
        if score > 0.3:
            return "Buy"
        elif score < -0.3:
            return "Sell"
        return "Hold"

class FundamentalAgent(BaseAgent):
    """Fundamental Analysis Expert - Market structure and volume analysis"""
    
    def __init__(self):
        super().__init__(weight=0.25)
    
    def analyze(self, analysis: Dict) -> Dict:
        """Analyze fundamental factors"""
        logger.info(f"[{self.name}] Starting fundamental analysis...")
        score = 0.0
        self.clear_reasoning()
        
        try:
            # Get fundamental data
            volume_data = analysis['volume_analysis']
            market_data = analysis['market_data']
            health_data = analysis.get('market_health', {})

            # Volume Analysis (40%)
            volume_score = self._analyze_volume_metrics(volume_data)
            score += volume_score * 0.4

            # Market Structure Analysis (35%)
            structure_score = self._analyze_market_structure(market_data)
            score += structure_score * 0.35

            # Market Health Analysis (25%)
            health_score = self._analyze_market_health(health_data)
            score += health_score * 0.25

            # Normalize and determine action
            final_score = self._normalize_score(score)
            confidence = self._calculate_confidence(final_score)
            action = self._determine_action(final_score)

            return {
                "agent": self.name,
                "action": action,
                "confidence": confidence,
                "score": final_score,
                "weight": self.weight,
                "reasoning": self.reasoning
            }
            
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {str(e)}")
            raise
    
    def _analyze_volume_metrics(self, volume_data: Dict) -> float:
        """Analyze volume patterns"""
        score = 0.0
        volume_change = volume_data['volume_change']
        volume_trend = volume_data['volume_trend']
        intensity = volume_data['volume_intensity']
        
        if volume_trend == "Increasing":
            score += 0.5
            if volume_change > 100:
                score += 0.3
                self._add_reasoning("Significant volume increase", "high")
            else:
                self._add_reasoning("Volume trending up", "normal")
        elif volume_trend == "Decreasing":
            score -= 0.3
            self._add_reasoning("Declining volume", "normal")

        if intensity in ["Very High", "High"]:
            score *= 1.2
            self._add_reasoning(f"High volume intensity: {intensity}", "high")
        elif intensity in ["Very Low", "Low"]:
            score *= 0.8
            self._add_reasoning(f"Low volume intensity: {intensity}", "normal")

        return score
    
    def _analyze_market_structure(self, market_data: Dict) -> float:
        """Analyze market structure"""
        score = 0.0
        spread = market_data.get('bid_ask_spread', 0.01)
        
        if spread < 0.001:
            score += 0.5
            self._add_reasoning("Excellent liquidity (tight spread)", "high")
        elif spread < 0.005:
            score += 0.3
            self._add_reasoning("Good liquidity", "normal")
        else:
            score -= 0.3
            self._add_reasoning("Poor liquidity (wide spread)", "high")
            
        return score
    
    def _analyze_market_health(self, health_data: Dict) -> float:
        """Analyze market health metrics"""
        score = 0.0
        
        liquidity = health_data.get('liquidity_score', 5)
        efficiency = health_data.get('efficiency', 5)
        stability = health_data.get('stability', 5)
        
        # Normalize scores to [-1, 1] range
        score += (liquidity - 5) / 5 * 0.4  # 40% weight
        score += (efficiency - 5) / 5 * 0.3  # 30% weight
        score += (stability - 5) / 5 * 0.3   # 30% weight
        
        if liquidity > 7:
            self._add_reasoning("Strong market liquidity", "high")
        elif liquidity < 3:
            self._add_reasoning("Poor market liquidity", "high")
            
        if stability < 3:
            self._add_reasoning("Market instability detected", "high")
            
        return score
    
    def _determine_action(self, score: float) -> str:
        """Determine action based on score"""
        if score > 0.25:
            return "Buy"
        elif score < -0.25:
            return "Sell"
        return "Hold"

class RiskAgent(BaseAgent):
    """Risk Manager - Focused on risk assessment and mitigation"""
    
    def __init__(self):
        super().__init__(weight=0.25)
        self.risk_thresholds = {
            'volatility': {'low': 0.2, 'high': 0.5},
            'rsi': {'oversold': 30, 'overbought': 70},
            'volume': {'significant': 50},
            'spread': {'wide': 0.005}
        }
    
    def analyze(self, analysis: Dict) -> Dict:
        """Analyze risk factors"""
        logger.info(f"[{self.name}] Conducting risk assessment...")
        self.clear_reasoning()
        
        try:
            # Calculate risk factors
            risk_factors = {
                'volatility': self._assess_volatility_risk(analysis['volatility']),
                'momentum': self._assess_momentum_risk(analysis['momentum']),
                'market': self._assess_market_risk(analysis['market_data'], analysis['volume_analysis']),
                'trend': self._assess_trend_risk(analysis['trend'])
            }
            
            # Calculate total risk score (0 to 1, where 1 is highest risk)
            total_risk = sum(score * weight for score, weight in [
                (risk_factors['volatility'], 0.35),
                (risk_factors['momentum'], 0.25),
                (risk_factors['market'], 0.25),
                (risk_factors['trend'], 0.15)
            ])
            
            # Generate risk assessment
            risk_assessment = {
                'risk_level': self._get_risk_level(total_risk),
                'risk_score': total_risk,
                'risk_factors': risk_factors,
                'highest_risk': max(risk_factors.items(), key=lambda x: x[1])[0]
            }
            
            # Determine action based on risk
            action, confidence = self._determine_risk_action(total_risk, analysis)
            
            return {
                "agent": self.name,
                "action": action,
                "confidence": confidence,
                "score": -total_risk,  # Inverse of risk score
                "weight": self.weight,
                "reasoning": self.reasoning,
                "risk_assessment": risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            raise
    
    def _assess_volatility_risk(self, volatility_data: Dict) -> float:
        """Assess risk based on volatility metrics"""
        risk_score = 0.0
        vol_level = volatility_data.get('volatility_level', 'Medium')
        annual_vol = volatility_data.get('annual_volatility', 0.3)
        
        if vol_level == "High":
            risk_score += 0.8
            self._add_reasoning(
                f"High volatility ({annual_vol*100:.1f}% annual) - increased risk",
                "high"
            )
        elif vol_level == "Medium":
            risk_score += 0.5
            self._add_reasoning(
                f"Medium volatility ({annual_vol*100:.1f}% annual)",
                "normal"
            )
        
        return self._normalize_score(risk_score)
    
    def _assess_momentum_risk(self, momentum_data: Dict) -> float:
        """Assess risk based on momentum indicators"""
        risk_score = 0.0
        rsi = momentum_data.get('rsi_value', 50)
        macd_trend = momentum_data.get('macd_trend', 'Neutral')
        
        # Extreme RSI conditions
        if rsi > self.risk_thresholds['rsi']['overbought']:
            risk_score += 0.7
            self._add_reasoning(
                f"Overbought RSI ({rsi:.1f}) - high reversal risk",
                "high"
            )
        elif rsi < self.risk_thresholds['rsi']['oversold']:
            risk_score += 0.6
            self._add_reasoning(
                f"Oversold RSI ({rsi:.1f}) - potential reversal",
                "high"
            )
            
        # MACD trend risk
        if macd_trend == "Bearish":
            risk_score += 0.3
            self._add_reasoning("Bearish MACD trend", "normal")
        
        return self._normalize_score(risk_score)
    
    def _assess_market_risk(self, market_data: Dict, volume_data: Dict) -> float:
        """Assess risk based on market conditions"""
        risk_score = 0.0
        
        # Spread risk
        spread = market_data.get('bid_ask_spread', 0.01)
        if spread > self.risk_thresholds['spread']['wide']:
            risk_score += 0.6
            self._add_reasoning(
                f"Wide spread ({spread*100:.4f}%) indicates liquidity risk",
                "high"
            )
        
        # Volume risk
        volume_change = volume_data.get('volume_change', 0)
        volume_intensity = volume_data.get('volume_intensity', 'Normal')
        
        if abs(volume_change) > self.risk_thresholds['volume']['significant']:
            risk_score += 0.4
            self._add_reasoning(
                f"Significant volume change ({volume_change:.1f}%)",
                "normal"
            )
            
        if volume_intensity in ["Very High", "Very Low"]:
            risk_score += 0.3
            self._add_reasoning(
                f"Extreme volume intensity: {volume_intensity}",
                "normal"
            )
        
        return self._normalize_score(risk_score)
    
    def _assess_trend_risk(self, trend_data: Dict) -> float:
        """Assess risk based on trend analysis"""
        risk_score = 0.0
        trend = trend_data.get('trend', 'Neutral')
        strength = trend_data.get('strength', 0)
        
        if trend == "Strongly Bullish" and strength == 3:
            risk_score += 0.4  # Risk of reversal
            self._add_reasoning(
                "Very strong trend - potential reversal risk",
                "normal"
            )
        elif trend == "Bearish":
            risk_score += 0.3
            self._add_reasoning(
                "Bearish trend - increased downside risk",
                "normal"
            )
            
        return self._normalize_score(risk_score)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score >= 0.7:
            return "Very High"
        elif risk_score >= 0.5:
            return "High"
        elif risk_score >= 0.3:
            return "Medium"
        elif risk_score >= 0.1:
            return "Low"
        return "Very Low"
    
    def _determine_risk_action(self, risk_score: float, analysis: Dict) -> Tuple[str, float]:
        """Determine action based on risk assessment"""
        trend = analysis.get('trend', {}).get('trend', 'Neutral')
        
        if risk_score >= 0.7:
            return "Sell", 90.0  # High confidence in risk-off
        elif risk_score >= 0.5:
            if trend in ["Strongly Bullish", "Bullish"]:
                return "Hold", 70.0  # Reduce exposure but don't exit completely
            else:
                return "Sell", 80.0
        elif risk_score >= 0.3:
            if trend in ["Strongly Bullish", "Bullish"]:
                return "Buy", 60.0  # Cautious buying
            else:
                return "Hold", 70.0
        else:
            if trend in ["Strongly Bullish", "Bullish"]:
                return "Buy", 80.0  # Low risk, follow trend
            else:
                return "Hold", 60.0

class SentimentAgent(BaseAgent):
    """Sentiment Analyst - Market psychology and external factors"""
    
    def __init__(self):
        super().__init__(weight=0.15)
        self.sentiment_thresholds = {
            'strong_positive': 0.7,
            'strong_negative': -0.7,
            'fear_greed_extreme': 80,
            'fear_extreme': 20
        }
    
    def analyze(self, analysis: Dict) -> Dict:
        """Analyze market sentiment"""
        logger.info(f"[{self.name}] Analyzing market sentiment...")
        self.clear_reasoning()
        
        try:
            # Initialize default sentiment data if not present
            if 'sentiment' not in analysis:
                analysis['sentiment'] = {
                    'score': 0,
                    'sentiment': 'Neutral',
                    'description': 'Insufficient data'
                }
            
            if 'social_media_sentiment' not in analysis:
                analysis['social_media_sentiment'] = 0
            
            if 'fear_greed_index' not in analysis:
                analysis['fear_greed_index'] = 50
            
            # Calculate sentiment factors
            sentiment_factors = {
                'market': self._analyze_market_sentiment(analysis),
                'technical': self._analyze_technical_sentiment(analysis),
                'momentum': self._analyze_momentum_sentiment(analysis)
            }
            
            # Calculate total sentiment score
            total_score = sum(sentiment_factors.values()) / len(sentiment_factors)
            final_score = self._normalize_score(total_score)
            
            # Determine action and confidence
            action = self._determine_sentiment_action(final_score)
            confidence = self._calculate_confidence(final_score)
            
            # Add overall sentiment analysis
            if final_score > 0.3:
                overall_sentiment = "Bullish"
            elif final_score < -0.3:
                overall_sentiment = "Bearish"
            else:
                overall_sentiment = "Neutral"
            
            return {
                "agent": self.name,
                "action": action,
                "confidence": confidence,
                "score": final_score,
                "weight": self.weight,
                "reasoning": self.reasoning,
                "sentiment_factors": sentiment_factors,
                "overall_sentiment": overall_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise
    
    def _analyze_market_sentiment(self, analysis: Dict) -> float:
        """Analyze market sentiment"""
        trend = analysis.get('trend', {}).get('trend', 'Neutral')
        trend_strength = analysis.get('trend', {}).get('strength', 0)
        
        score = 0.0
        
        if trend == "Strongly Bullish":
            score = 0.8
            self._add_reasoning("Strong bullish trend", "high")
        elif trend == "Bullish":
            score = 0.4
            self._add_reasoning("Bullish trend", "normal")
        elif trend == "Bearish":
            score = -0.4
            self._add_reasoning("Bearish trend", "normal")
            
        score *= (trend_strength / 3)  # Scale by trend strength
        return score
    
    def _analyze_technical_sentiment(self, analysis: Dict) -> float:
        """Analyze sentiment from technical indicators"""
        momentum = analysis.get('momentum', {}).get('overall_momentum', 'Neutral')
        rsi = analysis.get('momentum', {}).get('rsi_value', 50)
        
        score = 0.0
        
        if momentum == "Strong Bullish":
            score = 0.6
            self._add_reasoning("Strong bullish momentum", "high")
        elif momentum == "Bullish":
            score = 0.3
            self._add_reasoning("Bullish momentum", "normal")
        elif momentum == "Bearish":
            score = -0.3
            self._add_reasoning("Bearish momentum", "normal")
        elif momentum == "Strong Bearish":
            score = -0.6
            self._add_reasoning("Strong bearish momentum", "high")
        
        # Adjust for RSI extremes
        if rsi > 70:
            score -= 0.2
            self._add_reasoning("Overbought conditions (RSI)", "normal")
        elif rsi < 30:
            score += 0.2
            self._add_reasoning("Oversold conditions (RSI)", "normal")
        
        return score
    
    def _analyze_momentum_sentiment(self, analysis: Dict) -> float:
        """Analyze sentiment from momentum indicators"""
        vol_level = analysis.get('volatility', {}).get('volatility_level', 'Medium')
        price_change = analysis.get('market_data', {}).get('change_24h', 0)
        
        score = 0.0
        
        if vol_level == 'High':
            score -= 0.2
            self._add_reasoning("High volatility - cautious sentiment", "normal")
        elif vol_level == 'Low':
            score += 0.2
            self._add_reasoning("Low volatility - stable sentiment", "normal")
        
        if abs(price_change) > 5:
            if price_change > 0:
                score += 0.3
                self._add_reasoning(f"Strong positive price action: +{price_change:.1f}%", "high")
            else:
                score -= 0.3
                self._add_reasoning(f"Strong negative price action: {price_change:.1f}%", "high")
        
        return score
    
    def _determine_sentiment_action(self, score: float) -> str:
        """Determine action based on sentiment score"""
        if score > 0.3:
            return "Buy"
        elif score < -0.3:
            return "Sell"
        return "Hold"

class AIAnalystTeam:
    """Coordinating team of AI agents"""
    
    def __init__(self):
        # Initialize all agents
        self.agents = {
            "Technical": TechnicianAgent(),
            "Fundamental": FundamentalAgent(),
            "Risk": RiskAgent(),
            "Sentiment": SentimentAgent()
        }

    def _get_market_condition(self, agent_analyses: Dict) -> str:
        """Determine overall market condition from agent analyses"""
        if not agent_analyses:
            return "Unknown"
            
        # Extract technical analysis if available
        tech_analysis = agent_analyses.get('Technical', {})
        trend = tech_analysis.get('action', 'Unknown')
        
        # Map conditions
        condition_map = {
            'Buy': 'Bullish',
            'Sell': 'Bearish',
            'Hold': 'Neutral'
        }
        
        return condition_map.get(trend, 'Unknown')
    
    def generate_analysis(self, analysis: Dict) -> Dict:
        """Generate comprehensive analysis using all agents"""
        logger.info("[AIAnalystTeam] Starting team analysis...")
        
        try:
            # Collect individual analyses
            agent_analyses = {}
            for name, agent in self.agents.items():
                try:
                    # Clear previous reasoning
                    agent.clear_reasoning()
                    # Generate new analysis
                    result = agent.analyze(analysis)
                    agent_analyses[name] = result
                    logger.info(f"[{name}] Analysis complete: {result['action']} ({result['confidence']:.1f}% confidence)")
                except Exception as e:
                    logger.error(f"Error in {name} analysis: {e}")
                    continue
    
            # Calculate weighted consensus
            consensus = self._calculate_consensus(agent_analyses)
            
            # Get risk assessment from Risk agent
            risk_assessment = agent_analyses.get("Risk", {}).get('risk_assessment', {
                'risk_level': 'Unknown',
                'risk_score': 0.5
            })
            
            # Generate strategy based on consensus and analysis
            analysis['consensus'] = consensus
            analysis['risk_assessment'] = risk_assessment
            strategy = self._generate_strategy(analysis)
            
            return {
                "consensus": consensus,
                "agent_analyses": agent_analyses,
                "risk_assessment": risk_assessment,
                "strategy": strategy,
                "summary": self._generate_summary(consensus, agent_analyses, risk_assessment)
            }
            
        except Exception as e:
            logger.error(f"Team analysis failed: {str(e)}")
            raise AnalysisError(f"Team analysis failed: {str(e)}")
    
    def _calculate_consensus(self, agent_analyses: Dict) -> Dict:
        """Calculate weighted consensus from all agents"""
        action_scores = {"Buy": 0.0, "Sell": 0.0, "Hold": 0.0}
        total_weight = 0.0
        reasoning = []
        
        # Calculate weighted votes
        for name, analysis in agent_analyses.items():
            if not analysis:  # Skip if analysis failed
                continue
            
            weight = analysis['weight']
            confidence = analysis['confidence'] / 100.0
            action = analysis['action']
            
            # Weight the vote by both agent weight and confidence
            weighted_vote = weight * confidence
            action_scores[action] += weighted_vote
            total_weight += weighted_vote
            
            # Add weighted reasoning
            reasoning.extend(analysis['reasoning'])
        
        # Normalize scores
        if total_weight > 0:
            action_scores = {k: v/total_weight for k, v in action_scores.items()}
        
        # Determine final action
        final_action = max(action_scores.items(), key=lambda x: x[1])[0]
        confidence = action_scores[final_action] * 100
        
        # Check for close decisions
        second_best = sorted(action_scores.items(), key=lambda x: x[1])[-2]
        is_close_decision = (action_scores[final_action] - second_best[1]) < 0.1
        
        if is_close_decision:
            confidence *= 0.8  # Reduce confidence for close decisions
            reasoning.append("Close decision between multiple actions - reducing confidence")
        
        return {
            "action": final_action,
            "confidence": confidence,
            "action_scores": action_scores,
            "reasoning": reasoning,
            "is_close_decision": is_close_decision
        }
    
    def _generate_strategy(self, analysis: Dict) -> Dict:
        """Generate comprehensive analysis-based strategy"""
        try:
            # Extract required data from analysis
            market_data = analysis.get('market_data', {})
            current_price = market_data.get('last_price')
            
            # Get consensus and risk assessment
            consensus = analysis.get('consensus', {})
            risk_assessment = analysis.get('risk_assessment', {})
            
            # Extract action and risk level
            action = consensus.get('action', 'Hold')
            risk_level = risk_assessment.get('risk_level', 'Medium')
    
            if not current_price:
                logger.error("Missing current price data")
                return {}
    
            # For Hold action, return minimal strategy
            if action == "Hold":
                return {
                    'action': 'Hold',
                    'entry': {
                        'method': 'Wait',
                        'points': {
                            'price_levels': [],
                            'allocation': []
                        }
                    },
                    'exit': {
                        'take_profit': {
                            'levels': [],
                            'allocations': []
                        },
                        'stop_loss': {
                            'main_stop': None
                        }
                    }
                }
    
            # Set default stop loss percentage based on risk level
            sl_percentages = {
                'Low': 0.02,     # 2%
                'Medium': 0.015, # 1.5%
                'High': 0.01     # 1%
            }
            sl_percent = sl_percentages.get(risk_level, 0.02)
            
            # Set take profit percentage (usually 1.5-2x the risk)
            tp_percent = sl_percent * 2
            
            # Calculate entry, stop loss, and take profit prices
            if action == "Buy":
                entry_price = current_price
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:  # Sell
                entry_price = current_price
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
            
            strategy = {
                'action': action,
                'entry': {
                    'method': 'Single',
                    'points': {
                        'price_levels': [entry_price],
                        'allocation': [1.0]
                    }
                },
                'exit': {
                    'take_profit': {
                        'levels': [take_profit],
                        'allocations': [1.0]
                    },
                    'stop_loss': {
                        'main_stop': stop_loss
                    }
                }
            }
            
            logger.info(f"Generated strategy: {action} with {risk_level} risk")
            return strategy
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {}
        
    def _calculate_entry_points(self, price: float, action: str, volatility: Dict, risk_level: str) -> Dict:
        """Calculate entry levels based on volatility and risk"""
        try:
            atr = volatility.get('atr', price * 0.02)
            
            # Determine number of entry points based on risk
            num_entries = {'Low': 1, 'Medium': 2, 'High': 3}.get(risk_level, 2)
            
            levels = []
            allocations = []
            
            if action == "Buy":
                for i in range(num_entries):
                    level = price * (1 - (i + 1) * (atr / price))
                    levels.append(round(level, 8))
                    allocations.append(1.0 / num_entries)
            else:
                for i in range(num_entries):
                    level = price * (1 + (i + 1) * (atr / price))
                    levels.append(round(level, 8))
                    allocations.append(1.0 / num_entries)
            
            return {
                'levels': levels,
                'allocations': allocations
            }
            
        except Exception as e:
            logger.error(f"Error calculating entry points: {str(e)}")
            return {'levels': [price], 'allocations': [1.0]}
    
    def _calculate_exit_points(self, price: float, action: str, volatility: Dict, risk_level: str) -> Dict:
        """Calculate exit levels including take profit and stop loss"""
        try:
            atr = volatility.get('atr', price * 0.02)
            
            # Define take profit levels based on risk
            tp_multipliers = {
                'Low': [2.0],  # Conservative - single TP at 2x ATR
                'Medium': [1.5, 2.5],  # Moderate - two TPs
                'High': [1.0, 2.0, 3.0]  # Aggressive - three TPs
            }.get(risk_level, [2.0])  # Default to conservative
            
            # Calculate take profit levels and allocations
            tp_levels = []
            tp_allocations = []
            
            if action == "Buy":
                # For Buy: TPs above entry
                for mult in tp_multipliers:
                    tp_level = price * (1 + (mult * (atr / price)))
                    tp_levels.append(round(tp_level, 8))
                    
                # Stop loss below entry - tighter for higher risk levels
                sl_multiplier = 1.0 if risk_level == 'Low' else 0.8 if risk_level == 'Medium' else 0.6
                sl_main = price * (1 - (sl_multiplier * (atr / price)))
                sl_secondary = price * (1 - (sl_multiplier * 1.5 * (atr / price)))
                
            else:  # Sell
                # For Sell: TPs below entry
                for mult in tp_multipliers:
                    tp_level = price * (1 - (mult * (atr / price)))
                    tp_levels.append(round(tp_level, 8))
                    
                # Stop loss above entry
                sl_multiplier = 1.0 if risk_level == 'Low' else 0.8 if risk_level == 'Medium' else 0.6
                sl_main = price * (1 + (sl_multiplier * (atr / price)))
                sl_secondary = price * (1 + (sl_multiplier * 1.5 * (atr / price)))
            
            # Calculate allocations based on number of TP levels
            allocation_per_level = 1.0 / len(tp_levels)
            tp_allocations = [allocation_per_level] * len(tp_levels)
            
            return {
                'take_profit': {
                    'levels': tp_levels,
                    'allocations': tp_allocations
                },
                'stop_loss': {
                    'main_stop': round(sl_main, 8),
                    'secondary_stop': round(sl_secondary, 8),
                    'breakeven_trigger': f"{(atr/price*100):.1f}% in profit",
                    'trailing_activation': f"{(2*atr/price*100):.1f}% in profit"  # Activate trailing at 2x ATR
                },
                'risk_reward': {
                    'ratio': round((tp_levels[0] - price) / (price - sl_main), 2) if action == "Buy"
                             else round((price - tp_levels[0]) / (sl_main - price), 2),
                    'risk_per_trade': f"{abs(price - sl_main) / price * 100:.2f}%"
                }
            }
                
        except Exception as e:
            logger.error(f"Error calculating exit points: {str(e)}")
            return {
                'take_profit': {'levels': [], 'allocations': []},
                'stop_loss': {
                    'main_stop': price,
                    'secondary_stop': price,
                    'breakeven_trigger': '0% in profit',
                    'trailing_activation': '0% in profit'
                },
                'risk_reward': {'ratio': 0, 'risk_per_trade': '0%'}
            }
    
    def _recommend_timeframe(self, risk_level: str) -> str:
        """Recommend trading timeframe based on risk level"""
        timeframes = {
            'Low': '1d',
            'Medium': '4h',
            'High': '1h'
        }
        return timeframes.get(risk_level, '4h')
    
    def _generate_risk_management(self, analysis: Dict, consensus: Dict) -> Dict:
        """Generate risk management rules"""
        try:
            risk_level = analysis.get('risk_assessment', {}).get('risk_level', 'Medium')
            volatility = analysis['volatility']
            
            # Position limits based on risk level
            position_limits = {
                'Low': {'max': 1.0, 'min': 0.2},
                'Medium': {'max': 0.7, 'min': 0.15},
                'High': {'max': 0.4, 'min': 0.1}
            }.get(risk_level, {'max': 0.5, 'min': 0.1})
            
            # Adjust for volatility
            if volatility.get('volatility_level') == 'High':
                position_limits['max'] *= 0.8
            
            # Generate monitoring rules
            monitoring = [
                "Monitor volume for significant changes",
                "Track price action around key levels",
                "Watch for sentiment shifts"
            ]
            
            if risk_level in ['High', 'Very High']:
                monitoring.extend([
                    "Continuous volatility monitoring",
                    "Real-time news sentiment tracking"
                ])
            
            return {
                'position_limits': position_limits,
                'monitoring': monitoring,
                'adjustment_rules': [
                    "Adjust position size based on confirmation",
                    "Scale out at resistance/support levels",
                    "Lock in profits at key levels"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating risk management: {str(e)}")
            return {
                'position_limits': {'max': 0.5, 'min': 0.1},
                'monitoring': ["Monitor price action"],
                'adjustment_rules': ["Adjust as needed"]
            }

    def _get_trend_analysis(self, agent_analyses: Dict) -> str:
        """Extract trend analysis from agent insights"""
        technical = agent_analyses.get('Technical', {})
        fundamental = agent_analyses.get('Fundamental', {})
        
        # Get trend signals
        tech_action = technical.get('action', 'Unknown')
        tech_confidence = technical.get('confidence', 0)
        fund_action = fundamental.get('action', 'Unknown')
        
        if tech_confidence > 80:
            return f"Strong {tech_action} signal ({tech_confidence:.1f}% confidence)"
        elif tech_action == fund_action:
            return f"Confirmed {tech_action} trend"
        else:
            return "Mixed trend signals"
    
    def _generate_summary(self, consensus: Dict, agent_analyses: Dict, risk_assessment: Dict) -> str:
        try:
            summary = [
                f"Final Recommendation: {consensus['action']} with {consensus['confidence']:.1f}% confidence\n",
                "Market Analysis:",
                f"• Market Condition: {self._get_market_condition(agent_analyses)}",
                f"• Risk Level: {risk_assessment.get('risk_level', 'Unknown')}",
                f"• Trend Analysis: {self._get_trend_analysis(agent_analyses)}",
                "\nKey Factors:",
                "• Technical Indicators:",
                *[f"  - {reason}" for reason in agent_analyses.get('Technical', {}).get('reasoning', [])],
                "\n• Fundamental Analysis:",
                *[f"  - {reason}" for reason in agent_analyses.get('Fundamental', {}).get('reasoning', [])],
                "\n• Risk Assessment:",
                *[f"  - {reason}" for reason in agent_analyses.get('Risk', {}).get('reasoning', [])],
                "\nAction Plan:",
                f"• Recommended Position Size: {consensus.get('position_size', 0):.1f}%",
                "• Entry/Exit Strategy:",
                "  - Use specified entry points with proper position sizing",
                "  - Place stop loss orders at designated levels",
                "  - Consider market volatility for position adjustment"
            ]
            
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating analysis summary"

class StrategyAgent:
    """Advanced strategy generation with risk-adjusted position sizing"""
    
    def _generate_strategy(self, analysis: Dict) -> Dict:
        """Generate comprehensive analysis-based strategy"""
        try:
            consensus = analysis.get('consensus', {})
            risk_assessment = analysis.get('risk_assessment', {})
            market_data = analysis.get('market_data', {})
            
            # Get required parameters
            action = consensus.get('action', 'Hold')
            risk_level = risk_assessment.get('risk_level', 'Medium')
            current_price = market_data.get('last_price')
    
            if not current_price:
                logger.error("Missing current price data")
                return {}
    
            # Set default stop loss percentage based on risk level
            sl_percentages = {
                'Low': 0.02,     # 2%
                'Medium': 0.015, # 1.5%
                'High': 0.01     # 1%
            }
            sl_percent = sl_percentages.get(risk_level, 0.02)
            
            # Set take profit percentage (usually 1.5-2x the risk)
            tp_percent = sl_percent * 2
            
            # Calculate entry, stop loss, and take profit prices
            if action == "Buy":
                entry_price = current_price
                stop_loss = entry_price * (1 - sl_percent)
                take_profit = entry_price * (1 + tp_percent)
            else:  # Sell
                entry_price = current_price
                stop_loss = entry_price * (1 + sl_percent)
                take_profit = entry_price * (1 - tp_percent)
            
            return {
                'action': action,
                'entry': {
                    'method': 'Single',
                    'points': {
                        'price_levels': [entry_price],
                        'allocation': [1.0]
                    }
                },
                'exit': {
                    'take_profit': {
                        'levels': [take_profit],
                        'allocations': [1.0]
                    },
                    'stop_loss': {
                        'main_stop': stop_loss
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return {}
    
    def _calculate_position_size(self, confidence: float, risk_level: str) -> Dict:
        """Calculate recommended position size based on confidence and risk"""
        base_size = confidence / 100.0  # Start with confidence-based size
        
        # Risk-based adjustments
        risk_multipliers = {
            "Very Low": 1.0,
            "Low": 0.8,
            "Medium": 0.6,
            "High": 0.4,
            "Very High": 0.2
        }
        
        adjusted_size = base_size * risk_multipliers.get(risk_level, 0.5)
        max_position = min(adjusted_size * 100, 100)  # Cap at 100%
        
        return {
            "max_position": max_position,
            "suggested_size": max_position * 0.8,  # Conservative suggestion
            "risk_level": risk_level,
            "confidence": confidence
        }
    
    def _generate_entry_strategy(self, action: str, market_data: Dict, risk_level: str) -> Dict:
        """Generate detailed entry strategy"""
        current_price = market_data['last_price']
        
        if action == "Buy":
            entry_points = self._calculate_entry_points(current_price, action, risk_level)
            method = "Scaled Buy" if risk_level in ["High", "Very High"] else "Direct Entry"
        elif action == "Sell":
            entry_points = self._calculate_entry_points(current_price, action, risk_level)
            method = "Scaled Sell" if risk_level in ["High", "Very High"] else "Direct Entry"
        else:  # Hold
            entry_points = self._calculate_range_points(current_price, risk_level)
            method = "Range Trade"
        
        return {
            "method": method,
            "points": entry_points,
            "conditions": self._generate_entry_conditions(action, risk_level),
            "timeframe": self._recommend_timeframe(risk_level, action)
        }
    
    def _generate_exit_strategy(self, action: str, market_data: Dict, risk_assessment: Dict) -> Dict:
        """Generate comprehensive exit strategy"""
        return {
            "take_profit": self._calculate_take_profit_levels(action, market_data),
            "stop_loss": self._calculate_stop_loss_levels(action, market_data, risk_assessment),
            "trailing_stop": self._calculate_trailing_stop(action, risk_assessment),
            "time_based": self._generate_time_based_exits(action, risk_assessment)
        }
    
    def _generate_risk_management(self, action: str, risk_assessment: Dict, market_data: Dict) -> Dict:
        """Generate risk management rules"""
        return {
            "position_limits": self._calculate_position_limits(risk_assessment),
            "stop_loss": self._calculate_stop_loss_levels(action, market_data, risk_assessment),
            "monitoring": self._generate_monitoring_rules(risk_assessment),
            "adjustments": self._generate_adjustment_rules(action, risk_assessment)
        }
    
    def _recommend_timeframe(self, risk_level: str, action: str) -> str:
        """Recommend trading timeframe based on risk and action"""
        timeframes = {
            "Very Low": "1d",
            "Low": "4h",
            "Medium": "1h",
            "High": "15m",
            "Very High": "5m"
        }
        return timeframes.get(risk_level, "1h")
    
    def _calculate_entry_points(self, current_price: float, action: str, risk_level: str) -> Dict:
        """Calculate strategic entry points based on action and risk"""
        # Define scaling factors based on risk
        scale_factors = {
            "Very Low": [1.0],  # Single entry
            "Low": [0.6, 0.4],  # Two entries
            "Medium": [0.4, 0.3, 0.3],  # Three entries
            "High": [0.3, 0.3, 0.2, 0.2],  # Four entries
            "Very High": [0.2, 0.2, 0.2, 0.2, 0.2]  # Five entries
        }
        
        factors = scale_factors.get(risk_level, [0.5, 0.5])
        
        # Calculate price levels for entries
        if action == "Buy":
            levels = [
                current_price * (1 - 0.01 * i)  # 1% spacing for buys
                for i in range(len(factors))
            ]
        else:  # Sell
            levels = [
                current_price * (1 + 0.01 * i)  # 1% spacing for sells
                for i in range(len(factors))
            ]
            
        return {
            "price_levels": levels,
            "allocation": factors,
            "method": "Scaled" if len(factors) > 1 else "Single",
            "spacing": "1% between levels"
        }
    
    def _calculate_range_points(self, current_price: float, risk_level: str) -> Dict:
        """Calculate range trading points"""
        # Define range size based on risk
        range_sizes = {
            "Very Low": 0.02,  # 2% range
            "Low": 0.03,
            "Medium": 0.05,
            "High": 0.08,
            "Very High": 0.10
        }
        
        range_size = range_sizes.get(risk_level, 0.05)
        return {
            "upper_bound": current_price * (1 + range_size),
            "lower_bound": current_price * (1 - range_size),
            "mid_point": current_price,
            "range_size": f"{range_size*100}%"
        }
    
    def _generate_entry_conditions(self, action: str, risk_level: str) -> List[str]:
        """Generate specific entry conditions"""
        conditions = []
        
        # Base conditions
        if action == "Buy":
            conditions.extend([
                "Price below or at entry level",
                "Volume above 80% of 24h average",
                "No immediate resistance levels within 2%"
            ])
        elif action == "Sell":
            conditions.extend([
                "Price above or at entry level",
                "Volume above 80% of 24h average",
                "No immediate support levels within 2%"
            ])
        else:  # Hold/Range
            conditions.extend([
                "Price within defined range bounds",
                "Volume within normal ranges",
                "No strong trend developing"
            ])
            
        # Risk-specific conditions
        if risk_level in ["High", "Very High"]:
            conditions.extend([
                "Confirmation from multiple timeframes",
                "Clear support/resistance levels identified",
                "Reduced position size recommended"
            ])
            
        return conditions
    
    def _calculate_take_profit_levels(self, action: str, market_data: Dict) -> Dict:
        """Calculate multiple take-profit levels"""
        current_price = market_data['last_price']
        volatility = market_data.get('volatility', 0.02)  # Default to 2% if not provided
        
        # Adjust profit targets based on volatility
        base_targets = [0.02, 0.035, 0.05]  # 2%, 3.5%, 5%
        adjusted_targets = [t * (1 + volatility/0.02) for t in base_targets]
        
        if action == "Buy":
            levels = [current_price * (1 + t) for t in adjusted_targets]
        else:  # Sell
            levels = [current_price * (1 - t) for t in adjusted_targets]
            
        return {
            "levels": levels,
            "allocations": [0.4, 0.3, 0.3],  # Distribution across levels
            "adjustments": "Auto-adjusted based on volatility"
        }
    
    def _calculate_stop_loss_levels(self, action: str, market_data: Dict, risk_assessment: Dict) -> Dict:
        """Calculate stop-loss levels with multiple safety measures"""
        current_price = market_data['last_price']
        risk_level = risk_assessment['risk_level']
        
        # Base stop distances based on risk
        base_stops = {
            "Very Low": 0.05,  # 5%
            "Low": 0.04,
            "Medium": 0.03,
            "High": 0.02,
            "Very High": 0.015
        }
        
        stop_distance = base_stops.get(risk_level, 0.03)
        
        if action == "Buy":
            main_stop = current_price * (1 - stop_distance)
            secondary_stop = current_price * (1 - stop_distance * 1.5)
        else:  # Sell
            main_stop = current_price * (1 + stop_distance)
            secondary_stop = current_price * (1 + stop_distance * 1.5)
            
        return {
            "main_stop": main_stop,
            "secondary_stop": secondary_stop,
            "breakeven_trigger": f"{stop_distance*100}% in profit",
            "adjustment_rules": self._generate_stop_adjustment_rules(risk_level)
        }
    
    def _calculate_trailing_stop(self, action: str, risk_assessment: Dict) -> Dict:
        """Calculate trailing stop parameters"""
        risk_level = risk_assessment['risk_level']
        
        # Define trailing distances based on risk
        trail_distances = {
            "Very Low": 0.03,  # 3%
            "Low": 0.025,
            "Medium": 0.02,
            "High": 0.015,
            "Very High": 0.01
        }
        
        trail_distance = trail_distances.get(risk_level, 0.02)
        activation_trigger = trail_distance * 2  # Activate trailing at 2x the trail distance
        
        return {
            "trail_distance": trail_distance,
            "activation_trigger": activation_trigger,
            "adjustment_speed": "Dynamic based on volatility",
            "lock_in_rules": self._generate_lock_in_rules(risk_level)
        }
    
    def _generate_time_based_exits(self, action: str, risk_assessment: Dict) -> Dict:
        """Generate time-based exit rules"""
        risk_level = risk_assessment['risk_level']
        
        # Define maximum hold times based on risk
        max_hold_times = {
            "Very Low": "7 days",
            "Low": "3 days",
            "Medium": "24 hours",
            "High": "8 hours",
            "Very High": "4 hours"
        }
        
        return {
            "max_hold_time": max_hold_times.get(risk_level, "24 hours"),
            "review_intervals": self._generate_review_intervals(risk_level),
            "extension_conditions": self._generate_extension_conditions(action)
        }
    
    def _calculate_position_limits(self, risk_assessment: Dict) -> Dict:
        """Calculate position size limits based on risk"""
        risk_level = risk_assessment['risk_level']
        
        # Maximum position sizes based on risk
        max_sizes = {
            "Very Low": 1.0,    # 100% of normal size
            "Low": 0.8,         # 80% of normal size
            "Medium": 0.6,      # 60% of normal size
            "High": 0.4,        # 40% of normal size
            "Very High": 0.2    # 20% of normal size
        }
        
        return {
            "max_size": max_sizes.get(risk_level, 0.5),
            "recommended_size": max_sizes.get(risk_level, 0.5) * 0.8,  # 80% of max
            "min_position": 0.1,  # 10% minimum position size
            "scaling_rules": self._generate_scaling_rules(risk_level)
        }
    
    def _generate_monitoring_rules(self, risk_assessment: Dict) -> List[str]:
        """Generate monitoring rules based on risk"""
        risk_level = risk_assessment['risk_level']
        
        base_rules = [
            "Monitor volume for significant changes",
            "Track price action around key levels",
            "Watch for sentiment shifts"
        ]
        
        if risk_level in ["High", "Very High"]:
            base_rules.extend([
                "Continuous volatility monitoring",
                "Real-time news sentiment tracking",
                "Multiple timeframe confirmation"
            ])
            
        return base_rules
    
    def _generate_adjustment_rules(self, action: str, risk_assessment: Dict) -> List[str]:
        """Generate position adjustment rules"""
        base_rules = [
            "Adjust position size based on confirmation of trend",
            "Scale out at resistance/support levels",
            f"{('Pyramiding' if action == 'Buy' else 'Scaling')} allowed only with trend confirmation"
        ]
        
        if risk_assessment['risk_level'] in ["High", "Very High"]:
            base_rules.extend([
                "No position increases without clear confirmation",
                "Faster take-profit scaling",
                "Tighter stop-loss management"
            ])
            
        return base_rules
    
    def _generate_stop_adjustment_rules(self, risk_level: str) -> List[str]:
        """Generate rules for stop-loss adjustments"""
        rules = [
            "Move to breakeven after initial target hit",
            "Trail profit after second target"
        ]
        
        if risk_level in ["High", "Very High"]:
            rules.extend([
                "Tighten stops on volatility increase",
                "Use shorter-timeframe signals for exits"
            ])
            
        return rules
    
    def _generate_lock_in_rules(self, risk_level: str) -> List[str]:
        """Generate rules for locking in profits"""
        rules = [
            "Lock 50% of profits at first target",
            "Full lock-in at second target"
        ]
        
        if risk_level in ["High", "Very High"]:
            rules.extend([
                "Dynamic trailing based on volatility",
                "Accelerated lock-in on volume spikes"
            ])
            
        return rules
    
    def _generate_review_intervals(self, risk_level: str) -> List[str]:
        """Generate position review intervals"""
        base_intervals = ["Daily market open", "Major support/resistance tests"]
        
        if risk_level in ["High", "Very High"]:
            base_intervals.extend([
                "Every 4 hours",
                "On significant volume events",
                "After news releases"
            ])
            
        return base_intervals
    
    def _generate_extension_conditions(self, action: str) -> List[str]:
        """Generate conditions for extending position hold time"""
        return [
            f"Strong trend continuation in {'upward' if action == 'Buy' else 'downward'} direction",
            "Volume supporting price action",
            "No reversal signals on higher timeframes",
            "Risk level maintains or improves"
        ]
    
    def _generate_scaling_rules(self, risk_level: str) -> List[str]:
        """Generate position scaling rules"""
        base_rules = [
            "Scale in on pullbacks to support",
            "Scale out into strength"
        ]
        
        if risk_level in ["High", "Very High"]:
            base_rules.extend([
                "Maximum 3 scale-in points",
                "Minimum 25% position size per entry",
                "No scaling after 50% drawdown"
            ])
            
        return base_rules

class ChartBuilder:
    """Enhanced chart creation with interactive features"""
    # Define chart colors as class variables
    MA_COLORS = {
        'sma_20': '#FF1493',    # Deep Pink
        'sma_50': '#4169E1',    # Royal Blue
        'sma_200': '#32CD32',   # Lime Green
        'ema_20': '#FFD700',    # Gold
        'ema_50': '#FF4500',    # Orange Red
        'ema_200': '#9370DB',   # Medium Purple
    }
    
    BB_COLORS = {
        'bbands_upper': '#E6B333',    # Gold
        'bbands_middle': '#33E6CC',   # Turquoise
        'bbands_lower': '#E633FF'     # Pink
    }
    
    @staticmethod
    def _convert_to_array(data) -> np.ndarray:
        """Convert input data to numpy array safely"""
        if isinstance(data, (float, int)):
            return np.array([data])
        if isinstance(data, pd.Series):
            return data.values
        if isinstance(data, list):
            return np.array(data)
        if isinstance(data, np.ndarray):
            return data
        return np.array([data])
        
    @classmethod
    def create_price_chart(cls, data: pd.DataFrame, analysis: Dict) -> go.Figure:
        """Create advanced price chart with indicators"""
        
        # Create subplots for price and volume with more spacing
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,  # Increased spacing between subplots
            row_heights=[0.7, 0.3],
            subplot_titles=("Price & Indicators", "Volume Analysis")
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='Price',
                legendgroup='Price',
                legendgrouptitle_text='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages with distinct colors
        if 'indicators' in analysis:
            # Sort indicators by period to ensure consistent legend order
            ma_indicators = []
            for ma_type in ['SMA', 'EMA']:
                for period in [20, 50, 200]:
                    ma_key = f'{ma_type.lower()}_{period}'
                    if ma_key in analysis['indicators']:
                        ma_indicators.append((ma_key, ma_type, period))
            
            # Sort by type and period
            ma_indicators.sort(key=lambda x: (x[1], x[2]))
            
            # Add traces in sorted order
            for ma_key, ma_type, period in ma_indicators:
                values = cls._convert_to_array(analysis['indicators'][ma_key])
                color = cls.MA_COLORS.get(ma_key, '#000000')
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=values,
                        name=f'{ma_type}({period})',
                        line=dict(
                            width=2,
                            color=color
                        ),
                        opacity=0.8,
                        legendgroup=ma_type,
                        legendgrouptitle_text=ma_type
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands with distinct colors
        bb_keys = ['bbands_upper', 'bbands_middle', 'bbands_lower']
        if 'indicators' in analysis and all(k in analysis['indicators'] for k in bb_keys):
            for key in bb_keys:
                values = cls._convert_to_array(analysis['indicators'][key])
                color = cls.BB_COLORS.get(key, '#000000')
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=values,
                        name=key.replace('bbands_', 'BB ').title(),
                        line=dict(
                            width=1,
                            dash='dash',
                            color=color
                        ),
                        opacity=0.5,
                        legendgroup='Bollinger',
                        legendgrouptitle_text='Bollinger Bands'
                    ),
                    row=1, col=1
                )
        
        # Add volume bars with distinct colors for up/down
        colors = ['#2ca02c' if close > open else '#d62728'  # Green for up, Red for down 
                 for open, close in zip(data['open'], data['close'])]
                 
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.8,
                legendgroup='Volume',
                legendgrouptitle_text='Volume'
            ),
            row=2, col=1
        )
        
        # Add volume MA with distinct color
        volume_ma = data['volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volume_ma,
                name='Volume MA(20)',
                line=dict(
                    color='#17becf',  # Light blue
                    width=2
                ),
                opacity=0.7,
                legendgroup='Volume'
            ),
            row=2, col=1
        )
        
        # Update layout with better spacing and organization
        fig.update_layout(
            height=800,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            title_text="Price Analysis Chart with Indicators",
            title_x=0,  # Left Aligned
            title_font_size=20,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0, 0, 0, 0.8)',
                font=dict(size=10, color='white'),
                bordercolor='rgba(255, 255, 255, 0.3)',
                borderwidth=1,
                itemwidth=30
            ),
            margin=dict(l=50, r=50, t=120, b=50),  # Increased top margin for legend
            template="plotly_dark",
            width=None,  # Allow dynamic width
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
        )
        
        # Update axes labels and grid
        fig.update_yaxes(
            title_text="Price",
            row=1,
            col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_yaxes(
            title_text="Volume",
            row=2,
            col=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)'
        )
        
        fig.update_xaxes(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zerolinecolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        )
        
        # Add chart watermark
        fig.add_annotation(
            text="Crypto Analysis Assistant",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=30, color='rgba(128, 128, 128, 0.1)'),
            textangle=-30
        )
        
        return fig
        
    @staticmethod
    def create_indicators_chart(data: pd.DataFrame, analysis: Dict) -> go.Figure:
        """Create comprehensive technical indicators chart"""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("RSI & Stochastic", "MACD", "ATR & BB Width"),
            row_heights=[0.33, 0.33, 0.33]
        )
        
        # Ensure indicators exist
        if 'indicators' not in analysis:
            return fig
            
        indicators = analysis['indicators']
        
        # RSI
        if 'rsi' in indicators:
            rsi_values = pd.Series(indicators['rsi'], index=data.index)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=rsi_values,
                    name='RSI',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
        # Add RSI levels
        fig.add_hline(y=70, line_color="red", line_dash="dash", row=1, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dash", row=1, col=1)
        
        # MACD
        if all(k in indicators for k in ['macd', 'macd_signal']):
            # MACD Line
            macd_values = pd.Series(indicators['macd'], index=data.index)
            signal_values = pd.Series(indicators['macd_signal'], index=data.index)
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=macd_values,
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=signal_values,
                    name='Signal',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            
            # MACD Histogram
            hist_values = macd_values - signal_values
            colors = ['red' if x < 0 else 'green' for x in hist_values]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=hist_values,
                    name='MACD Hist',
                    marker_color=colors,
                    opacity=0.5
                ),
                row=2, col=1
            )
        
        # ATR
        if 'atr' in indicators:
            atr_values = pd.Series(indicators['atr'], index=data.index)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=atr_values,
                    name='ATR',
                    line=dict(color='blue', width=1)
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Technical Indicators Analysis",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_dark"
        )
        
        # Update axes labels
        fig.update_yaxes(title_text="RSI", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="ATR", row=3, col=1)
        
        return fig

def fetch_external_sentiment_data() -> Dict[str, Any]:
    # Simulate fetching Fear & Greed index and Social Media sentiment
    # To produce more varied scenarios, vary the values randomly
    # but here we keep static since no user input:
    fear_greed_index = 80  # More greedy to push more buy signals
    social_media_sentiment = 0.4  # Positive sentiment
    return {
        "fear_greed_index": fear_greed_index,
        "social_media_sentiment": social_media_sentiment
    }

def display_technical_analysis(analysis: Dict):
    """Display comprehensive technical analysis results"""
    st.markdown("""
        <style>
            .block-container {
                max-width: 100% !important;
                width: 100% !important;
                padding: 2rem 5rem !important;
            }
            .element-container, .stPlotlyChart {
                width: 100% !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.subheader("Technical Analysis Summary")
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend = analysis['trend']
        st.metric(
            "Trend",
            trend['trend'],
            f"{trend['sma20_diff']:.2f}% vs SMA20",
            delta_color="normal"
        )
        
    with col2:
        momentum = analysis['momentum']
        st.metric(
            "Momentum",
            momentum['overall_momentum'],
            f"RSI: {momentum['rsi_value']:.1f}"
        )
        
    with col3:
        vol = analysis['volatility']
        st.metric(
            "Volatility",
            vol['volatility_level'],
            f"ATR: {vol['atr']:.2f}"
        )
    
    # Support and Resistance
    st.subheader("Key Price Levels")
    sr_data = analysis['support_resistance']
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Resistance Levels**")
        for level in sr_data['resistance_levels']:
            st.write(f"• {level:.2f}")
            
    with col2:
        st.write("**Support Levels**")
        for level in sr_data['support_levels']:
            st.write(f"• {level:.2f}")
            
    st.write(f"**Pivot Point:** {sr_data['pivot_point']:.2f}")

def display_risk_analysis(analysis: Dict):
    """Display risk analysis and metrics"""
    st.subheader("Risk Analysis")
    
    # Risk Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        vol_data = analysis['volatility']
        st.metric(
            "Annual Volatility",
            f"{vol_data['annual_volatility']*100:.1f}%",
            vol_data['volatility_level']
        )
        
        st.metric(
            "BB Width",
            f"{vol_data['bollinger_width']*100:.1f}%"
        )
        
    with col2:
        st.metric(
            "ATR",
            f"{vol_data['atr']:.2f}"
        )
        
        if 'risk_level' in analysis:
            st.metric(
                "Overall Risk Level",
                analysis['risk_level']
            )
    
    # Risk Factors
    if 'risk_factors' in analysis:
        st.subheader("Risk Factors")
        for factor, value in analysis['risk_factors'].items():
            st.write(f"**{factor.title()}:** {value:.2f}")

def display_strategy_recommendations(analysis: Dict):
    """Display strategy recommendations and trade setup"""
    st.subheader("Trading Strategy")
    
    if 'strategy' in analysis:
        strategy = analysis['strategy']
        
        # Entry Strategy
        st.write("**Entry Strategy**")
        entry = strategy['entry']
        st.write(f"Method: {entry['method']}")
        for i, (price, alloc) in enumerate(zip(
            entry['points']['price_levels'],
            entry['points']['allocation']
        )):
            st.write(f"• Entry {i+1}: {price:.2f} ({alloc*100:.0f}% allocation)")
            
        # Exit Strategy
        st.write("**Exit Strategy**")
        exit_strategy = strategy['exit']
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Take Profit Levels:")
            for i, (price, alloc) in enumerate(zip(
                exit_strategy['take_profit']['levels'],
                exit_strategy['take_profit']['allocations']
            )):
                st.write(f"• TP {i+1}: {price:.2f} ({alloc*100:.0f}%)")
                
        with col2:
            st.write("Stop Loss Levels:")
            st.write(f"• Main: {exit_strategy['stop_loss']['main_stop']:.2f}")
            st.write(f"• Secondary: {exit_strategy['stop_loss']['secondary_stop']:.2f}")
        
        # Risk Management
        st.write("**Position Sizing**")
        position = strategy['position_size']
        st.write(f"• Maximum Size: {position['max_position']:.1f}%")
        st.write(f"• Suggested Size: {position['suggested_size']:.1f}%")
        
        # Trading Timeframe
        st.write(f"**Recommended Timeframe:** {strategy['timeframe']}")

def display_market_structure(analysis: Dict):
    """Display market structure analysis"""
    st.subheader("Market Structure Analysis")
    
    # Market Data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "24h Volume",
            f"${analysis.get('market_data', {}).get('volume_24h', 0):,.0f}",
            help="Trading volume in the last 24 hours"
        )
    
    with col2:
        st.metric(
            "Bid-Ask Spread",
            f"{analysis.get('market_data', {}).get('bid_ask_spread', 0)*100:.3f}%",
            help="Current spread between bid and ask prices"
        )
        
    with col3:
        st.metric(
            "Price Change 24h",
            f"{analysis.get('market_data', {}).get('change_24h', 0):.2f}%",
            help="Price change in the last 24 hours"
        )
    
    # Market Health
    if 'market_health' in analysis:
        st.subheader("Market Health")
        health = analysis['market_health']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Liquidity Score",
                f"{health.get('liquidity_score', 0):.2f}/10",
                help="Measure of market liquidity"
            )
            
        with col2:
            st.metric(
                "Market Efficiency",
                f"{health.get('efficiency', 0):.2f}/10",
                help="Measure of market efficiency"
            )
            
        with col3:
            st.metric(
                "Market Stability",
                f"{health.get('stability', 0):.2f}/10",
                help="Measure of market stability"
            )

def display_ai_analysis(analysis: Dict):
    """Display AI analysis results and recommendations"""
    st.subheader("AI Analysis")

    # Display consensus if available
    if 'consensus' in analysis:
        consensus = analysis['consensus']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Final Recommendation",
                consensus['action'],
                f"{consensus['confidence']:.1f}% Confidence"
            )
        
        with col2:
            st.metric(
                "Risk Level",
                analysis.get('risk_assessment', {}).get('risk_level', 'Unknown')
            )
        
        # Show action scores in a horizontal bar chart
        if 'action_scores' in consensus:
            scores = consensus['action_scores']
            fig = go.Figure(go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation='h',
                marker_color=['green' if x == max(scores.values()) else 'gray' for x in scores.values()]
            ))
            
            fig.update_layout(
                title="Action Probability Distribution",
                xaxis_title="Probability",
                yaxis_title="Action",
                height=200,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display individual agent analyses
    if 'agent_analyses' in analysis:
        st.subheader("Agent Insights")
        
        cols = st.columns(len(analysis['agent_analyses']))
        for col, (agent_name, agent_analysis) in zip(cols, analysis['agent_analyses'].items()):
            with col:
                st.metric(
                    agent_name,
                    agent_analysis['action'],
                    f"{agent_analysis['confidence']:.1f}% Confidence"
                )
                
                if 'reasoning' in agent_analysis:
                    with st.expander(f"{agent_name} Reasoning"):
                        for reason in agent_analysis['reasoning']:
                            st.write(reason)
    
    # Display risk assessment
    if 'risk_assessment' in analysis:
        st.subheader("Risk Assessment")
        risk_data = analysis['risk_assessment']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'risk_score' in risk_data:
                st.metric(
                    "Risk Score",
                    f"{risk_data['risk_score']:.2f}",
                    help="Overall risk score (0-1)"
                )
                
        with col2:
            if 'risk_level' in risk_data:
                st.metric(
                    "Risk Level",
                    risk_data['risk_level'],
                    help="Qualitative risk assessment"
                )
                
        with col3:
            if 'highest_risk' in risk_data:
                st.metric(
                    "Highest Risk Factor",
                    risk_data['highest_risk'],
                    help="Factor contributing most to risk"
                )
        
        # Display detailed risk factors if available
        if 'risk_factors' in risk_data:
            st.write("**Risk Factors Breakdown:**")
            risk_factors = risk_data['risk_factors']
            
            fig = go.Figure(go.Bar(
                x=list(risk_factors.values()),
                y=list(risk_factors.keys()),
                orientation='h',
                marker_color='red'
            ))
            
            fig.update_layout(
                title="Risk Factors Analysis",
                xaxis_title="Risk Score",
                yaxis_title="Factor",
                height=300,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display sentiment analysis
    if any(k in analysis for k in ['sentiment', 'social_media_sentiment', 'fear_greed_index']):
        st.subheader("Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'sentiment' in analysis:
                st.metric(
                    "Market Sentiment",
                    analysis['sentiment']['sentiment'],
                    f"Score: {analysis['sentiment']['score']:.2f}"
                )
                
        with col2:
            if 'social_media_sentiment' in analysis:
                st.metric(
                    "Social Sentiment",
                    _get_sentiment_label(analysis['social_media_sentiment']),
                    f"Score: {analysis['social_media_sentiment']:.2f}"
                )
                
        with col3:
            if 'fear_greed_index' in analysis:
                st.metric(
                    "Fear & Greed Index",
                    _get_fear_greed_label(analysis['fear_greed_index']),
                    analysis['fear_greed_index']
                )

def display_performance_metrics(analysis: Dict):
    """Display performance metrics and market conditions"""
    st.subheader("Market Performance Metrics")
    
    if 'market_data' in analysis:
        market_data = analysis['market_data']
        
        # Price Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${market_data['last_price']:.4f}",
                f"{market_data['change_24h']:.2f}%",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "24h Volume",
                f"${market_data['volume_24h']:,.0f}",
                help="Trading volume in the last 24 hours"
            )
            
        with col3:
            st.metric(
                "Bid-Ask Spread",
                f"{market_data['bid_ask_spread']*100:.3f}%",
                help="Current spread between bid and ask prices"
            )
        
        # Volume Analysis
        st.subheader("Volume Analysis")
        volume_data = analysis['volume_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Volume Trend",
                volume_data['volume_trend'],
                f"{volume_data['volume_change']:.1f}% vs Average"
            )
            
            st.metric(
                "Volume Intensity",
                volume_data['volume_intensity']
            )
            
        with col2:
            st.metric(
                "Price-Volume Correlation",
                f"{volume_data['price_volume_correlation']:.2f}",
                help="Correlation between price and volume movements"
            )
            
            st.metric(
                "OBV Trend",
                volume_data['obv_trend']
            )
        
        # Sentiment Metrics
        st.subheader("Sentiment Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fg_index = analysis['fear_greed_index']
            st.metric(
                "Fear & Greed Index",
                fg_index,
                _get_fear_greed_label(fg_index)
            )
            
        with col2:
            sentiment = analysis['sentiment']
            st.metric(
                "Market Sentiment",
                sentiment['sentiment'],
                f"Score: {sentiment['score']:.2f}"
            )
            
        with col3:
            sm_sentiment = analysis['social_media_sentiment']
            st.metric(
                "Social Sentiment",
                _get_sentiment_label(sm_sentiment),
                f"Score: {sm_sentiment:.2f}"
            )
        
        # Market Health Indicators
        if 'market_health' in analysis:
            st.subheader("Market Health Indicators")
            health_data = analysis['market_health']
            
            health_metrics = {
                "Trend Strength": ("trend_strength", "Strength of current trend"),
                "Market Depth": ("market_depth", "Order book depth and liquidity"),
                "Price Stability": ("price_stability", "Price stability measure")
            }
            
            cols = st.columns(len(health_metrics))
            for col, (label, (key, help_text)) in zip(cols, health_metrics.items()):
                with col:
                    st.metric(
                        label,
                        f"{health_data[key]:.2f}",
                        help=help_text
                    )

def display_risk_metrics(analysis: Dict):
    """Display detailed risk metrics"""
    st.subheader("Risk Analysis")
    
    # Get risk assessment data with defaults
    risk = analysis.get('risk_assessment', {})
    volatility_data = analysis.get('volatility', {})
    market_data = analysis.get('market_data', {})
    
    # Calculate overall risk level if not present
    if 'risk_level' not in risk:
        vol_level = volatility_data.get('volatility_level', 'Medium')
        annual_vol = volatility_data.get('annual_volatility', 0)
        
        if annual_vol > 0.5 or vol_level == 'High':
            risk_level = 'High'
        elif annual_vol > 0.3 or vol_level == 'Medium':
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
            
        risk['risk_level'] = risk_level
    
    # Display risk metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Level",
            risk.get('risk_level', 'Unknown'),
            help="Overall risk assessment"
        )
    
    with col2:
        st.metric(
            "Volatility",
            volatility_data.get('volatility_level', 'Unknown'),
            f"{volatility_data.get('annual_volatility', 0)*100:.1f}%",
            help="Market volatility metrics"
        )
    
    with col3:
        st.metric(
            "ATR",
            f"{volatility_data.get('atr', 0):.2f}",
            f"{volatility_data.get('atr_percentage', 0):.1f}%",
            help="Average True Range"
        )
    
    # Display risk factors if available
    if 'risk_factors' in risk:
        st.subheader("Risk Factors")
        risk_factors = risk['risk_factors']
        
        # Create a bar chart for risk factors
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(risk_factors.values()),
            y=list(risk_factors.keys()),
            orientation='h',
            marker_color=['red' if v > 0.6 else 'orange' if v > 0.3 else 'blue' 
                         for v in risk_factors.values()]
        ))
        
        fig.update_layout(
            title="Risk Factor Analysis",
            xaxis_title="Risk Score",
            yaxis_title="Factor",
            height=300,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Display market risk metrics
    st.subheader("Market Risk Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bollinger Band Analysis
        bb_width = volatility_data.get('bollinger_width', 0)
        st.metric(
            "Bollinger Width",
            f"{bb_width*100:.1f}%",
            help="Wider bands indicate higher volatility"
        )
        
        # Volume Analysis
        volume_analysis = analysis.get('volume_analysis', {})
        st.metric(
            "Volume Intensity",
            volume_analysis.get('volume_intensity', 'Normal'),
            f"{volume_analysis.get('volume_change', 0):.1f}%",
            help="Volume intensity and change"
        )
    
    with col2:
        # Market Depth Analysis
        st.metric(
            "Spread",
            f"{market_data.get('bid_ask_spread', 0)*100:.3f}%",
            help="Bid-Ask spread indicates liquidity risk"
        )
        
        # Price Channel Analysis
        if 'channel_width' in volatility_data:
            st.metric(
                "Price Channel Width",
                f"{volatility_data['channel_width']:.1f}%",
                help="Width of price channel as percentage"
            )
    
    # Display risk alerts if available
    if 'risk_alerts' in risk:
        st.subheader("Risk Alerts")
        
        for alert in risk['risk_alerts']:
            severity = alert.get('severity', 'normal')
            color = {
                'high': 'red',
                'medium': 'orange',
                'low': 'blue'
            }.get(severity, 'white')
            
            st.markdown(
                f"<span style='color: {color}'>•</span> {alert['message']}",
                unsafe_allow_html=True
            )

def _get_fear_greed_label(value: float) -> str:
    """Get descriptive label for fear & greed index"""
    if value >= 80:
        return "Extreme Greed"
    elif value >= 60:
        return "Greed"
    elif value >= 40:
        return "Neutral"
    elif value >= 20:
        return "Fear"
    else:
        return "Extreme Fear"

def _get_sentiment_label(score: float) -> str:
    """Get descriptive label for sentiment score"""
    if score >= 0.7:
        return "Very Positive"
    elif score >= 0.3:
        return "Positive"
    elif score >= -0.3:
        return "Neutral"
    elif score >= -0.7:
        return "Negative"
    else:
        return "Very Negative"

def display_strategy_details(analysis: Dict):
    """Display detailed trading strategy information"""
    st.subheader("Trading Strategy Details")
    
    if 'strategy' in analysis:
        strategy = analysis['strategy']
        
        # Entry Strategy
        st.write("**Entry Strategy**")
        entry = strategy['entry']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Method: {entry['method']}")
            st.write("Price Levels:")
            for i, (price, alloc) in enumerate(zip(
                entry['points']['price_levels'],
                entry['points']['allocation']
            )):
                st.write(f"• Entry {i+1}: ${price:.2f} ({alloc*100:.0f}%)")
                
        with col2:
            st.write("Entry Conditions:")
            for condition in entry['conditions']:
                st.write(f"• {condition}")
        
        # Exit Strategy
        st.write("**Exit Strategy**")
        exit_strategy = strategy['exit']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Take Profit Levels:")
            for i, (price, alloc) in enumerate(zip(
                exit_strategy['take_profit']['levels'],
                exit_strategy['take_profit']['allocations']
            )):
                st.write(f"• TP {i+1}: ${price:.2f} ({alloc*100:.0f}%)")
                
        with col2:
            st.write("Stop Loss Strategy:")
            st.write(f"• Main Stop: ${exit_strategy['stop_loss']['main_stop']:.2f}")
            st.write(f"• Secondary: ${exit_strategy['stop_loss']['secondary_stop']:.2f}")
            st.write(f"• Breakeven: {exit_strategy['stop_loss']['breakeven_trigger']}")
        
        # Risk Management
        st.write("**Risk Management**")
        risk_mgmt = strategy['risk_management']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Position Limits:")
            st.write(f"• Maximum: {risk_mgmt['position_limits']['max_size']*100:.1f}%")
            st.write(f"• Recommended: {risk_mgmt['position_limits']['recommended_size']*100:.1f}%")
            
        with col2:
            st.write("Monitoring Rules:")
            for rule in risk_mgmt['monitoring']:
                st.write(f"• {rule}")
        
        # Additional Strategy Information
        if 'timeframe' in strategy:
            st.write(f"**Recommended Timeframe:** {strategy['timeframe']}")
            
        if 'adjustments' in strategy:
            st.write("**Adjustment Rules:**")
            for rule in strategy['adjustments']:
                st.write(f"• {rule}")

def create_strategy_chart(price_data: pd.DataFrame, strategy: Dict, indicators: Dict) -> go.Figure:
    """Create an interactive strategy visualization chart"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price & Strategy Levels', 'Volume Analysis')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['open'],
            high=price_data['high'],
            low=price_data['low'],
            close=price_data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    ma_colors = {'20': 'blue', '50': 'orange', '200': 'red'}
    for period, color in ma_colors.items():
        if f'sma_{period}' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=indicators[f'sma_{period}'],
                    name=f'SMA {period}',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # Add strategy levels
    entry_points = strategy['entry']['points']
    for i, (price, alloc) in enumerate(zip(
        entry_points['price_levels'],
        entry_points['allocation']
    )):
        fig.add_hline(
            y=price,
            line_color='blue',
            line_dash='dash',
            line_width=1,
            opacity=0.7,
            annotation=dict(
                text=f"Entry {i+1} ({alloc*100:.0f}%)",
                xanchor='left',
                yanchor='bottom'
            ),
            row=1, col=1
        )
    
    # Add take profit levels
    tp_levels = strategy['exit']['take_profit']
    for i, (price, alloc) in enumerate(zip(
        tp_levels['levels'],
        tp_levels['allocations']
    )):
        fig.add_hline(
            y=price,
            line_color='green',
            line_dash='dash',
            line_width=1,
            opacity=0.7,
            annotation=dict(
                text=f"TP {i+1} ({alloc*100:.0f}%)",
                xanchor='left',
                yanchor='bottom'
            ),
            row=1, col=1
        )
    
    # Add stop loss levels
    sl_data = strategy['exit']['stop_loss']
    fig.add_hline(
        y=sl_data['main_stop'],
        line_color='red',
        line_dash='dash',
        line_width=1,
        opacity=0.7,
        annotation=dict(
            text="Main Stop",
            xanchor='left',
            yanchor='top'
        ),
        row=1, col=1
    )
    
    fig.add_hline(
        y=sl_data['secondary_stop'],
        line_color='red',
        line_dash='dot',
        line_width=1,
        opacity=0.5,
        annotation=dict(
            text="Secondary Stop",
            xanchor='left',
            yanchor='top'
        ),
        row=1, col=1
    )
    
    # Add volume bars with color coding
    colors = ['red' if close < open else 'green' 
             for open, close in zip(price_data['open'], price_data['close'])]
    
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.8
        ),
        row=2, col=1
    )
    
    # Add volume MA
    volume_ma = pd.Series(price_data['volume']).rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=volume_ma,
            name='Volume MA(20)',
            line=dict(color='blue', width=1),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        title=dict(
            text="Strategy Visualization",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_strategy_dashboard(analysis: Dict):
    """Create an interactive strategy dashboard"""
    if 'strategy' not in analysis:
        st.warning("No strategy information available")
        return

    strategy = analysis['strategy']
    
    # Display Strategy Recommendation
    st.markdown(f"""
    ### Recommended Action: {strategy.get('action', 'Hold')}
    **Entry Strategy**: {strategy.get('entry', {}).get('method', 'Single')}
    """)

    # Display Entry Points in a dedicated section
    st.subheader("Entry Points")
    entry_points = strategy.get('entry', {}).get('points', {})
    price_levels = entry_points.get('price_levels', [])
    allocations = entry_points.get('allocation', [])
    
    if price_levels and allocations:
        for i, (price, alloc) in enumerate(zip(price_levels, allocations)):
            if price is not None:
                st.markdown(f"• Entry {i+1}: ${price:,.2f} ({alloc*100:.0f}%)")
    else:
        st.markdown("• Waiting for entry opportunity")

    # Display Exit Points in a dedicated section
    st.subheader("Exit Points")
    exit_data = strategy.get('exit', {})
    
    # Take Profit
    tp_data = exit_data.get('take_profit', {})
    tp_levels = tp_data.get('levels', [])
    tp_allocations = tp_data.get('allocations', [])
    
    if tp_levels and tp_allocations:
        for i, (level, alloc) in enumerate(zip(tp_levels, tp_allocations)):
            if level is not None:
                st.markdown(f"• TP {i+1}: ${level:,.2f} ({alloc*100:.0f}%)")
    
    # Stop Loss
    sl_price = exit_data.get('stop_loss', {}).get('main_stop')
    if sl_price:
        st.markdown(f"• Stop Loss: ${sl_price:,.2f}")

    # Calculate and display Risk/Reward Ratio
    st.subheader("Risk/Reward Ratio")
    try:
        if price_levels and tp_levels and sl_price:
            entry_price = price_levels[0]
            tp_level = tp_levels[0]
            
            if all([entry_price, tp_level, sl_price]):
                if strategy.get('action') == "Buy":
                    reward = tp_level - entry_price
                    risk = entry_price - sl_price
                else:  # Sell
                    reward = entry_price - tp_level
                    risk = sl_price - entry_price
                    
                if abs(risk) > 0:
                    rr_ratio = abs(reward / risk)
                    st.markdown(f"**{rr_ratio:.2f}**")
                else:
                    st.markdown("N/A (Invalid risk value)")
            else:
                st.markdown("N/A (Missing price data)")
        else:
            st.markdown("N/A (No active position)")
    except Exception as e:
        logger.error(f"Error calculating R/R ratio: {str(e)}")
        st.markdown("N/A (Calculation error)")

def _calculate_rr_ratio(strategy: Dict) -> Optional[float]:
    """Calculate risk/reward ratio based on strategy levels"""
    try:
        # Extract entry price
        entry_points = strategy.get('entry', {}).get('points', {})
        entry_price = float(entry_points.get('price_levels', [0])[0])
        
        # Extract exit prices
        exit_data = strategy.get('exit', {})
        tp_data = exit_data.get('take_profit', {})
        tp_price = float(tp_data.get('levels', [0])[0])
        
        sl_data = exit_data.get('stop_loss', {})
        sl_price = float(sl_data.get('main_stop', 0))
        
        # Validate prices
        if not all([entry_price, tp_price, sl_price]):
            return None
            
        # Calculate ratio based on action
        action = strategy.get('action', 'Hold')
        if action == "Buy":
            reward = tp_price - entry_price
            risk = entry_price - sl_price
        else:  # Sell
            reward = entry_price - tp_price
            risk = sl_price - entry_price
            
        # Avoid division by zero
        if abs(risk) < 1e-8:
            return None
            
        return abs(reward / risk)
        
    except (TypeError, ValueError, IndexError, ZeroDivisionError) as e:
        logger.error(f"Error calculating R/R ratio: {str(e)}")
        return None

def _calculate_potential_profit(strategy: Dict, position_size: float) -> Optional[float]:
    """Calculate potential profit based on strategy and position size"""
    try:
        # Get entry points
        entry_points = strategy.get('entry', {}).get('points', {})
        if not entry_points:
            return None
            
        levels = entry_points.get('levels', [])
        allocations = entry_points.get('allocations', [])
        
        if not levels or not allocations:
            return None
            
        # Calculate average entry
        entry_avg = sum(p * a for p, a in zip(levels, allocations))
        
        # Get take profit levels
        exit_points = strategy.get('exit', {})
        tp_levels = exit_points.get('take_profit', {}).get('levels', [])
        tp_allocations = exit_points.get('take_profit', {}).get('allocations', [])
        
        if not tp_levels or not tp_allocations:
            return None
            
        # Calculate average take profit
        tp_avg = sum(p * a for p, a in zip(tp_levels, tp_allocations))
        
        # Calculate profit percentage based on action
        action = strategy.get('action', 'Hold')
        if action == "Buy":
            profit_pct = (tp_avg - entry_avg) / entry_avg
        else:  # Sell
            profit_pct = (entry_avg - tp_avg) / entry_avg
        
        return profit_pct * position_size * 100
        
    except Exception as e:
        logger.error(f"Error calculating potential profit: {e}")
        return None

def _calculate_max_drawdown(strategy: Dict, position_size: float) -> Optional[float]:
    """Calculate maximum drawdown based on strategy and position size"""
    try:
        # Get entry points
        entry_points = strategy.get('entry', {}).get('points', {})
        if not entry_points:
            return None
            
        levels = entry_points.get('levels', [])
        allocations = entry_points.get('allocations', [])
        
        if not levels or not allocations:
            return None
            
        # Calculate average entry
        entry_avg = sum(p * a for p, a in zip(levels, allocations))
        
        # Get stop loss
        stop_loss = strategy.get('exit', {}).get('stop_loss', {}).get('main_stop')
        if not stop_loss:
            return None
        
        # Calculate drawdown percentage based on action
        action = strategy.get('action', 'Hold')
        if action == "Buy":
            drawdown_pct = (entry_avg - stop_loss) / entry_avg
        else:  # Sell
            drawdown_pct = (stop_loss - entry_avg) / entry_avg
        
        return abs(drawdown_pct * position_size * 100)
        
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return None

def create_strategy_metrics_dashboard(strategy: Dict, risk_assessment: Dict) -> None:
    """Create an interactive metrics dashboard for the strategy"""
    st.subheader("Strategy Metrics Dashboard")
    
    # Position Size Controls
    col1, col2 = st.columns(2)
    
    with col1:
        position_size = st.slider(
            "Position Size (%)",
            min_value=0,
            max_value=100,
            value=int(strategy['position_size']['suggested_size'] * 100),
            step=5
        )
        
    with col2:
        leverage = st.selectbox(
            "Leverage",
            [1, 2, 3, 5, 10],
            index=0,
            help="Select leverage multiplier"
        )
    
    # Risk Metrics
    st.write("**Risk Metrics**")
    col1, col2, col3, col4 = st.columns(4)
    
    rr_ratio = _calculate_rr_ratio(strategy)
    potential_profit = _calculate_potential_profit(strategy, position_size/100) * leverage
    max_drawdown = _calculate_max_drawdown(strategy, position_size/100) * leverage
    
    with col1:
        st.metric(
            "Risk/Reward Ratio",
            f"{rr_ratio:.2f}",
            help="Ratio of potential profit to risk"
        )
    
    with col2:
        st.metric(
            "Potential Profit",
            f"{potential_profit:.1f}%",
            help="Maximum potential profit based on take profit levels"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{max_drawdown:.1f}%",
            help="Maximum potential loss based on stop loss"
        )
    
    with col4:
        st.metric(
            "Risk Level",
            risk_assessment['risk_level'],
            help="Overall risk assessment"
        )
    
    # Strategy Statistics
    st.write("**Strategy Statistics**")
    stats = _calculate_strategy_statistics(strategy)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Entry Points:")
        for i, (price, alloc) in enumerate(zip(
            strategy['entry']['points']['price_levels'],
            strategy['entry']['points']['allocation']
        )):
            st.write(f"• Entry {i+1}: ${price:.2f} ({alloc*100:.0f}%)")
    
    with col2:
        st.write("Exit Points:")
        for i, (price, alloc) in enumerate(zip(
            strategy['exit']['take_profit']['levels'],
            strategy['exit']['take_profit']['allocations']
        )):
            st.write(f"• TP {i+1}: ${price:.2f} ({alloc*100:.0f}%)")
        st.write(f"• Stop Loss: ${strategy['exit']['stop_loss']['main_stop']:.2f}")

def _calculate_strategy_statistics(strategy: Dict) -> Dict:
    """Calculate additional strategy statistics"""
    entry_points = strategy['entry']['points']['price_levels']
    tp_points = strategy['exit']['take_profit']['levels']
    sl_point = strategy['exit']['stop_loss']['main_stop']
    
    return {
        'avg_entry': np.mean(entry_points),
        'entry_range': max(entry_points) - min(entry_points),
        'avg_tp': np.mean(tp_points),
        'tp_range': max(tp_points) - min(tp_points),
        'sl_distance': abs(np.mean(entry_points) - sl_point)
    }

def display_sentiment_analysis(analysis: Dict):
    """Display sentiment analysis results"""
    st.subheader("Sentiment Analysis")
    
    # Internal Market Sentiment
    st.markdown("### Internal Sentiment")
    
    # Get sentiment data with proper defaults
    sentiment = analysis.get('sentiment', {
        'score': 0,
        'sentiment': 'Neutral',
        'description': 'Market sentiment based on technical and on-chain metrics'
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Internal Market Sentiment",
            sentiment.get('sentiment', 'Neutral'),
            f"Score: {sentiment.get('score', 0):.2f}"
        )
        
    with col2:
        st.write(sentiment.get('description', ''))
    
    # External Sentiment Factors
    st.markdown("### External Sentiment Factors")
    
    # Fear & Greed Index
    fear_greed = analysis.get('fear_greed_index', 50)
    label = _get_fear_greed_label(fear_greed)
    
    st.metric(
        "Fear & Greed Index",
        fear_greed,
        label
    )
    st.write("CMC Fear & Greed Index Explanation: Lower values = fear, higher = greed.")
    
    # Social Media Sentiment
    social_sentiment = analysis.get('social_media_sentiment', 0)
    label = _get_sentiment_label(social_sentiment)
    
    st.metric(
        "Social Media Sentiment",
        f"{social_sentiment:.2f}",
        label
    )
    st.write("Social Media Sentiment: Aggregated user sentiment from various platforms.")

def main():
    """Main application entry point"""
    
    # Initialize session state if not present
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    try:
        # Page configuration MUST be the first Streamlit command
        st.set_page_config(
            page_title="Crypto Analysis Assistant",
            layout="wide"
        )
    except Exception:
        pass  # Ignore if already configured

    # Hide all default Streamlit elements and manage app button completely
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {padding-top: 0rem;}
            div[data-testid="manage-app-button"] {display: none !important;}
            section[data-testid="stSidebar"] > div:last-child {display: none !important;}
        </style>
    """, unsafe_allow_html=True)
    
    # Get cookie manager instance
    cookie_manager = get_manager()

    # Check token and update session state
    token = cookie_manager.get("auth_token")
    if token and verify_jwt_token(token):
        st.session_state.authenticated = True
    else:
        st.session_state.authenticated = False
    
    # Check authentication
    if not check_authentication():
        login_page()
        return
        
    # Add custom CSS for full-width layout
    st.markdown("""
        <style>
            .block-container {
                max-width: 100% !important;
                width: 100% !important;
                padding: 2rem 5rem !important;
            }
            .element-container, .stPlotlyChart {
                width: 100% !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'price_data' not in st.session_state:
        st.session_state.price_data = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = None
    
    # Initialize configuration
    config = Config()
    
    # App Header
    st.title("🚀 Crypto Analysis Assistant")
    st.markdown("""
    Analyze cryptocurrency markets using technical analysis, market data, and AI insights.
    """)
    
    # Sidebar Header with full-width logo and title below
    st.sidebar.image("logo-v3.png", use_container_width=True)  # Logo takes full width
    st.sidebar.markdown("<h3 style='text-align: center; margin-top: 0;'>Crypto Assistant</h3>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")  # Divider line
        
    # Sidebar Configuration
    st.sidebar.title("Configuration")
    
    # Exchange API Status
    st.sidebar.subheader("Exchange API Configuration")
    show_api_config = st.sidebar.expander("Exchange API Status")
    
    with show_api_config:
        st.markdown("""
        💡 **Note**: API keys are loaded from environment variables.
        Configure them in your .env file for better security.
        """)
        
        # Show API key status for each exchange
        st.subheader("Binance")
        if config.binance_api_key and config.binance_secret:
            st.success("API keys configured")
        else:
            st.info("No API keys found in environment")
        
        st.subheader("KuCoin")
        if config.kucoin_api_key and config.kucoin_secret:
            st.success("API keys configured")
        else:
            st.info("No API keys found in environment")
        
        st.subheader("Huobi")
        if config.huobi_api_key and config.huobi_secret:
            st.success("API keys configured")
        else:
            st.info("No API keys found in environment")
   
    # Analysis Settings
    st.sidebar.subheader("Analysis Settings")
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        options=['1m', '5m', '15m', '1h', '4h', '1d', '1w'],
        index=5,  # Default to '1d'
        help="Select the timeframe for analysis"
    )
    
    lookback = st.sidebar.slider(
        "Number of Candles",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Select the number of historical candles to analyze"
    )
    
    # Technical Indicators Selection
    show_indicators = st.sidebar.expander("Technical Indicators")
    with show_indicators:
        selected_indicators = []
        if st.checkbox("Moving Averages", value=True):
            selected_indicators.extend(['SMA', 'EMA'])
        if st.checkbox("Momentum Indicators", value=True):
            selected_indicators.extend(['RSI', 'MACD'])
        if st.checkbox("Volatility Indicators", value=True):
            selected_indicators.extend(['BB', 'ATR'])
        
        config.technical_indicators = selected_indicators
    
    # Main content area - Trading Symbol Input
    st.subheader("Enter Trading Symbol")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter trading symbol (e.g., BTC/USDT):",
            help="Enter the trading pair exactly as shown on your preferred exchange"
        ).strip()
    
    with col2:
        exchange = st.selectbox(
            "Select Exchange",
            ["Binance", "KuCoin", "Huobi"],
            help="Choose the exchange to fetch data from"
        )
    
    # Example pairs
    st.markdown("""
    **Example pairs**: 
    - BTC/USDT
    - ETH/USDT 
    - SOL/USDT
    - BNB/USDT
    """)
    
    # Analysis Button
    if st.button("Analyze", type="primary"):
        if not symbol:
            st.error("Please enter a trading symbol")
            return
        
        try:
            # Initialize components
            data_fetcher = DataFetcher(config)
            market_analyzer = MarketAnalyzer(config)
            chart_builder = ChartBuilder()
            
            # Fetch and analyze data
            with st.spinner(f"Fetching market data for {symbol}..."):
                # Fetch price data
                price_data = data_fetcher.fetch_price_data(
                    symbol,
                    timeframe=timeframe,
                    limit=lookback
                )
                
                if price_data.empty:
                    st.error(f"No data available for {symbol}")
                    return
                
                # Perform analysis
                analysis = market_analyzer.analyze_price(price_data)
                
                # Generate AI analysis
                ai_team = AIAnalystTeam()
                ai_analysis = ai_team.generate_analysis(analysis)
                analysis.update(ai_analysis)

                # Store results in session state
                st.session_state.analysis_results = analysis
                st.session_state.price_data = price_data
                st.session_state.current_symbol = symbol
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Price Chart",
                    "Technical Analysis",
                    "Market Structure",
                    "AI Analysis",
                    "Strategy"
                ])
                
                with tab1:
                    st.subheader("Price Action & Volume")
                    price_chart = chart_builder.create_price_chart(price_data, analysis)
                    st.plotly_chart(price_chart, use_container_width=True)
                
                with tab2:
                    display_technical_analysis(analysis)
                    indicators_chart = chart_builder.create_indicators_chart(price_data, analysis)
                    st.plotly_chart(indicators_chart, use_container_width=True)
                
                with tab3:
                    display_market_structure(analysis)
                    display_sentiment_analysis(analysis)
                
                with tab4:
                    display_ai_analysis(analysis)
                    display_risk_metrics(analysis)
                
                with tab5:
                    create_strategy_dashboard(analysis)
        
        except DataFetchError as e:
            st.error(f"Failed to fetch data: {str(e)}")
            logger.error(f"Data fetch error: {e}", exc_info=True)
        except ValidationError as e:
            st.error(f"Data validation error: {str(e)}")
            logger.error(f"Validation error: {e}", exc_info=True)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            logger.error(f"Analysis error: {e}", exc_info=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Crypto Analysis Assistant v1.0</p>
            <p>⚠️ This is not financial advice. Always do your own research prior investment.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
