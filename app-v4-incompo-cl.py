# Enhanced Crypto Analysis Application

import os
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import datetime
import time
from datetime import timezone
import numpy as np
import pandas as pd
import requests
import ccxt
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import jwt
import bcrypt
import streamlit as st
import extra_streamlit_components as stx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import talib
import warnings
from web3 import Web3
from eth_typing import BlockNumber
import aiohttp
import asyncio
from web3 import Web3
from eth_typing import BlockNumber
from typing import Dict, List, Optional, Set, Tuple, Any
import aiohttp
import joblib
from pathlib import Path
from datetime import datetime, timezone
from sklearn.base import BaseEstimator
import websockets
import hmac
import hashlib
import base64
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.orm import declarative_base
import math
import nest_asyncio

nest_asyncio.apply()
Base = declarative_base()

st.set_page_config(
    page_title="Enhanced Crypto Analysis",
    layout="wide"
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Custom exceptions
class CryptoAnalysisError(Exception):
    """Base exception class"""
    pass

class APIError(CryptoAnalysisError):
    """API related errors"""
    pass

class DataError(CryptoAnalysisError):
    """Data processing errors"""
    pass

class ModelError(CryptoAnalysisError):
    """ML model related errors"""
    pass

class DataFetchError(Exception):
    """Error for data fetching failures"""
    pass

class AnalysisError(Exception):
    """Error for analysis failures"""
    pass

@dataclass
class APIConfig:
    """Enhanced API Configuration"""
    # Premium APIs
    cryptocompare_key: Optional[str] = None
    messari_key: Optional[str] = None
    santiment_key: Optional[str] = None
    glassnode_key: Optional[str] = None
    
    # Exchange APIs
    binance_key: Optional[str] = None
    binance_secret: Optional[str] = None
    ftx_key: Optional[str] = None
    ftx_secret: Optional[str] = None
    
    # API States
    using_premium: bool = False
    active_apis: Dict[str, bool] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        # Load API keys
        self._load_api_keys()
        self._initialize_api_states()

        print("Loading API keys...")
        print(f"Binance key present: {bool(self.binance_key)}")
        print(f"Messari key present: {bool(self.messari_key)}")
        print(f"Santiment key present: {bool(self.santiment_key)}")

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        self.cryptocompare_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        self.messari_key = os.getenv('MESSARI_API_KEY')
        self.santiment_key = os.getenv('SANTIMENT_API_KEY')
        self.glassnode_key = os.getenv('GLASSNODE_API_KEY')
        
        self.binance_key = os.getenv('BINANCE_API_KEY')
        self.binance_secret = os.getenv('BINANCE_SECRET')
        self.huobi_key = os.getenv('HUOBI_API_KEY')
        self.huobi_secret = os.getenv('HUOBI_SECRET')

        # Set premium status
        self.using_premium = any([
            self.cryptocompare_key,
            self.messari_key,
            self.santiment_key,
            self.glassnode_key
        ])
        
    def _initialize_api_states(self):
        """Initialize API availability and rate limits"""
        self.active_apis = {
            # Free APIs
            'coingecko': True,
            'binance_public': True,
            'alternative_me': True,
            
            # Premium APIs
            'cryptocompare': bool(self.cryptocompare_key),
            'messari': bool(self.messari_key),
            'santiment': bool(self.santiment_key),
            'glassnode': bool(self.glassnode_key)
        }
        
        self.rate_limits = {
            'coingecko': 50,  # per minute
            'binance_public': 1200,  # per minute
            'alternative_me': 1,  # per minute
            'cryptocompare': 100000 if self.cryptocompare_key else 0,
            'messari': 30 if self.messari_key else 0,
            'santiment': 100 if self.santiment_key else 0,
            'glassnode': 100 if self.glassnode_key else 0
        }
    
    def get_best_api_for(self, data_type: str) -> str:
        """Get best available API for specific data type"""
        api_priority = {
            'market_data': [
                ('glassnode', self.glassnode_key),
                ('messari', self.messari_key),
                ('cryptocompare', self.cryptocompare_key),
                ('coingecko', True)
            ],
            'sentiment': [
                ('santiment', self.santiment_key),
                ('cryptocompare', self.cryptocompare_key),
                ('alternative_me', True)
            ],
            'onchain': [
                ('glassnode', self.glassnode_key),
                ('santiment', self.santiment_key),
                ('messari', self.messari_key)
            ],
            'news': [
                ('cryptocompare', self.cryptocompare_key),
                ('messari', self.messari_key)
            ]
        }
        
        # Return first available API in priority list
        for api, key in api_priority.get(data_type, []):
            if self.active_apis[api] and key:
                return api
        
        # Return default free API if no premium available
        return {
            'market_data': 'coingecko',
            'sentiment': 'alternative_me',
            'onchain': None,
            'news': None
        }.get(data_type)

@dataclass
class Config:
    """Enhanced application configuration"""
    def __init__(self):
        self.min_whale_size = 1000000
        self.analysis_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.max_lookback_days = 365
        self.min_data_points = 30
        self.model_save_path = "models"
        self.training_data_days = 365
        self.min_training_samples = 1000
        self.model_update_interval = 24
        self.sequence_length = 60
        self.train_test_split = 0.8
        self.max_position_size = 0.1
        self.risk_free_rate = 0.02
        self.cache_duration = 300
        self.db_url = os.getenv('DATABASE_URL', 'sqlite:///crypto_assistant.db')
        
        self._initialize_technical_indicators()
        self._initialize_agent_weights()
        self._initialize_paths()
        self._load_env_config()

        self.secondary_data_config = {
            'cache_duration': 300,
            'fallback_order': ['cryptocompare', 'messari', 'glassnode', 'santiment'],
            'retry_attempts': 3,
            'retry_delay': 1,
            'timeout': 30
        }
        
    def __post_init__(self):
        self._initialize_technical_indicators()
        self._initialize_agent_weights()
        self._initialize_paths()
        self._load_env_config()

        self.secondary_data_config = {
            'cache_duration': 300,  # 5 minutes
            'fallback_order': ['cryptocompare', 'messari', 'glassnode', 'santiment'],
            'retry_attempts': 3,
            'retry_delay': 1,  # seconds
            'timeout': 30  # seconds
            }

    def _initialize_paths(self):
        """Initialize required paths"""
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def _load_env_config(self):
        """Load configuration from environment"""
        load_dotenv()
        
        # Override defaults with environment variables
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', self.max_position_size))
        self.risk_free_rate = float(os.getenv('RISK_FREE_RATE', self.risk_free_rate))
        self.cache_duration = int(os.getenv('CACHE_DURATION', self.cache_duration))
        self.model_save_path = os.getenv('MODEL_SAVE_PATH', self.model_save_path)

    def _initialize_technical_indicators(self):
        """Initialize technical indicator configurations"""
        self.technical_indicators = {
            'MA': {
                'periods': [20, 50, 200],
                'types': ['SMA', 'EMA']
            },
            'RSI': {
                'period': 14,
                'overbought': 70,
                'oversold': 30
            },
            'MACD': {
                'fast_period': 12,
                'slow_period': 26,
                'signal_period': 9
            },
            'BB': {
                'period': 20,
                'std_dev': 2
            }
        }
    
    def _initialize_agent_weights(self):
        """Initialize AI agent weights"""
        self.agent_weights = {
            'TechnicalAgent': 0.3,
            'SentimentAgent': 0.2,
            'On_chain': 0.2,
            'MarketRegimeAgent': 0.15,
            'whale': 0.15
        }
    
    def __post_init__(self):
        # Existing post init code...
        os.makedirs(self.model_save_path, exist_ok=True)

class DatabaseManager:
    """Manages database operations and session state"""
    
    def __init__(self, config: Config):
        self.config = config
        self.engine = create_engine(config.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema"""
        Base.metadata.create_all(self.engine)
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.Session()
    
    def save_model_state(self, model_state: Dict):
        """Save model state to database"""
        with self.get_session() as session:
            state = ModelState(
                timestamp=datetime.datetime.now(timezone.utc),
                state=json.dumps(model_state)
            )
            session.add(state)
            session.commit()
    
    def get_latest_model_state(self) -> Optional[Dict]:
        """Get latest model state"""
        with self.get_session() as session:
            state = session.query(ModelState).order_by(
                ModelState.timestamp.desc()
            ).first()
            
            if state:
                return json.loads(state.state)
            return None
    
    def save_prediction(self, prediction: Dict):
        """Save prediction to database"""
        with self.get_session() as session:
            pred = Prediction(
                timestamp=datetime.datetime.now(timezone.utc),
                symbol=prediction['symbol'],
                timeframe=prediction['timeframe'],
                action=prediction['action'],
                confidence=prediction['confidence'],
                price=prediction['price']
            )
            session.add(pred)
            session.commit()
    
    def get_prediction_history(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get prediction history"""
        with self.get_session() as session:
            predictions = session.query(Prediction).filter(
                Prediction.symbol == symbol,
                Prediction.timeframe == timeframe
            ).order_by(
                Prediction.timestamp.desc()
            ).limit(limit).all()
            
            return [p.to_dict() for p in predictions]

class SecondaryDataSource:
    """Implements secondary data source fetching"""  
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.session = aiohttp.ClientSession()  # Add session initialization
        self.coingecko_session = self._init_coingecko()
        self.rate_limiter = TokenBucketLimiter(rate_limit=50, per_second=60)
    
    def _init_coingecko(self) -> requests.Session:
        """Initialize CoinGecko session"""
        session = requests.Session()
        session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })
        return session
        
    async def fetch_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            # Try CoinGecko
            coingecko_data = await self._fetch_from_coingecko(symbol, timeframe)
            if coingecko_data is not None:
                return coingecko_data

            # Try CryptoCompare
            cryptocompare_data = await self._fetch_from_cryptocompare(symbol, timeframe)
            if cryptocompare_data is not None:
                return cryptocompare_data

            return None
        except Exception as e:
            logger.error(f"Secondary data fetch error: {str(e)}")
            return None

    async def _fetch_from_coingecko(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            coin_id = await self._get_coingecko_id(symbol.split('/')[0].lower())
            if not coin_id:
                return None

            async with self.session.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                params={
                    'vs_currency': 'usdt',
                    'days': self._timeframe_to_days(timeframe),
                    'interval': self._timeframe_to_interval(timeframe)
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_coingecko_data(data)
                return None
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed: {str(e)}")
            return None

    def _get_coingecko_id(self, symbol: str) -> str:
        """Get CoinGecko ID for symbol"""
        # Cache this data in practice
        try:
            with self.rate_limiter:
                response = self.coingecko_session.get(
                    "https://api.coingecko.com/api/v3/coins/list"
                )
                coins = response.json()
                
            for coin in coins:
                if coin['symbol'] == symbol:
                    return coin['id']
            raise ValueError(f"Symbol {symbol} not found in CoinGecko")
            
        except Exception as e:
            logger.error(f"Error getting CoinGecko ID: {str(e)}")
            raise
    
    def _process_coingecko_data(self, data: Dict) -> pd.DataFrame:
        """Process CoinGecko data into OHLCV format"""
        # Extract price and volume data
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Merge and process
        df = prices.merge(volumes, on='timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Generate OHLCV data
        ohlcv = df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        return ohlcv
    
    def _timeframe_to_days(self, timeframe: str, limit: int) -> int:
        """Convert timeframe and limit to days for API"""
        conversions = {
            '1m': 1/1440,  # 1 minute
            '5m': 5/1440,
            '15m': 15/1440,
            '1h': 1/24,
            '4h': 4/24,
            '1d': 1
        }
        return int(limit * conversions[timeframe]) + 1
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to CoinGecko interval"""
        if timeframe in ['1m', '5m', '15m']:
            return 'minutely'
        elif timeframe in ['1h', '4h']:
            return 'hourly'
        return 'daily'
    
    def _resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Resample data to desired timeframe"""
        return df.resample(self._to_pandas_freq(timeframe)).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
    
    def _to_pandas_freq(self, timeframe: str) -> str:
        """Convert timeframe to pandas frequency"""
        conversions = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        return conversions[timeframe]

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()
        if hasattr(self, 'coingecko_session'):
            self.coingecko_session.close()

class TokenBucketLimiter:
    """Rate limiter using token bucket algorithm"""
    
    def __init__(self, rate_limit: int, per_second: int):
        self.rate_limit = rate_limit
        self.per_second = per_second
        self.tokens = rate_limit
        self.last_update = time.monotonic()
        self.lock = threading.Lock()
    
    def __enter__(self):
        with self.lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            
            # Replenish tokens
            self.tokens = min(
                self.rate_limit,
                self.tokens + time_passed * (self.rate_limit / self.per_second)
            )
            
            # Wait if needed
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) * (self.per_second / self.rate_limit)
                time.sleep(sleep_time)
                self.tokens = 1
            
            # Use token
            self.tokens -= 1
            self.last_update = now
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class DataFetcher:
    """Enhanced data fetcher with secondary sources"""
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.api_config = api_config
        self.exchanges = self._initialize_exchanges()
        self.secondary_manager = SecondaryDataManager(config, api_config)
        self.cache = {}
    
    async def fetch_market_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        try:
            # Try primary exchanges
            primary_data = await self._fetch_from_primary(symbol, timeframe)
            if primary_data is not None and not primary_data.empty:
                logger.info("Data fetched from primary source")
                return primary_data

            # Try secondary sources
            secondary_data = await self._fetch_from_secondary(symbol, timeframe)
            if secondary_data is not None and not secondary_data.empty:
                logger.info("Data fetched from secondary source")
                return secondary_data

            raise DataError(f"No data available for {symbol}")

        except Exception as e:
            logger.error(f"Market data fetch error: {str(e)}")
            raise

    async def _fetch_from_primary(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        for exchange_id, exchange in self.exchanges.items():
            try:
                logger.debug(f"Trying {exchange_id}")
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
                if ohlcv:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"Error fetching from {exchange_id}: {str(e)}")
        return None

    async def _fetch_from_secondary(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            secondary = SecondaryDataSource(self.config, self.api_config)
            return await secondary.fetch_market_data(symbol, timeframe)
        except Exception as e:
            logger.warning(f"Secondary source fetch failed: {str(e)}")
            return None

    async def _fetch_from_exchanges(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch from primary exchanges"""
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Fetch OHLCV data
                ohlcv = await exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe
                )

                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df

            except Exception as e:
                logger.warning(f"Error fetching from {exchange_id}: {str(e)}")
                continue
        return None

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()
        await self.secondary_manager.cleanup()

class MarketDataValidator:
    """Enhanced data validation"""
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """Validate OHLCV data"""
        try:
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            
            # Check DataFrame type
            if not isinstance(df, pd.DataFrame):
                raise DataError("Input must be a pandas DataFrame")
            
            # Check required columns
            if not required_columns.issubset(df.columns):
                raise DataError(f"Missing columns: {required_columns - set(df.columns)}")
            
            # Check for missing values
            if df[list(required_columns)].isnull().any().any():
                raise DataError("Data contains missing values")
            
            # Check data types
            if not all(df[col].dtype.kind in 'fc' for col in required_columns):
                raise DataError("Non-numeric data in price columns")
            
            # Validate price relationships
            if not all([
                (df['high'] >= df['low']).all(),
                (df['high'] >= df['open']).all(),
                (df['high'] >= df['close']).all(),
                (df['low'] <= df['open']).all(),
                (df['low'] <= df['close']).all()
            ]):
                raise DataError("Invalid price relationships")
            
            # Validate index
            if not isinstance(df.index, pd.DatetimeIndex):
                raise DataError("Index must be datetime")
            
            # Check for duplicates
            if df.index.duplicated().any():
                raise DataError("Duplicate timestamps found")
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            raise DataError(f"Validation failed: {str(e)}")

class SecondaryDataManager:
    """Manages secondary data sources with fallback and rate limiting"""
    
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.api_config = api_config
        self.sources = self._initialize_sources()
        self.cache = ExpiringCache(max_age_seconds=300)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_sources(self) -> Dict:
        """Initialize all secondary data sources"""
        return {
            'coingecko': CoinGeckoAPI(),
            'cryptocompare': CryptoCompareAPI(self.api_config.cryptocompare_key),
            'messari': MessariAPI(self.api_config.messari_key),
            'santiment': SantimentAPI(self.api_config.santiment_key),
            'glassnode': GlassNodeAPI(self.api_config.glassnode_key)
        }
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1d',
        sources: List[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get market data from multiple sources with fallback"""
        try:
            # Use specified sources or all available
            sources = sources or list(self.sources.keys())
            
            # Try each source in order
            for source_name in sources:
                try:
                    if not self.api_config.active_apis.get(source_name):
                        continue
                        
                    source = self.sources[source_name]
                    data = await source.get_market_data(symbol, timeframe)
                    
                    if data is not None and not data.empty:
                        self.logger.info(f"Got market data from {source_name}")
                        return data
                        
                except Exception as e:
                    self.logger.warning(f"Error fetching from {source_name}: {str(e)}")
                    continue
            
            raise DataError("Could not fetch market data from any source")
            
        except Exception as e:
            self.logger.error(f"Market data fetch failed: {str(e)}")
            return None

    async def cleanup(self):
        if hasattr(self, 'data_fetcher'):
            await self.data_fetcher.cleanup()

class CryptoCompareAPI:
    """Enhanced CryptoCompare API integration"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = self._init_session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=100000 if api_key else 50,
            per_second=86400  # Daily limit
        )
    
    def _init_session(self) -> aiohttp.ClientSession:
        """Initialize API session"""
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Apikey {self.api_key}' if self.api_key else None
        }
        return aiohttp.ClientSession(headers={k: v for k, v in headers.items() if v})
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Get market data from CryptoCompare"""
        try:
            # Convert symbol
            base, quote = symbol.split('/')
            
            # Convert timeframe
            limit = self._calculate_limit(timeframe)
            cc_timeframe = self._convert_timeframe(timeframe)
            
            # Fetch data
            async with self.rate_limiter:
                url = f"https://min-api.cryptocompare.com/data/v2/histo{cc_timeframe}"
                params = {
                    'fsym': base.upper(),
                    'tsym': quote.upper(),
                    'limit': limit
                }
                
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    if data['Response'] != 'Success':
                        raise APIError(f"CryptoCompare error: {data['Message']}")
                    
                    return self._process_response(data['Data']['Data'])
                    
        except Exception as e:
            logger.error(f"CryptoCompare fetch failed: {str(e)}")
            return None
    
    def _process_response(self, data: List[Dict]) -> pd.DataFrame:
        """Process CryptoCompare response into DataFrame"""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
        
        # Rename columns to standard format
        df.rename(columns={
            'volumefrom': 'volume',
            'volumeto': 'quote_volume'
        }, inplace=True)
        
        return df
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to CryptoCompare format"""
        conversions = {
            '1m': 'minute',
            '5m': 'minute',
            '15m': 'minute',
            '30m': 'minute',
            '1h': 'hour',
            '2h': 'hour',
            '4h': 'hour',
            '1d': 'day'
        }
        return conversions.get(timeframe, 'day')
    
    def _calculate_limit(self, timeframe: str) -> int:
        """Calculate appropriate limit for timeframe"""
        limits = {
            '1m': 2000,
            '5m': 2000,
            '15m': 2000,
            '30m': 2000,
            '1h': 2000,
            '2h': 1000,
            '4h': 1000,
            '1d': 2000
        }
        return limits.get(timeframe, 2000)

class MessariAPI:
    """Messari API integration"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = self._init_session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=30 if api_key else 20,
            per_second=60
        )
        self.asset_cache = {}
    
    def _init_session(self) -> aiohttp.ClientSession:
        """Initialize API session"""
        headers = {
            'Accept': 'application/json',
            'x-messari-api-key': self.api_key if self.api_key else None
        }
        return aiohttp.ClientSession(headers={k: v for k, v in headers.items() if v})
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Get market data from Messari"""
        try:
            # Get asset ID
            asset_id = await self._get_asset_id(symbol)
            if not asset_id:
                return None
            
            # Convert timeframe
            interval = self._convert_timeframe(timeframe)
            
            # Fetch data
            async with self.rate_limiter:
                url = f"https://data.messari.io/api/v1/assets/{asset_id}/metrics/price/time-series"
                params = {
                    'interval': interval,
                    'fields': 'timestamp,open,high,low,close,volume'
                }
                
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._process_response(data['data'])
                    
        except Exception as e:
            logger.error(f"Messari fetch failed: {str(e)}")
            return None
    
    async def _get_asset_id(self, symbol: str) -> Optional[str]:
        """Get Messari asset ID for symbol"""
        try:
            # Check cache
            if symbol in self.asset_cache:
                return self.asset_cache[symbol]
            
            base = symbol.split('/')[0].lower()
            
            async with self.rate_limiter:
                url = "https://data.messari.io/api/v2/assets"
                params = {'fields': 'id,symbol'}
                
                async with self.session.get(url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    for asset in data['data']:
                        if asset['symbol'].lower() == base:
                            self.asset_cache[symbol] = asset['id']
                            return asset['id']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Messari asset ID: {str(e)}")
            return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Messari format"""
        conversions = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return conversions.get(timeframe, '1d')
    
    def _process_response(self, data: Dict) -> pd.DataFrame:
        """Process Messari response into DataFrame"""
        df = pd.DataFrame(data['values'], columns=data['columns'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

class SantimentAPI:
    """Santiment API integration"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = self._init_session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=100 if api_key else 10,
            per_second=60
        )
    
    def _init_session(self) -> aiohttp.ClientSession:
        """Initialize API session"""
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}' if self.api_key else None
        }
        return aiohttp.ClientSession(headers={k: v for k, v in headers.items() if v})
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Get market and social data from Santiment"""
        try:
            base = symbol.split('/')[0].lower()
            
            async with self.rate_limiter:
                query = """
                query ($slug: String!, $from: DateTime!, $to: DateTime!, $interval: String!) {
                    getMetric(metric: "price_usd") {
                        timeseriesData(
                            slug: $slug,
                            from: $from,
                            to: $to,
                            interval: $interval
                        ) {
                            datetime
                            value
                        }
                    }
                }
                """
                
                variables = {
                    'slug': base,
                    'from': (datetime.now() - timedelta(days=30)).isoformat(),
                    'to': datetime.now().isoformat(),
                    'interval': self._convert_timeframe(timeframe)
                }
                
                async with self.session.post(
                    'https://api.santiment.net/graphql',
                    json={'query': query, 'variables': variables}
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return self._process_response(data)
                    
        except Exception as e:
            logger.error(f"Santiment fetch failed: {str(e)}")
            return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to Santiment format"""
        conversions = {
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return conversions.get(timeframe, '1d')
    
    def _process_response(self, data: Dict) -> pd.DataFrame:
        """Process Santiment response into DataFrame"""
        try:
            timeseries = data['data']['getMetric']['timeseriesData']
            df = pd.DataFrame(timeseries)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            df.rename(columns={'value': 'close'}, inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error processing Santiment response: {str(e)}")
            return None

class GlassNodeAPI:
    """GlassNode API integration"""
    
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.session = self._init_session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=100 if api_key else 0,
            per_second=3600
        )
    
    def _init_session(self) -> aiohttp.ClientSession:
        """Initialize API session with proper headers and configuration"""
        headers = {
            'Accept': 'application/json',
            'X-Api-Key': self.api_key if self.api_key else None,
            'User-Agent': 'CryptoAssistant/1.0',
            'Content-Type': 'application/json'
        }
        
        # Remove None values from headers
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Configure timeout and other session parameters
        timeout = aiohttp.ClientTimeout(total=30)
        
        # Create session with retry capability
        retry_options = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            raise_for_status=True,
            trust_env=True,
            connector=aiohttp.TCPConnector(
                limit=10,  # Connection pool size
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                ssl=False  # Required for some environments
            )
        )
        
        return retry_options 

    async def get_market_data(
        self,
        symbol: str,
        timeframe: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """Get market data from GlassNode"""
        try:
            if not self.api_key:
                return None
                
            base = symbol.split('/')[0].lower()
            
            # Get different metrics
            metrics = [
                'price_usd_close',
                'price_usd_ohlc',
                'volume_exchanged',
                'market_cap_realized_usd'
            ]
            
            dfs = []
            for metric in metrics:
                async with self.rate_limiter:
                    url = f"https://api.glassnode.com/v1/metrics/{metric}"
                    params = {
                        'a': base,
                        'i': self._convert_timeframe(timeframe),
                        'f': 'json'
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()
                        df = pd.DataFrame(data)
                        df['t'] = pd.to_datetime(df['t'], unit='s')
                        df.set_index('t', inplace=True)
                        dfs.append(df)
            
            # Combine all metrics
            combined = pd.concat(dfs, axis=1)
            return self._process_response(combined)
            
        except Exception as e:
            logger.error(f"GlassNode fetch failed: {str(e)}")
            return None
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to GlassNode format"""
        conversions = {
            '1h': '1h',
            '4h': '4h',
            '1d': '24h',
            '1w': '1w'
        }
        return conversions.get(timeframe, '24h')
    
    def _process_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GlassNode response into standard format"""
        # Rename columns to standard format
        columns = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'mc': 'market_cap'
        }
        df.rename(columns=columns, inplace=True)
        return df

class CoinGeckoAPI:
    """CoinGecko API wrapper with rate limiting"""
    
    def __init__(self):
        self.session = self._init_session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=10,  # Reduced rate limit for free tier
            per_second=60
        )
        self._coin_list_cache = {}
    
    def _init_session(self) -> requests.Session:
        """Initialize requests session"""
        session = requests.Session()
        session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'Mozilla/5.0'
        })
        return session
    
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Get market data from CoinGecko"""
        try:
            # Remove trading pair separator and get base currency
            base = symbol.split('/')[0].lower()
            
            async with self.rate_limiter:
                # Get coin ID
                coin_id = await self._get_coin_id(base)
                if not coin_id:
                    logger.warning(f"Coin ID not found for {base}")
                    return None

                # Fetch market data with retry logic
                for attempt in range(3):
                    try:
                        response = self.session.get(
                            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                            params={
                                'vs_currency': 'usd',  # Changed from usdt
                                'days': self._calculate_days(timeframe, limit),
                                'interval': 'daily'  # Simplified interval
                            },
                            timeout=10
                        )
                        response.raise_for_status()
                        data = response.json()
                        
                        return self._process_response(data)
                    
                    except requests.exceptions.RequestException as e:
                        if attempt == 2:  # Last attempt
                            raise
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        except Exception as e:
            logger.error(f"CoinGecko API error: {str(e)}")
            return None
    
    async def _get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko coin ID for symbol"""
        try:
            # Check cache
            if symbol in self._coin_list_cache:
                return self._coin_list_cache[symbol]
            
            async with self.rate_limiter:
                response = self.session.get(
                    "https://api.coingecko.com/api/v3/coins/list",
                    timeout=10
                )
                response.raise_for_status()
                coins = response.json()
            
            # Update cache
            self._coin_list_cache = {
                coin['symbol'].lower(): coin['id']
                for coin in coins
            }
            
            return self._coin_list_cache.get(symbol)
            
        except Exception as e:
            logger.error(f"Error getting coin ID: {str(e)}")
            return None

    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to CoinGecko format"""
        conversions = {
            '1m': 'minutely',
            '5m': 'minutely',
            '15m': 'minutely',
            '1h': 'hourly',
            '4h': 'hourly',
            '1d': 'daily'
        }
        return conversions.get(timeframe, 'daily')
    
    def _calculate_days(self, timeframe: str, limit: int) -> int:
        """Calculate number of days needed"""
        intervals_per_day = {
            '1m': 1440,
            '5m': 288,
            '15m': 96,
            '1h': 24,
            '4h': 6,
            '1d': 1
        }
        
        intervals = intervals_per_day.get(timeframe, 1)
        return math.ceil(limit / intervals) + 1
    
    def _process_response(self, data: Dict) -> pd.DataFrame:
        """Process CoinGecko response into OHLCV format"""
        # Extract price and volume data
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        
        # Merge data
        df = prices.merge(volumes, on='timestamp')
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Create OHLCV format
        ohlcv = df.resample('1min').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })
        
        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        return ohlcv

class AlternativeAPI:
    """Alternative.me Fear & Greed Index API wrapper"""
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        self.session = requests.Session()
        self.rate_limiter = TokenBucketLimiter(
            rate_limit=1,  # 1 request
            per_second=60  # per minute
        )
    
    async def get_market_data(self, symbol: str, timeframe: str = '1d') -> Optional[pd.DataFrame]:
        """Get Fear & Greed index data"""
        try:
            with self.rate_limiter:
                response = self.session.get(
                    self.base_url,
                    params={
                        'limit': 365,  # Get one year of data
                        'format': 'json'
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if 'data' not in data:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                
                # Add columns to match OHLCV format
                value = df['value'].astype(float)
                df['open'] = value
                df['high'] = value
                df['low'] = value
                df['close'] = value
                df['volume'] = 0
                
                return df
                
        except Exception as e:
            logger.error(f"Alternative.me API error: {str(e)}")
            return None

class ModelManager:
    """Handles model training, persistence and loading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.model_metadata = {}
        self._load_models()
    
    def _load_models(self):
        """Load saved models"""
        try:
            for model_file in os.listdir(self.config.model_save_path):
                if model_file.endswith('.joblib'):
                    model_path = os.path.join(self.config.model_save_path, model_file)
                    model_name = model_file.replace('.joblib', '')
                    
                    self.models[model_name] = joblib.load(model_path)
                    self._load_model_metadata(model_name)
                    
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def _load_model_metadata(self, model_name: str):
        """Load model metadata"""
        try:
            metadata_path = os.path.join(
                self.config.model_save_path,
                f"{model_name}_metadata.json"
            )
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
            else:
                self.model_metadata[model_name] = {
                    'last_training': None,
                    'performance_metrics': {},
                    'training_size': 0
                }
                
        except Exception as e:
            logger.error(f"Error loading model metadata: {str(e)}")
            self.model_metadata[model_name] = {}
    
    def save_model(self, model_name: str, model: Any, metadata: Dict):
        """Save model and its metadata"""
        try:
            # Save model
            model_path = os.path.join(self.config.model_save_path, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata['last_training'] = datetime.datetime.now(timezone.utc).isoformat()
            metadata_path = os.path.join(self.config.model_save_path, f"{model_name}_metadata.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update internal state
            self.models[model_name] = model
            self.model_metadata[model_name] = metadata
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def get_model(self, model_name: str) -> Tuple[Any, Dict]:
        """Get model and its metadata"""
        return self.models.get(model_name), self.model_metadata.get(model_name, {})
    
    def needs_training(self, model_name: str) -> bool:
        """Check if model needs retraining"""
        metadata = self.model_metadata.get(model_name, {})
        last_training = metadata.get('last_training')
        
        if not last_training:
            return True
            
        last_training = datetime.datetime.fromisoformat(last_training)
        hours_since_training = (
            datetime.datetime.now(timezone.utc) - last_training
        ).total_seconds() / 3600
        
        return hours_since_training >= self.config.model_update_interval

class MLTrainingManager:
    """Manages ML model training and validation"""
    
    def __init__(self, config: Config, model_manager: ModelManager):
        self.config = config
        self.model_manager = model_manager
        self.feature_generators = self._initialize_feature_generators()
        
    def _initialize_feature_generators(self) -> Dict:
        """Initialize feature generators for each model type"""
        return {
            'technical': TechnicalFeatureGenerator(),
            'sentiment': SentimentFeatureGenerator(),
            'regime': RegimeFeatureGenerator()
        }
    
    def train_models(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train all ML models"""
        performance_metrics = {}
        
        for model_type, generator in self.feature_generators.items():
            try:
                # Check if model needs training
                if not self.model_manager.needs_training(model_type):
                    continue
                
                # Prepare training data
                X, y = generator.generate_features(historical_data)
                if len(X) < self.config.min_training_samples:
                    logger.warning(f"Insufficient training data for {model_type}")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    shuffle=False  # Time series data
                )
                
                # Train model
                model = self._train_model(model_type, X_train, y_train)
                
                # Evaluate
                metrics = self._evaluate_model(model, X_test, y_test)
                performance_metrics[model_type] = metrics
                
                # Save model
                self.model_manager.save_model(
                    model_type,
                    model,
                    {
                        'performance_metrics': metrics,
                        'training_size': len(X_train),
                        'features': list(X.columns)
                    }
                )
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                continue
        
        return performance_metrics
    
    def _train_model(self, model_type: str, X: pd.DataFrame, y: pd.Series) -> Any:
        """Train specific model type"""
        if model_type == 'technical':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'sentiment':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'regime':
            model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        return model
    
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Evaluate model performance"""
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        return {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, average='weighted'),
            'recall': recall_score(y, predictions, average='weighted'),
            'f1': f1_score(y, predictions, average='weighted'),
            'log_loss': log_loss(y, probabilities)
        }

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
        
    def run_backtest(
        self,
        strategy: Any,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1
    ) -> Dict:
        """Run backtest for a strategy"""
        try:
            # Initialize backtest state
            self.results = {
                'trades': [],
                'positions': [],
                'equity': [initial_capital],
                'returns': [],
                'drawdowns': []
            }
            
            current_position = None
            capital = initial_capital
            
            # Run simulation
            for i in range(len(data) - 1):
                current_data = data.iloc[:i+1]
                next_bar = data.iloc[i+1]
                
                # Get strategy signals
                signals = strategy.generate_signals(current_data)
                
                # Process existing position
                if current_position:
                    pnl = self._process_position(
                        current_position,
                        next_bar,
                        signals
                    )
                    capital += pnl
                
                # Enter new position
                if not current_position and signals['enter']:
                    current_position = self._enter_position(
                        signals['direction'],
                        next_bar,
                        capital,
                        position_size
                    )
                
                # Update equity curve
                self.results['equity'].append(capital)
                
                # Calculate returns
                if i > 0:
                    daily_return = (
                        self.results['equity'][-1] /
                        self.results['equity'][-2] - 1
                    )
                    self.results['returns'].append(daily_return)
                
                # Calculate drawdown
                if len(self.results['equity']) > 1:
                    peak = max(self.results['equity'][:-1])
                    drawdown = (capital - peak) / peak
                    self.results['drawdowns'].append(drawdown)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics()
            self.results['metrics'] = metrics
            
            return self.results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise
    
    def _process_position(
        self,
        position: Dict,
        next_bar: pd.Series,
        signals: Dict
    ) -> float:
        """Process existing position"""
        # Check exit conditions
        if self._should_exit(position, next_bar, signals):
            pnl = self._calculate_pnl(position, next_bar['close'])
            self.results['trades'].append({
                'entry_price': position['entry_price'],
                'exit_price': next_bar['close'],
                'direction': position['direction'],
                'pnl': pnl,
                'hold_time': len(self.results['equity']) - position['entry_time']
            })
            return pnl
        return 0.0
    
    def _should_exit(
        self,
        position: Dict,
        next_bar: pd.Series,
        signals: Dict
    ) -> bool:
        """Check exit conditions"""
        # Stop loss
        if position['direction'] == 'long':
            if next_bar['low'] <= position['stop_loss']:
                return True
        else:
            if next_bar['high'] >= position['stop_loss']:
                return True
        
        # Take profit
        if position['direction'] == 'long':
            if next_bar['high'] >= position['take_profit']:
                return True
        else:
            if next_bar['low'] <= position['take_profit']:
                return True
        
        # Signal exit
        if signals.get('exit', False):
            return True
        
        return False
    
    def _enter_position(
        self,
        direction: str,
        next_bar: pd.Series,
        capital: float,
        position_size: float
    ) -> Dict:
        """Enter new position"""
        entry_price = next_bar['open']
        position_capital = capital * position_size
        size = position_capital / entry_price
        
        # Calculate stop loss and take profit
        atr = self._calculate_atr(next_bar)
        if direction == 'long':
            stop_loss = entry_price - 2 * atr
            take_profit = entry_price + 3 * atr
        else:
            stop_loss = entry_price + 2 * atr
            take_profit = entry_price - 3 * atr
        
        position = {
            'direction': direction,
            'size': size,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': len(self.results['equity'])
        }
        
        self.results['positions'].append(position)
        return position
    
    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate position P&L"""
        if position['direction'] == 'long':
            return position['size'] * (exit_price - position['entry_price'])
        else:
            return position['size'] * (position['entry_price'] - exit_price)
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = np.array(self.results['returns'])
        equity = np.array(self.results['equity'])
        drawdowns = np.array(self.results['drawdowns'])
        trades = self.results['trades']
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        return {
            'total_return': (equity[-1] - equity[0]) / equity[0],
            'annual_return': self._calculate_annual_return(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': np.min(drawdowns) if len(drawdowns) > 0 else 0,
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'profit_factor': self._calculate_profit_factor(winning_trades, losing_trades),
            'avg_trade': np.mean([t['pnl'] for t in trades]) if trades else 0,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
            'avg_hold_time': np.mean([t['hold_time'] for t in trades]) if trades else 0
        }
    
    def _calculate_annual_return(self, returns: np.ndarray) -> float:
        """Calculate annualized return"""
        if len(returns) == 0:
            return 0.0
        return np.mean(returns) * 252  # Assuming daily data
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - self.config.risk_free_rate/252
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_profit_factor(
        self,
        winning_trades: List[Dict],
        losing_trades: List[Dict]
    ) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        if gross_loss == 0:
            return float('inf')
        return gross_profit / gross_loss if gross_loss > 0 else 0
    
    def _calculate_atr(self, bar: pd.Series, period: int = 14) -> float:
        """Calculate ATR for position sizing"""
        return 0.02 * bar['close']  # Simplified ATR calculation

    def _calculate_monthly_returns(self, equity: np.ndarray) -> List[float]:
        """Calculate monthly returns"""
        if len(equity) < 22:  # Minimum one month of data
            return []
            
        monthly_equity = []
        for i in range(0, len(equity), 22):  # Approximate trading days in a month
            monthly_equity.append(equity[i])
            
        returns = np.diff(monthly_equity) / monthly_equity[:-1]
        return returns.tolist()

    def _calculate_recovery_factor(self, equity: np.ndarray) -> float:
        """Calculate recovery factor"""
        if len(equity) < 2:
            return 0.0
        total_return = equity[-1] - equity[0]
        max_drawdown = self._calculate_max_drawdown_amount(equity)
        return total_return / max_drawdown if max_drawdown > 0 else 0.0

    def _calculate_max_drawdown_amount(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown in currency amount"""
        peak = equity[0]
        max_dd = 0
        
        for value in equity[1:]:
            if value > peak:
                peak = value
            dd = peak - value
            max_dd = max(max_dd, dd)
        
        return max_dd

class ModelState(Base):
    """Model state database model"""
    __tablename__ = 'model_states'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    state = Column(Text, nullable=False)

class Prediction(Base):
    """Prediction database model"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    action = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'action': self.action,
            'confidence': self.confidence,
            'price': self.price
        }

class EnhancedDataFetcher:
    """Enhanced data fetching with multi-source validation"""   
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.api_config = api_config
        self.session = aiohttp.ClientSession()
        self.exchanges = {}
        self.cache = {}
        self.data_sources = self._initialize_data_sources()
        self._initialize_exchanges()
        self.secondary_sources = SecondaryDataSource(config, api_config)

    def _initialize_data_sources(self) -> Dict:
        return {
            'coingecko': CoinGeckoAPI(),
            'cryptocompare': CryptoCompareAPI(self.api_config.cryptocompare_key),
            'messari': MessariAPI(self.api_config.messari_key),
            'santiment': SantimentAPI(self.api_config.santiment_key),
            'glassnode': GlassNodeAPI(self.api_config.glassnode_key),
            'alternative': AlternativeAPI()
        }

    def _initialize_exchanges(self):
        exchange_configs = {
            'binance': {'apiKey': self.api_config.binance_key, 'secret': self.api_config.binance_secret},
            #'kucoin': {'apiKey': self.api_config.kucoin_key, 'secret': self.api_config.kucoin_secret},
            'huobi': {'apiKey': self.api_config.huobi_key, 'secret': self.api_config.huobi_secret}
            #'ftx': {'apiKey': self.api_config.ftx_key, 'secret': self.api_config.ftx_secret}
        }

        for name, config in exchange_configs.items():
            try:
                exchange_class = getattr(ccxt, name)
                self.exchanges[name] = exchange_class({
                    **config,
                    'enableRateLimit': True,
                    'asyncio_loop': asyncio.get_event_loop()
                })
                logger.info(f"Initialized {name} exchange")
            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")

    async def fetch_market_data(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Try primary exchanges
            primary_data = await self._fetch_from_primary(symbol, timeframe)
            if primary_data is not None and not primary_data.empty:
                self.cache[cache_key] = primary_data
                return primary_data

            # Try secondary sources
            secondary_data = await self.secondary_sources.fetch_market_data(symbol, timeframe)
            if secondary_data is not None and not secondary_data.empty:
                self.cache[cache_key] = secondary_data
                return secondary_data

            raise DataError(f"No data available for {symbol}")

        except Exception as e:
            logger.error(f"Market data fetch error: {e}")
            raise

    async def _fetch_from_primary(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        for exchange_name, exchange in self.exchanges.items():
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
                if ohlcv:
                    df = pd.DataFrame(
                        ohlcv,
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.error(f"{exchange_name} fetch failed: {e}")
                continue
        return None

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()
        if hasattr(self, 'secondary_sources'):
            await self.secondary_sources.cleanup()

    def _fetch_from_secondary(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """Fetch from secondary sources"""
        try:
            # Try CoinGecko first
            data = self.data_sources['coingecko'].get_market_data(
                symbol,
                timeframe,
                limit
            )
            
            if data is not None:
                data['source'] = 'coingecko'
                return data
            
            # Try CryptoCompare as fallback
            if self.api_config.cryptocompare_key:
                data = self.data_sources['cryptocompare'].get_market_data(
                    symbol,
                    timeframe,
                    limit
                )
                
                if data is not None:
                    data['source'] = 'cryptocompare'
                    return data
            
            return None
            
        except Exception as e:
            logger.warning(f"Secondary source fetch failed: {str(e)}")
            return None
    
    def _validate_and_reconcile(
        self,
        primary_data: Optional[pd.DataFrame],
        secondary_data: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Validate and reconcile data from multiple sources"""
        if primary_data is None and secondary_data is None:
            raise DataFetchError("No data available from any source")
        
        if primary_data is None:
            return secondary_data
            
        if secondary_data is None:
            return primary_data
        
        try:
            # Align timestamps
            common_index = primary_data.index.intersection(secondary_data.index)
            primary = primary_data.loc[common_index]
            secondary = secondary_data.loc[common_index]
            
            # Check for significant deviations
            close_diff_pct = abs(
                (primary['close'] - secondary['close']) / primary['close']
            )
            
            significant_deviation = close_diff_pct > 0.01  # 1% threshold
            
            if significant_deviation.any():
                logger.warning("Significant price deviations detected between sources")
                
                # Use weighted average for deviating prices
                weights = {'binance': 0.6, 'kucoin': 0.5, 'huobi': 0.5,
                         'coingecko': 0.4, 'cryptocompare': 0.4}
                
                primary_weight = weights.get(primary['source'].iloc[0], 0.5)
                secondary_weight = weights.get(secondary['source'].iloc[0], 0.5)
                
                # Normalize weights
                total_weight = primary_weight + secondary_weight
                primary_weight /= total_weight
                secondary_weight /= total_weight
                
                # Combine data
                for col in ['open', 'high', 'low', 'close']:
                    primary.loc[significant_deviation, col] = (
                        primary.loc[significant_deviation, col] * primary_weight +
                        secondary.loc[significant_deviation, col] * secondary_weight
                    )
            
            return primary
            
        except Exception as e:
            logger.error(f"Data reconciliation error: {str(e)}")
            return primary_data
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate market data"""
        try:
            # Check for missing values
            if df.isnull().any().any():
                return False
            
            # Check for negative values
            if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
                return False
            
            # Check price relationships
            valid_prices = (
                (df['high'] >= df['low']).all() and
                (df['high'] >= df['open']).all() and
                (df['high'] >= df['close']).all() and
                (df['low'] <= df['open']).all() and
                (df['low'] <= df['close']).all()
            )
            
            if not valid_prices:
                return False
            
            # Check for duplicate timestamps
            if df.index.duplicated().any():
                return False
            
            # Check for large gaps
            time_diffs = df.index.to_series().diff()
            expected_diff = pd.Timedelta(self._get_expected_interval(df))
            
            if (time_diffs > expected_diff * 2).any():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False
    
    def _normalize_symbol(self, symbol: str, exchange) -> str:
        """Normalize trading symbol for specific exchange"""
        try:
            # Remove slash if present
            normalized = symbol.replace('/', '')
            
            # Check if symbol exists on exchange
            markets = exchange.load_markets()
            
            # Try direct match
            if symbol in markets:
                return symbol
                
            # Try normalized version
            if normalized in markets:
                return normalized
            
            # Try with different separators
            separators = ['/', '-', '_', '']
            base, quote = symbol.split('/')
            
            for sep in separators:
                test_symbol = f"{base}{sep}{quote}"
                if test_symbol in markets:
                    return test_symbol
            
            raise ValueError(f"Symbol {symbol} not found on {exchange.id}")
            
        except Exception as e:
            logger.error(f"Symbol normalization error: {str(e)}")
            raise
    
    def _get_expected_interval(self, df: pd.DataFrame) -> str:
        """Get expected time interval from data"""
        common_intervals = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        # Calculate median time difference
        median_diff = df.index.to_series().diff().median()
        
        # Find closest matching interval
        for timeframe, pd_freq in common_intervals.items():
            if pd.Timedelta(pd_freq) == median_diff:
                return pd_freq
        
        return '1D'  # Default to daily

class BaseAgent(ABC):
    """Enhanced base agent with improved decision making"""
    
    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight
        self.confidence_history = []
        self.prediction_history = []
        
    @abstractmethod
    async def analyze(self, data: pd.DataFrame, market_data: Dict) -> Dict:
        """Analyze market data and generate predictions"""
        pass
    
    def _calculate_confidence(self, score: float, market_conditions: Dict) -> float:
        """Calculate confidence score with market conditions"""
        base_confidence = abs(score) * 100
        
        # Adjust confidence based on market conditions
        volatility = market_conditions.get('volatility', 0.5)
        trend_strength = market_conditions.get('trend_strength', 0.5)
        
        # Reduce confidence in high volatility
        if volatility > 0.7:
            base_confidence *= 0.8
        
        # Increase confidence in strong trends
        if trend_strength > 0.7:
            base_confidence *= 1.2
        
        return min(base_confidence, 100)
    
    def _validate_prediction(self, prediction: Dict) -> Dict:
        """Validate prediction output"""
        required_keys = {'action', 'confidence', 'reasoning'}
        
        if not all(key in prediction for key in required_keys):
            raise ValueError(f"Missing required keys in prediction: {required_keys - prediction.keys()}")
        
        if prediction['confidence'] < 0 or prediction['confidence'] > 100:
            raise ValueError("Confidence must be between 0 and 100")
        
        return prediction
    
    def update_history(self, prediction: Dict):
        """Update prediction history"""
        self.prediction_history.append({
            'timestamp': datetime.datetime.now(timezone.utc),
            'action': prediction['action'],
            'confidence': prediction['confidence']
        })
        
        # Keep only recent history
        if len(self.prediction_history) > 1000:
            self.prediction_history.pop(0)
    
    def calculate_historical_accuracy(self) -> float:
        """Calculate historical prediction accuracy"""
        if not self.prediction_history:
            return 0.5  # Default accuracy
        
        # Implementation depends on the ability to verify past predictions
        # This is a placeholder for actual implementation
        return 0.7

class MarketRegimeAgent(BaseAgent):
    """Detect and analyze market regimes"""
    
    def __init__(self, weight: float = 0.15):
        super().__init__("MarketRegime", weight)
        self.classifier = self._build_classifier()
    
    def _build_classifier(self) -> GradientBoostingClassifier:
        """Build market regime classifier"""
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime classification"""
        df = data.copy()
        
        # Volatility features
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Trend features
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / df['close']
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_std'] = df['volume'].rolling(window=20).std()
        
        return df.dropna()
    
    async def analyze(self, data: pd.DataFrame, market_data: Dict) -> Dict:
        """Analyze market regime"""
        try:
            features_df = self._extract_regime_features(data)
            
            # Classify market regime
            regime_features = features_df.iloc[-1:]
            regime_prob = self.classifier.predict_proba(regime_features)
            
            # Determine current regime
            regimes = ['trending', 'volatile', 'ranging']
            current_regime = regimes[regime_prob.argmax()]
            confidence = float(regime_prob.max() * 100)
            
            # Generate trading action based on regime
            action = self._get_regime_action(current_regime, features_df)
            
            return self._validate_prediction({
                'action': action,
                'confidence': confidence,
                'regime': current_regime,
                'reasoning': [
                    f"Detected {current_regime} market regime",
                    f"Volatility: {features_df['volatility'].iloc[-1]:.3f}",
                    f"Trend strength: {features_df['trend_strength'].iloc[-1]:.3f}"
                ]
            })
            
        except Exception as e:
            logger.error(f"Market regime analysis error: {str(e)}")
            raise ModelError(f"Regime analysis failed: {str(e)}")
    
    def _get_regime_action(self, regime: str, features_df: pd.DataFrame) -> str:
        """Determine action based on market regime"""
        if regime == 'trending':
            # Check trend direction
            sma_20 = features_df['sma_20'].iloc[-1]
            sma_50 = features_df['sma_50'].iloc[-1]
            return "Buy" if sma_20 > sma_50 else "Sell"
        elif regime == 'volatile':
            return "Hold"  # Conservative in volatile markets
        else:  # ranging
            # Check position within range
            upper = features_df['high'].rolling(20).max().iloc[-1]
            lower = features_df['low'].rolling(20).min().iloc[-1]
            current = features_df['close'].iloc[-1]
            
            position_in_range = (current - lower) / (upper - lower)
            
            if position_in_range > 0.8:
                return "Sell"  # Near range top
            elif position_in_range < 0.2:
                return "Buy"  # Near range bottom
            return "Hold"

class TechnicalAnalysisAgent(BaseAgent):
    """Enhanced Technical Analysis with ML Models"""  
    def __init__(self, weight: float = 0.3):
        super().__init__("Technical", weight)
        self.model_manager = None
        self.training_manager = None
        self.scaler = StandardScaler()
        self.pattern_recognizer = PatternRecognizer()
        
    def analyze(self, data: pd.DataFrame, market_data: Dict) -> Dict:
        """Enhanced technical analysis with multiple approaches"""
        try:
            # Extract comprehensive features
            features_df = self._extract_features(data)
            
            # Get predictions from different sources
            ml_prediction = self._get_ml_prediction(features_df)
            pattern_prediction = self.pattern_recognizer.analyze(data)
            indicator_prediction = self._analyze_indicators(features_df)
            trend_prediction = self._analyze_trend(features_df)
            
            # Combine predictions with dynamic weighting
            combined_prediction = self._combine_predictions([
                (ml_prediction, 0.4),      # ML model
                (pattern_prediction, 0.2),  # Pattern recognition
                (indicator_prediction, 0.2),# Technical indicators
                (trend_prediction, 0.2)     # Trend analysis
            ])
            
            # Add market context analysis
            market_context = self._analyze_market_context(features_df)
            final_prediction = self._adjust_for_market_context(
                combined_prediction,
                market_context
            )
            
            return self._validate_prediction(final_prediction)
            
        except Exception as e:
            logger.error(f"Technical analysis error: {str(e)}")
            raise ModelError(f"Technical analysis failed: {str(e)}")
    
    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive technical features"""
        df = data.copy()
        
        # Price action features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Momentum features
        for period in [14, 28, 56]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
            df[f'willr_{period}'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Trend features
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
            
            # Trend strength indicators
            df[f'trend_strength_{period}'] = (
                (df['close'] - df[f'sma_{period}']) / 
                df[f'sma_{period}']
            ).abs()
        
        # Volatility features
        for period in [14, 28]:
            df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            df[f'natr_{period}'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Volume features
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'])
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Pattern recognition features
        df = self._add_candlestick_patterns(df)
        
        return df.dropna()
    
    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features"""
        # Single candle patterns
        df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
        df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Double candle patterns
        df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
        
        # Triple candle patterns
        df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        return df
    
    def _get_ml_prediction(self, features: pd.DataFrame) -> Dict:
        """Get prediction from ML model"""
        if not self.model_manager or not self.model_manager.has_model('technical'):
            return {
                'action': 'Hold',
                'confidence': 0,
                'type': 'ml_model'
            }
        
        try:
            model, metadata = self.model_manager.get_model('technical')
            features_scaled = self.scaler.transform(features[metadata['features']])
            
            prediction = model.predict_proba(features_scaled)
            confidence = float(np.max(prediction) * 100)
            
            return {
                'action': self._convert_prediction(prediction),
                'confidence': confidence,
                'type': 'ml_model'
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {str(e)}")
            return {
                'action': 'Hold',
                'confidence': 0,
                'type': 'ml_model'
            }
    
    def _analyze_indicators(self, features: pd.DataFrame) -> Dict:
        """Analyze technical indicators"""
        latest = features.iloc[-1]
        
        # RSI Analysis
        rsi_signals = [
            1 if latest[f'rsi_{period}'] < 30 else
            -1 if latest[f'rsi_{period}'] > 70 else
            0 for period in [14, 28, 56]
        ]
        
        # MACD Analysis
        macd_signal = 1 if latest['macd'] > latest['macd_signal'] else -1
        
        # Volume Analysis
        volume_signal = self._analyze_volume_trend(features)
        
        # Combine signals
        total_signal = sum(rsi_signals) + macd_signal + volume_signal
        signal_count = len(rsi_signals) + 2  # RSIs + MACD + Volume
        
        # Calculate confidence and action
        confidence = abs(total_signal) / signal_count * 100
        
        if total_signal > 0:
            action = 'Buy'
        elif total_signal < 0:
            action = 'Sell'
        else:
            action = 'Hold'
            
        return {
            'action': action,
            'confidence': min(confidence, 100),
            'type': 'indicators'
        }
    
    def _analyze_volume_trend(self, features: pd.DataFrame) -> int:
        """Analyze volume trend"""
        recent_volume = features['volume'].tail(20)
        volume_ma = recent_volume.mean()
        current_volume = recent_volume.iloc[-1]
        
        volume_trend = current_volume / volume_ma
        
        if volume_trend > 1.5:
            return 1 if features['returns'].iloc[-1] > 0 else -1
        elif volume_trend < 0.5:
            return -1 if features['returns'].iloc[-1] > 0 else 1
        return 0
    
    def _analyze_trend(self, features: pd.DataFrame) -> Dict:
        """Analyze price trend"""
        latest = features.iloc[-1]
        
        # Analyze multiple timeframe trends
        trend_signals = []
        
        for period in [20, 50, 200]:
            # Price vs MA
            if latest['close'] > latest[f'sma_{period}']:
                trend_signals.append(1)
            else:
                trend_signals.append(-1)
                
            # MA slopes
            ma_slope = (
                latest[f'sma_{period}'] - 
                features[f'sma_{period}'].iloc[-5]
            ) / latest[f'sma_{period}']
            
            trend_signals.append(1 if ma_slope > 0 else -1)
        
        # Calculate overall trend
        trend_score = sum(trend_signals) / len(trend_signals)
        confidence = abs(trend_score) * 100
        
        if trend_score > 0.3:
            action = 'Buy'
        elif trend_score < -0.3:
            action = 'Sell'
        else:
            action = 'Hold'
            
        return {
            'action': action,
            'confidence': min(confidence, 100),
            'type': 'trend'
        }
    
    def _adjust_for_market_context(self, prediction: Dict, context: Dict) -> Dict:
        """Adjust prediction based on market context"""
        confidence = prediction['confidence']
        
        # Adjust for volatility
        if context['volatility'] > 0.5:
            confidence *= 0.8
            prediction['reasoning'].append("High volatility - reduced confidence")
            
        # Adjust for trend strength
        if context['trend_strength'] > 0.7:
            confidence *= 1.2
            prediction['reasoning'].append("Strong trend - increased confidence")
            
        # Adjust for volume support
        if context['volume_support']:
            confidence *= 1.1
            prediction['reasoning'].append("Strong volume support")
        else:
            confidence *= 0.9
            prediction['reasoning'].append("Weak volume support")
            
        prediction['confidence'] = min(confidence, 100)
        return prediction
    
    def _analyze_market_context(self, features: pd.DataFrame) -> Dict:
        """Analyze broader market context"""
        latest = features.iloc[-1]
        
        return {
            'volatility': latest['natr_14'],
            'trend_strength': latest['trend_strength_50'],
            'volume_support': self._check_volume_support(features),
            'squeeze': latest['bb_width'] < 0.1
        }
    
    def _check_volume_support(self, features: pd.DataFrame) -> bool:
        """Check if volume supports the current move"""
        recent = features.tail(5)
        
        # Volume increasing with price
        volume_increasing = recent['volume'].is_monotonic_increasing
        price_change = recent['close'].iloc[-1] / recent['close'].iloc[0] - 1
        
        return volume_increasing and (
            (price_change > 0 and recent['returns'].mean() > 0) or
            (price_change < 0 and recent['returns'].mean() < 0)
        )

    def _check_head_shoulders(self, data: pd.DataFrame) -> bool:
        """Check for head and shoulders pattern"""
        window = data.tail(self.window_size)
        highs = argrelextrema(window['high'].values, np.greater)[0]
        
        if len(highs) < 3:
            return False
        
        # Get last three peaks
        last_three_highs = window['high'].iloc[highs[-3:]]
        
        # Check pattern formation
        # Middle peak should be highest (head)
        if (last_three_highs.iloc[1] > last_three_highs.iloc[0] and 
            last_three_highs.iloc[1] > last_three_highs.iloc[2]):
            # Shoulders should be at similar levels
            shoulder_diff = abs(
                last_three_highs.iloc[0] - last_three_highs.iloc[2]
            ) / last_three_highs.iloc[0]
            
            if shoulder_diff < 0.03:  # 3% tolerance
                # Verify neckline
                lows_between = window['low'].iloc[highs[-3]:highs[-1]]
                if min(lows_between) < window['low'].iloc[-1]:
                    return True
        
        return False
    
    def _check_inverse_head_shoulders(self, data: pd.DataFrame) -> bool:
        """Check for inverse head and shoulders pattern"""
        window = data.tail(self.window_size)
        lows = argrelextrema(window['low'].values, np.less)[0]
        
        if len(lows) < 3:
            return False
        
        # Get last three troughs
        last_three_lows = window['low'].iloc[lows[-3:]]
        
        # Check pattern formation
        # Middle trough should be lowest (head)
        if (last_three_lows.iloc[1] < last_three_lows.iloc[0] and 
            last_three_lows.iloc[1] < last_three_lows.iloc[2]):
            # Shoulders should be at similar levels
            shoulder_diff = abs(
                last_three_lows.iloc[0] - last_three_lows.iloc[2]
            ) / last_three_lows.iloc[0]
            
            if shoulder_diff < 0.03:  # 3% tolerance
                # Verify neckline
                highs_between = window['high'].iloc[lows[-3]:lows[-1]]
                if max(highs_between) > window['high'].iloc[-1]:
                    return True
        
        return False
    
    def _check_triangle(self, data: pd.DataFrame) -> Optional[str]:
        """Check for triangle patterns (ascending, descending, symmetrical)"""
        window = data.tail(self.window_size)
        highs = argrelextrema(window['high'].values, np.greater)[0]
        lows = argrelextrema(window['low'].values, np.less)[0]
        
        if len(highs) < 2 or len(lows) < 2:
            return None
        
        # Calculate trend lines
        high_slope = self._calculate_slope(window['high'].iloc[highs])
        low_slope = self._calculate_slope(window['low'].iloc[lows])
        
        # Classify triangle pattern
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            return 'ascending'
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            return 'descending'
        elif high_slope < -0.001 and low_slope > 0.001:
            return 'symmetrical'
        
        return None
    
    def _calculate_slope(self, points: pd.Series) -> float:
        """Calculate slope of line through points"""
        if len(points) < 2:
            return 0
        
        x = np.arange(len(points))
        y = points.values
        
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def _get_consensus_action(self, patterns: List[Tuple[str, str]]) -> str:
        """Get consensus action from multiple patterns"""
        pattern_weights = {
            'double_bottom': 1.0,
            'double_top': 1.0,
            'head_shoulders': 1.2,
            'inverse_head_shoulders': 1.2,
            'ascending_triangle': 0.8,
            'descending_triangle': 0.8,
            'symmetrical_triangle': 0.6
        }
        
        buy_score = 0.0
        sell_score = 0.0
        
        for pattern, action in patterns:
            weight = pattern_weights.get(pattern, 1.0)
            if action == 'Buy':
                buy_score += weight
            else:
                sell_score += weight
        
        if buy_score > sell_score:
            return 'Buy'
        elif sell_score > buy_score:
            return 'Sell'
        return 'Hold'
    
    def _calculate_pattern_confidence(self, patterns: List[Tuple[str, str]], data: pd.DataFrame) -> float:
        """Calculate confidence score for pattern predictions"""
        # Base confidence on number and reliability of patterns
        base_confidence = min(len(patterns) * 25, 100)  # 25% per pattern, max 100%
        
        # Adjust confidence based on pattern completion
        completion_scores = []
        for pattern, _ in patterns:
            score = self._calculate_pattern_completion(pattern, data)
            completion_scores.append(score)
        
        # Average completion score
        avg_completion = sum(completion_scores) / len(completion_scores)
        
        # Final confidence is weighted average
        final_confidence = (base_confidence * 0.7) + (avg_completion * 0.3)
        
        return min(final_confidence, 100)
    
    def _calculate_pattern_completion(self, pattern: str, data: pd.DataFrame) -> float:
        """Calculate how complete/ideal a pattern formation is"""
        if pattern in ['double_bottom', 'double_top']:
            return self._calculate_double_pattern_completion(data)
        elif pattern in ['head_shoulders', 'inverse_head_shoulders']:
            return self._calculate_hs_pattern_completion(data)
        elif 'triangle' in pattern:
            return self._calculate_triangle_completion(data)
        return 50.0  # Default completion score
    
    def _calculate_double_pattern_completion(self, data: pd.DataFrame) -> float:
        """Calculate completion score for double top/bottom patterns"""
        window = data.tail(self.window_size)
        
        # Check symmetry of the formation
        extremes = argrelextrema(window['close'].values, np.less)[0]
        if len(extremes) < 2:
            return 50.0
        
        # Calculate time symmetry
        time_diff = extremes[-1] - extremes[-2]
        ideal_time_diff = self.window_size // 3
        time_score = 100 * (1 - abs(time_diff - ideal_time_diff) / ideal_time_diff)
        
        # Calculate price symmetry
        price_values = window['close'].iloc[extremes[-2:]]
        price_diff = abs(price_values.iloc[1] - price_values.iloc[0])
        price_score = 100 * (1 - price_diff / price_values.iloc[0])
        
        return (time_score * 0.4) + (price_score * 0.6)
    
    def _calculate_hs_pattern_completion(self, data: pd.DataFrame) -> float:
        """Calculate completion score for head and shoulders patterns"""
        window = data.tail(self.window_size)
        highs = argrelextrema(window['high'].values, np.greater)[0]
        
        if len(highs) < 3:
            return 50.0
        
        # Get the three peaks
        peaks = window['high'].iloc[highs[-3:]]
        
        # Check head height relative to shoulders
        head_height = peaks.iloc[1] - min(peaks.iloc[0], peaks.iloc[2])
        ideal_head_height = (peaks.iloc[0] + peaks.iloc[2]) * 0.1  # 10% above shoulders
        head_score = 100 * (1 - abs(head_height - ideal_head_height) / ideal_head_height)
        
        # Check shoulder symmetry
        shoulder_diff = abs(peaks.iloc[0] - peaks.iloc[2])
        shoulder_score = 100 * (1 - shoulder_diff / peaks.iloc[0])
        
        return (head_score * 0.6) + (shoulder_score * 0.4)
    
    def _calculate_triangle_completion(self, data: pd.DataFrame) -> float:
        """Calculate completion score for triangle patterns"""
        window = data.tail(self.window_size)
        
        # Calculate trend line convergence
        highs = window['high'].rolling(5).max()
        lows = window['low'].rolling(5).min()
        
        high_slope = self._calculate_slope(highs)
        low_slope = self._calculate_slope(lows)
        
        # Perfect triangle has slopes of equal magnitude but opposite signs
        slope_diff = abs(abs(high_slope) - abs(low_slope))
        slope_score = 100 * (1 - slope_diff / max(abs(high_slope), abs(low_slope)))
        
        # Check volume pattern (should decrease)
        volume_trend = self._calculate_slope(window['volume'])
        volume_score = 100 * (1 - max(volume_trend, 0) / window['volume'].max())
        
        return (slope_score * 0.7) + (volume_score * 0.3)

    def train(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train technical analysis models"""
        if not self.training_manager:
            raise ValueError("Training manager not initialized")
        
        try:
            # Extract features for training
            features_df = self._extract_features(historical_data)
            
            # Generate labels
            labels = self._generate_training_labels(historical_data)
            
            # Split data
            train_size = int(len(features_df) * 0.8)
            train_features = features_df[:train_size]
            train_labels = labels[:train_size]
            
            val_features = features_df[train_size:]
            val_labels = labels[train_size:]
            
            # Scale features
            self.scaler.fit(train_features)
            train_features_scaled = self.scaler.transform(train_features)
            val_features_scaled = self.scaler.transform(val_features)
            
            # Train model
            model = self._train_model(train_features_scaled, train_labels)
            
            # Evaluate
            metrics = self._evaluate_model(model, val_features_scaled, val_labels)
            
            # Save model
            self.model_manager.save_model(
                'technical',
                model,
                {
                    'performance_metrics': metrics,
                    'features': list(features_df.columns),
                    'scaler': self.scaler
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training error: {str(e)}")
            raise
        
    def _generate_training_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Generate training labels for supervised learning"""
        # Calculate future returns
        future_returns = data['close'].shift(-5).pct_change(5)
        
        # Create labels based on return thresholds
        labels = np.zeros(len(data))
        labels[future_returns > 0.02] = 2  # Buy
        labels[future_returns < -0.02] = 0  # Sell
        labels[(future_returns >= -0.02) & (future_returns <= 0.02)] = 1  # Hold
        
        return labels[:-5]  # Remove last 5 rows due to shift

class SentimentAnalysisAgent(BaseAgent):
    """Enhanced Sentiment Analysis with NLP"""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Sentiment", weight)
        self.nlp_model = self._initialize_nlp_model()
        self.news_cache = {}
        self.social_cache = {}
    
    def _initialize_nlp_model(self):
        """Initialize NLP model for sentiment analysis"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            return (model, tokenizer)
        except Exception as e:
            logger.error(f"NLP model initialization error: {str(e)}")
            return None
    
    async def analyze(self, data: pd.DataFrame, market_data: Dict) -> Dict:
        """Perform comprehensive sentiment analysis"""
        try:
            # Gather sentiment data from multiple sources
            news_sentiment = await self._analyze_news_sentiment()
            social_sentiment = await self._analyze_social_sentiment()
            market_sentiment = self._analyze_market_sentiment(data)
            onchain_sentiment = await self._analyze_onchain_sentiment()
            
            # Combine sentiment scores
            combined_sentiment = self._combine_sentiment_scores([
                (news_sentiment, 0.3),
                (social_sentiment, 0.3),
                (market_sentiment, 0.2),
                (onchain_sentiment, 0.2)
            ])
            
            # Generate trading action
            action = self._get_sentiment_action(combined_sentiment)
            confidence = abs(combined_sentiment['score']) * 100
            
            return self._validate_prediction({
                'action': action,
                'confidence': min(confidence, 100),
                'sentiment_score': combined_sentiment['score'],
                'reasoning': combined_sentiment['reasoning']
            })
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            raise ModelError(f"Sentiment analysis failed: {str(e)}")
    
    async def _analyze_news_sentiment(self) -> Dict:
        """Analyze sentiment from news sources"""
        try:
            news_data = await self._fetch_crypto_news()
            sentiments = []
            
            if self.nlp_model:
                model, tokenizer = self.nlp_model
                
                for article in news_data:
                    inputs = tokenizer(article['title'], return_tensors="pt", truncation=True)
                    outputs = model(**inputs)
                    sentiment = torch.softmax(outputs.logits, dim=1)
                    sentiments.append(float(sentiment[0][1] - sentiment[0][0]))  # Positive - Negative
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'score': avg_sentiment,
                'source': 'news',
                'reasoning': [f"News sentiment: {avg_sentiment:.2f}"]
            }
            
        except Exception as e:
            logger.warning(f"News sentiment analysis error: {str(e)}")
            return {'score': 0, 'source': 'news', 'reasoning': ["News analysis failed"]}
    
    async def _analyze_social_sentiment(self) -> Dict:
        """Analyze sentiment from social media"""
        try:
            twitter_sentiment = await self._analyze_twitter_sentiment()
            reddit_sentiment = await self._analyze_reddit_sentiment()
            
            combined_score = (twitter_sentiment['score'] * 0.6 + 
                            reddit_sentiment['score'] * 0.4)
            
            return {
                'score': combined_score,
                'source': 'social',
                'reasoning': twitter_sentiment['reasoning'] + reddit_sentiment['reasoning']
            }
            
        except Exception as e:
            logger.warning(f"Social sentiment analysis error: {str(e)}")
            return {'score': 0, 'source': 'social', 'reasoning': ["Social analysis failed"]}
    
    def _analyze_market_sentiment(self, data: pd.DataFrame) -> Dict:
        """Analyze sentiment from market indicators"""
        try:
            # Calculate fear and greed indicators
            rsi = talib.RSI(data['close'])[-1]
            mfi = talib.MFI(data['high'], data['low'], data['close'], data['volume'])[-1]
            
            # Get volatility
            atr = talib.ATR(data['high'], data['low'], data['close'])[-1]
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            volatility = atr / typical_price.iloc[-1]
            
            # Calculate sentiment score
            sentiment_score = 0.0
            reasoning = []
            
            # RSI interpretation
            if rsi > 70:
                sentiment_score -= 0.3
                reasoning.append("Overbought conditions (RSI)")
            elif rsi < 30:
                sentiment_score += 0.3
                reasoning.append("Oversold conditions (RSI)")
            
            # MFI interpretation
            if mfi > 80:
                sentiment_score -= 0.3
                reasoning.append("Overbought conditions (MFI)")
            elif mfi < 20:
                sentiment_score += 0.3
                reasoning.append("Oversold conditions (MFI)")
            
            # Volatility adjustment
            if volatility > 0.05:  # High volatility
                sentiment_score *= 0.8
                reasoning.append("High volatility - reduced confidence")
            
            return {
                'score': sentiment_score,
                'source': 'market',
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"Market sentiment analysis error: {str(e)}")
            return {'score': 0, 'source': 'market', 'reasoning': ["Market analysis failed"]}

    def _combine_sentiment_scores(self, sentiments: List[Tuple[Dict, float]]) -> Dict:
        """Combine sentiment scores from different sources"""
        total_score = 0
        total_weight = 0
        all_reasoning = []
        
        for sentiment, weight in sentiments:
            if sentiment['score'] != 0:  # Only consider valid sentiments
                total_score += sentiment['score'] * weight
                total_weight += weight
                all_reasoning.extend(sentiment['reasoning'])
        
        if total_weight > 0:
            final_score = total_score / total_weight
        else:
            final_score = 0
            all_reasoning.append("Insufficient sentiment data")
        
        return {
            'score': final_score,
            'reasoning': all_reasoning
        }
    
    def _get_sentiment_action(self, sentiment: Dict) -> str:
        """Convert sentiment score to trading action"""
        score = sentiment['score']
        
        if score > 0.3:
            return "Buy"
        elif score < -0.3:
            return "Sell"
        return "Hold"
    
    async def _fetch_crypto_news(self) -> List[Dict]:
        """Fetch crypto news from multiple sources"""
        try:
            news_data = []
            
            # Try CryptoCompare first if API key available
            if self.api_config.cryptocompare_key:
                news_data.extend(await self._fetch_cryptocompare_news())
            
            # Fallback or additional sources
            if not news_data:
                news_data.extend(await self._fetch_alternative_news())
            
            return news_data
            
        except Exception as e:
            logger.error(f"News fetching error: {str(e)}")
            return []
    
    async def _analyze_twitter_sentiment(self) -> Dict:
        """Analyze Twitter sentiment"""
        try:
            tweets = await self._fetch_twitter_data()
            sentiments = []
            
            if self.nlp_model and tweets:
                model, tokenizer = self.nlp_model
                
                for tweet in tweets:
                    inputs = tokenizer(tweet['text'], return_tensors="pt", truncation=True)
                    outputs = model(**inputs)
                    sentiment = torch.softmax(outputs.logits, dim=1)
                    sentiments.append(float(sentiment[0][1] - sentiment[0][0]))
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'score': avg_sentiment,
                'reasoning': [f"Twitter sentiment: {avg_sentiment:.2f}"]
            }
            
        except Exception as e:
            logger.warning(f"Twitter analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Twitter analysis failed"]}
    
    async def _analyze_reddit_sentiment(self) -> Dict:
        """Analyze Reddit sentiment"""
        try:
            reddit_posts = await self._fetch_reddit_data()
            sentiments = []
            
            if self.nlp_model and reddit_posts:
                model, tokenizer = self.nlp_model
                
                for post in reddit_posts:
                    inputs = tokenizer(post['title'], return_tensors="pt", truncation=True)
                    outputs = model(**inputs)
                    sentiment = torch.softmax(outputs.logits, dim=1)
                    sentiments.append(float(sentiment[0][1] - sentiment[0][0]))
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'score': avg_sentiment,
                'reasoning': [f"Reddit sentiment: {avg_sentiment:.2f}"]
            }
            
        except Exception as e:
            logger.warning(f"Reddit analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Reddit analysis failed"]}
    
    async def _analyze_onchain_sentiment(self) -> Dict:
        """Analyze on-chain metrics for sentiment"""
        try:
            onchain_data = await self._fetch_onchain_data()
            
            if not onchain_data:
                return {'score': 0, 'reasoning': ["No on-chain data available"]}
            
            # Analyze exchange flows
            inflow_ratio = onchain_data.get('exchange_inflow', 0)
            outflow_ratio = onchain_data.get('exchange_outflow', 0)
            
            # Analyze whale movements
            whale_accumulation = onchain_data.get('whale_accumulation', 0)
            
            # Calculate sentiment score
            sentiment_score = 0.0
            reasoning = []
            
            # Exchange flow analysis
            flow_ratio = outflow_ratio - inflow_ratio
            if flow_ratio > 0.1:
                sentiment_score += 0.3
                reasoning.append("Net exchange outflows (bullish)")
            elif flow_ratio < -0.1:
                sentiment_score -= 0.3
                reasoning.append("Net exchange inflows (bearish)")
            
            # Whale analysis
            if whale_accumulation > 0.1:
                sentiment_score += 0.4
                reasoning.append("Whale accumulation detected")
            elif whale_accumulation < -0.1:
                sentiment_score -= 0.4
                reasoning.append("Whale distribution detected")
            
            return {
                'score': sentiment_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"On-chain analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["On-chain analysis failed"]}

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()

class EnhancedSentimentAnalyzer(BaseAgent):
    """Enhanced sentiment analysis with NLP and multi-source analysis"""
    
    def __init__(self, weight: float = 0.2):
        super().__init__("Sentiment", weight)
        self.nlp_model = self._initialize_nlp_model()
        self.news_analyzer = NewsAnalyzer()
        self.social_analyzer = SocialMediaAnalyzer()
        self.on_chain_analyzer = OnChainAnalyzer()
    
    async def analyze(self, data: pd.DataFrame, market_data: Dict) -> Dict:
        """Comprehensive sentiment analysis"""
        try:
            # Gather sentiment data from multiple sources
            news_sentiment = await self.news_analyzer.analyze(market_data['symbol'])
            social_sentiment = await self.social_analyzer.analyze(market_data['symbol'])
            market_sentiment = self._analyze_market_sentiment(data)
            on_chain_sentiment = await self.on_chain_analyzer.analyze(market_data['symbol'])
            
            # Calculate weighted sentiment score
            sentiment_score = self._calculate_weighted_sentiment([
                (news_sentiment, 0.3),
                (social_sentiment, 0.3),
                (market_sentiment, 0.2),
                (on_chain_sentiment, 0.2)
            ])
            
            # Generate action and confidence
            action = self._determine_action(sentiment_score)
            confidence = self._calculate_confidence(sentiment_score)
            
            reasoning = self._generate_reasoning([
                news_sentiment,
                social_sentiment,
                market_sentiment,
                on_chain_sentiment
            ])
            
            return {
                'action': action,
                'confidence': confidence,
                'sentiment_score': sentiment_score['score'],
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {
                'action': 'Hold',
                'confidence': 0,
                'sentiment_score': 0,
                'reasoning': [f"Analysis failed: {str(e)}"]
            }
    
    def _calculate_weighted_sentiment(self, sentiments: List[Tuple[Dict, float]]) -> Dict:
        """Calculate weighted sentiment score"""
        total_score = 0
        total_weight = 0
        reasoning = []
        
        for sentiment, weight in sentiments:
            if sentiment['score'] != 0:
                total_score += sentiment['score'] * weight
                total_weight += weight
                reasoning.extend(sentiment.get('reasoning', []))
        
        if total_weight == 0:
            return {'score': 0, 'reasoning': ["Insufficient sentiment data"]}
        
        return {
            'score': total_score / total_weight,
            'reasoning': reasoning
        }
    
    def _determine_action(self, sentiment: Dict) -> str:
        """Determine action based on sentiment score"""
        score = sentiment['score']
        if score > 0.3:
            return "Buy"
        elif score < -0.3:
            return "Sell"
        return "Hold"

class NewsAnalyzer:
    """Advanced news sentiment analysis"""
    
    def __init__(self):
        self.nlp_model = self._load_nlp_model()
        self.news_cache = ExpiringDict(max_len=100, max_age_seconds=3600)
    
    async def analyze(self, symbol: str) -> Dict:
        """Analyze news sentiment"""
        try:
            # Fetch news from multiple sources
            news_data = await self._fetch_news(symbol)
            if not news_data:
                return {'score': 0, 'reasoning': ["No recent news data"]}
            
            # Analyze sentiment for each article
            sentiments = []
            important_news = []
            
            for article in news_data:
                sentiment = self._analyze_article(article)
                sentiments.append(sentiment['score'])
                
                if abs(sentiment['score']) > 0.5:  # Important news
                    important_news.append(sentiment['headline'])
            
            # Calculate aggregate sentiment
            avg_sentiment = np.mean(sentiments) if sentiments else 0
            
            return {
                'score': avg_sentiment,
                'reasoning': [
                    f"News sentiment: {avg_sentiment:.2f}",
                    *[f"Important: {news}" for news in important_news[:3]]
                ]
            }
            
        except Exception as e:
            logger.error(f"News analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["News analysis failed"]}
    
    def _analyze_article(self, article: Dict) -> Dict:
        """Analyze single article sentiment"""
        try:
            # Extract relevant text
            text = f"{article['title']} {article['summary']}"
            
            # Get NLP sentiment
            sentiment = self.nlp_model(text)[0]
            
            # Convert to score (-1 to 1)
            score = sentiment['score'] * (1 if sentiment['label'] == 'POSITIVE' else -1)
            
            return {
                'score': score,
                'headline': article['title'],
                'confidence': sentiment['score']
            }
            
        except Exception as e:
            logger.error(f"Article analysis error: {str(e)}")
            return {'score': 0, 'headline': '', 'confidence': 0}

class SocialMediaAnalyzer:
    """Social media sentiment analysis"""
    
    def __init__(self):
        self.nlp_model = self._load_nlp_model()
        self.social_cache = ExpiringDict(max_len=100, max_age_seconds=1800)
    
    async def analyze(self, symbol: str) -> Dict:
        """Analyze social media sentiment"""
        try:
            # Get data from different platforms
            twitter_data = await self._fetch_twitter_data(symbol)
            reddit_data = await self._fetch_reddit_data(symbol)
            
            # Analyze each platform
            twitter_sentiment = self._analyze_platform_data(twitter_data)
            reddit_sentiment = self._analyze_platform_data(reddit_data)
            
            # Weight and combine
            weighted_score = (
                twitter_sentiment['score'] * 0.6 +
                reddit_sentiment['score'] * 0.4
            )
            
            return {
                'score': weighted_score,
                'reasoning': [
                    f"Twitter sentiment: {twitter_sentiment['score']:.2f}",
                    f"Reddit sentiment: {reddit_sentiment['score']:.2f}",
                    *twitter_sentiment.get('highlights', []),
                    *reddit_sentiment.get('highlights', [])
                ]
            }
            
        except Exception as e:
            logger.error(f"Social media analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Social analysis failed"]}
    
    def _analyze_platform_data(self, data: List[Dict]) -> Dict:
        """Analyze sentiment for a social media platform"""
        if not data:
            return {'score': 0, 'highlights': []}
        
        sentiments = []
        highlights = []
        
        for post in data:
            sentiment = self._analyze_post(post)
            sentiments.append(sentiment['score'])
            
            if abs(sentiment['score']) > 0.7:  # Significant sentiment
                highlights.append(f"High impact post: {sentiment['summary']}")
        
        return {
            'score': np.mean(sentiments),
            'highlights': highlights[:3]  # Top 3 significant posts
        }
    
    def _analyze_post(self, post: Dict) -> Dict:
        """Analyze single social media post"""
        try:
            # Clean and preprocess text
            text = self._preprocess_social_text(post['text'])
            
            # Get sentiment
            sentiment = self.nlp_model(text)[0]
            
            # Calculate engagement weight
            engagement = self._calculate_engagement_score(post)
            
            # Weighted sentiment score
            score = sentiment['score'] * (1 if sentiment['label'] == 'POSITIVE' else -1)
            weighted_score = score * engagement
            
            return {
                'score': weighted_score,
                'summary': text[:100] + "...",
                'engagement': engagement
            }
            
        except Exception as e:
            logger.error(f"Post analysis error: {str(e)}")
            return {'score': 0, 'summary': '', 'engagement': 0}
    
    def _calculate_engagement_score(self, post: Dict) -> float:
        """Calculate post engagement score"""
        likes = post.get('likes', 0)
        replies = post.get('replies', 0)
        shares = post.get('shares', 0)
        
        # Weight different engagement types
        engagement = (likes + replies * 2 + shares * 3) / 100
        return min(engagement, 1.0)  # Cap at 1.0

class OnChainAnalyzer:
    """On-chain metrics analysis"""
    
    def __init__(self):
        self.client = self._initialize_client()
        self.cache = ExpiringDict(max_len=100, max_age_seconds=900)
    
    async def analyze(self, symbol: str) -> Dict:
        """Analyze on-chain metrics"""
        try:
            # Get on-chain data
            metrics = await self._fetch_chain_metrics(symbol)
            if not metrics:
                return {'score': 0, 'reasoning': ["No on-chain data available"]}
            
            # Analyze different aspects
            exchange_flow = self._analyze_exchange_flows(metrics)
            whale_activity = self._analyze_whale_activity(metrics)
            holder_distribution = self._analyze_holder_distribution(metrics)
            
            # Combine metrics
            sentiment_score = (
                exchange_flow['score'] * 0.4 +
                whale_activity['score'] * 0.4 +
                holder_distribution['score'] * 0.2
            )
            
            return {
                'score': sentiment_score,
                'reasoning': [
                    *exchange_flow['reasoning'],
                    *whale_activity['reasoning'],
                    *holder_distribution['reasoning']
                ]
            }
            
        except Exception as e:
            logger.error(f"On-chain analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["On-chain analysis failed"]}
    
    def _analyze_exchange_flows(self, metrics: Dict) -> Dict:
        """Analyze exchange inflow/outflow"""
        try:
            inflow = metrics['exchange_inflow']
            outflow = metrics['exchange_outflow']
            
            # Calculate net flow
            net_flow = outflow - inflow
            flow_ratio = net_flow / ((inflow + outflow) / 2)
            
            # Score based on flow ratio
            if abs(flow_ratio) < 0.1:
                score = 0
                reason = "Neutral exchange flows"
            elif flow_ratio > 0:
                score = min(flow_ratio, 1.0)
                reason = "Net outflow from exchanges (bullish)"
            else:
                score = max(flow_ratio, -1.0)
                reason = "Net inflow to exchanges (bearish)"
            
            return {
                'score': score,
                'reasoning': [f"{reason} ({flow_ratio:.2%})"]
            }
            
        except Exception as e:
            logger.error(f"Exchange flow analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Flow analysis failed"]}
    
    def _analyze_whale_activity(self, metrics: Dict) -> Dict:
        """Analyze whale wallet activity"""
        try:
            whale_accumulation = metrics['whale_accumulation']
            large_txs = metrics['large_transactions']
            
            # Analyze whale behavior
            if whale_accumulation > 0.1:
                score = min(whale_accumulation, 1.0)
                reason = "Whales accumulating"
            elif whale_accumulation < -0.1:
                score = max(whale_accumulation, -1.0)
                reason = "Whales distributing"
            else:
                score = 0
                reason = "Neutral whale activity"
            
            return {
                'score': score,
                'reasoning': [
                    f"{reason} ({whale_accumulation:.2%})",
                    f"Large transactions: {large_txs}"
                ]
            }
            
        except Exception as e:
            logger.error(f"Whale activity analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Whale analysis failed"]}
    
    def _analyze_holder_distribution(self, metrics: Dict) -> Dict:
        """Analyze holder distribution changes"""
        try:
            holder_change = metrics['holder_distribution_change']
            
            if abs(holder_change) < 0.02:
                score = 0
                reason = "Stable holder distribution"
            else:
                score = min(max(holder_change, -1.0), 1.0)
                reason = "Increasing holders" if holder_change > 0 else "Decreasing holders"
            
            return {
                'score': score,
                'reasoning': [f"{reason} ({holder_change:.2%})"]
            }
            
        except Exception as e:
            logger.error(f"Holder analysis error: {str(e)}")
            return {'score': 0, 'reasoning': ["Holder analysis failed"]}

class MarketRegimeAnalyzer:
    """Advanced market regime detection"""
    
    def __init__(self):
        self.model = self._initialize_model()
        self.window_size = 30
        
    def analyze(self, data: pd.DataFrame) -> Dict:
        """Detect and analyze market regime"""
        try:
            # Extract regime features
            features = self._extract_regime_features(data)
            
            # Detect current regime
            regime = self._detect_regime(features)
            
            # Analyze regime characteristics
            characteristics = self._analyze_regime_characteristics(
                data,
                regime,
                features
            )
            
            # Generate trading implications
            implications = self._generate_trading_implications(
                regime,
                characteristics
            )
            
            return {
                'regime': regime,
                'characteristics': characteristics,
                'implications': implications,
                'confidence': self._calculate_regime_confidence(features)
            }
            
        except Exception as e:
            logger.error(f"Regime analysis error: {str(e)}")
            return {
                'regime': 'unknown',
                'characteristics': {},
                'implications': {},
                'confidence': 0
            }

    def _extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        df = data.copy()
        
        # Return characteristics
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Volatility metrics
        df['volatility'] = df['returns'].rolling(20).std()
        df['relative_volatility'] = df['volatility'] / df['volatility'].rolling(100).mean()
        
        # Trend metrics
        for period in [20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'trend_strength_{period}'] = (
                (df['close'] - df[f'sma_{period}']) / 
                df[f'sma_{period}']
            ).abs()
        
        # Volume characteristics
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Momentum features
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
        
        # Pattern characteristics
        df['bb_width'] = self._calculate_bb_width(df)
        df['atr_ratio'] = self._calculate_atr_ratio(df)
        
        return df.dropna()
    
    def _detect_regime(self, features: pd.DataFrame) -> str:
        """Detect current market regime"""
        last_row = features.iloc[-1]
        
        # Check for high volatility regime
        if last_row['relative_volatility'] > 1.5:
            if last_row['trend_strength_50'] > 0.05:
                return 'volatile_trending'
            return 'volatile_ranging'
        
        # Check for trending regime
        if last_row['trend_strength_50'] > 0.03:
            if last_row['relative_volume'] > 1.2:
                return 'strong_trend'
            return 'weak_trend'
        
        # Check for accumulation/distribution
        if last_row['relative_volume'] > 1.5:
            if last_row['mfi'] > 60:
                return 'accumulation'
            elif last_row['mfi'] < 40:
                return 'distribution'
        
        # Check for range-bound
        if last_row['bb_width'] < 0.1:
            return 'ranging_tight'
        elif last_row['bb_width'] < 0.2:
            return 'ranging_normal'
            
        return 'ranging_wide'
    
    def _analyze_regime_characteristics(
        self,
        data: pd.DataFrame,
        regime: str,
        features: pd.DataFrame
    ) -> Dict:
        """Analyze characteristics of current regime"""
        last_row = features.iloc[-1]
        
        characteristics = {
            'volatility': last_row['volatility'],
            'trend_strength': last_row['trend_strength_50'],
            'volume_profile': last_row['relative_volume'],
            'momentum': last_row['rsi'],
            'support_resistance': self._identify_key_levels(data),
            'duration': self._estimate_regime_duration(features, regime)
        }
        
        # Add regime-specific metrics
        if 'volatile' in regime:
            characteristics['volatility_structure'] = self._analyze_volatility_structure(features)
        elif 'trend' in regime:
            characteristics['trend_structure'] = self._analyze_trend_structure(features)
        elif regime in ['accumulation', 'distribution']:
            characteristics['volume_structure'] = self._analyze_volume_structure(features)
        
        return characteristics
    
    def _generate_trading_implications(
        self,
        regime: str,
        characteristics: Dict
    ) -> Dict:
        """Generate trading implications based on regime"""
        implications = {
            'position_size': self._suggest_position_size(regime, characteristics),
            'entry_strategy': self._suggest_entry_strategy(regime, characteristics),
            'exit_strategy': self._suggest_exit_strategy(regime, characteristics),
            'risk_management': self._suggest_risk_management(regime, characteristics),
            'timeframe': self._suggest_timeframe(regime)
        }
        
        # Add regime-specific suggestions
        if 'volatile' in regime:
            implications.update(self._volatile_regime_suggestions(characteristics))
        elif 'trend' in regime:
            implications.update(self._trend_regime_suggestions(characteristics))
        else:
            implications.update(self._range_regime_suggestions(characteristics))
        
        return implications
    
    def _suggest_position_size(self, regime: str, characteristics: Dict) -> float:
        """Suggest position size based on regime"""
        base_size = 1.0  # Base position size
        
        # Adjust for volatility
        vol_factor = 1.0 / (1.0 + characteristics['volatility'])
        
        # Adjust for regime
        regime_factors = {
            'strong_trend': 1.0,
            'weak_trend': 0.8,
            'volatile_trending': 0.6,
            'volatile_ranging': 0.4,
            'ranging_normal': 0.7,
            'ranging_tight': 0.9,
            'ranging_wide': 0.5,
            'accumulation': 0.8,
            'distribution': 0.6
        }
        
        regime_factor = regime_factors.get(regime, 0.5)
        
        return base_size * vol_factor * regime_factor
    
    def _suggest_entry_strategy(self, regime: str, characteristics: Dict) -> Dict:
        """Suggest entry strategy based on regime"""
        if 'trend' in regime:
            return {
                'method': 'trend_following',
                'setup': ['breakout', 'pullback'],
                'confirmation': ['volume', 'momentum']
            }
        elif 'volatile' in regime:
            return {
                'method': 'mean_reversion',
                'setup': ['oversold', 'overbought'],
                'confirmation': ['volume', 'volatility_contraction']
            }
        else:
            return {
                'method': 'range_trading',
                'setup': ['support', 'resistance'],
                'confirmation': ['volume', 'price_action']
            }
    
    def _suggest_exit_strategy(self, regime: str, characteristics: Dict) -> Dict:
        """Suggest exit strategy based on regime"""
        strategy = {
            'take_profit': self._calculate_take_profit_levels(regime, characteristics),
            'stop_loss': self._calculate_stop_loss_levels(regime, characteristics),
            'trailing_stop': self._should_use_trailing_stop(regime),
            'time_stop': self._calculate_time_stop(regime)
        }
        
        if 'volatile' in regime:
            strategy['volatility_stops'] = True
        
        return strategy
    
    def _suggest_risk_management(self, regime: str, characteristics: Dict) -> Dict:
        """Suggest risk management parameters"""
        return {
            'position_size': self._suggest_position_size(regime, characteristics),
            'max_risk_per_trade': self._calculate_max_risk(regime),
            'correlation_hedging': self._suggest_hedging(regime),
            'stop_types': self._suggest_stop_types(regime)
        }
    
    def _suggest_timeframe(self, regime: str) -> str:
        """Suggest best timeframe for trading"""
        timeframes = {
            'strong_trend': '4h',
            'weak_trend': '1h',
            'volatile_trending': '15m',
            'volatile_ranging': '5m',
            'ranging_normal': '1h',
            'ranging_tight': '15m',
            'ranging_wide': '4h',
            'accumulation': '4h',
            'distribution': '1h'
        }
        return timeframes.get(regime, '1h')
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        try:
            # Calculate fractals
            highs = argrelextrema(data['high'].values, np.greater)[0]
            lows = argrelextrema(data['low'].values, np.less)[0]
            
            # Get recent levels
            recent_highs = data['high'].iloc[highs[-5:]].values
            recent_lows = data['low'].iloc[lows[-5:]].values
            
            # Cluster levels
            resistance_levels = self._cluster_levels(recent_highs)
            support_levels = self._cluster_levels(recent_lows)
            
            return {
                'resistance': resistance_levels,
                'support': support_levels,
                'current_range': resistance_levels[0] - support_levels[0]
            }
            
        except Exception as e:
            logger.error(f"Key levels identification error: {str(e)}")
            return {'resistance': [], 'support': [], 'current_range': 0}
    
    def _cluster_levels(self, levels: np.ndarray, tolerance: float = 0.02) -> List[float]:
        """Cluster similar price levels"""
        if len(levels) == 0:
            return []
            
        clusters = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - np.mean(current_cluster)) / np.mean(current_cluster) < tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return sorted(clusters, reverse=True)
    
    def _estimate_regime_duration(
        self,
        features: pd.DataFrame,
        current_regime: str
    ) -> int:
        """Estimate how long the current regime has lasted"""
        try:
            # Get regime history
            regime_history = [
                self._detect_regime(features.iloc[:i+1])
                for i in range(len(features)-30, len(features))
            ]
            
            # Count consecutive occurrences of current regime
            duration = 0
            for regime in reversed(regime_history):
                if regime == current_regime:
                    duration += 1
                else:
                    break
                    
            return duration
            
        except Exception as e:
            logger.error(f"Regime duration estimation error: {str(e)}")
            return 0
    
    def _calculate_regime_confidence(self, features: pd.DataFrame) -> float:
        """Calculate confidence in regime detection"""
        try:
            last_row = features.iloc[-1]
            
            # Base confidence on feature clarity
            confidences = []
            
            # Volatility confidence
            vol_confidence = min(
                abs(last_row['relative_volatility'] - 1) * 2,
                1.0
            )
            confidences.append(vol_confidence)
            
            # Trend confidence
            trend_confidence = min(
                last_row['trend_strength_50'] * 20,
                1.0
            )
            confidences.append(trend_confidence)
            
            # Volume confidence
            volume_confidence = min(
                abs(last_row['relative_volume'] - 1) * 2,
                1.0
            )
            confidences.append(volume_confidence)
            
            # Range confidence
            range_confidence = min(
                abs(last_row['bb_width'] - 0.15) / 0.15,
                1.0
            )
            confidences.append(range_confidence)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]
            final_confidence = sum(c * w for c, w in zip(confidences, weights))
            
            return min(final_confidence * 100, 100)
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {str(e)}")
            return 50.0  # Default moderate confidence

class WhaleActivityAnalyzer:
    """Enhanced whale activity analyzer with on-chain monitoring""" 
    def __init__(self, config: Config):
        self.config = config
        self.data_fetcher = OnChainDataFetcher(config)
        self.known_whales = set()
        self.historical_movements = []
        self.logger = logging.getLogger(__name__)
    
    async def analyze_whale_activity(
        self,
        symbol: str,
        timeframe: str = '1h'
    ) -> Dict:
        """Analyze recent whale activities with on-chain data"""
        try:
            # Get token details
            token_info = await self._get_token_info(symbol)
            if not token_info:
                return self._get_empty_analysis()
            
            # Fetch on-chain data
            recent_transfers = await self.data_fetcher.fetch_token_transfers(
                token_info['address'],
                token_info['chain']
            )
            
            whale_wallets = await self.data_fetcher.get_whale_wallets(
                token_info['address'],
                token_info['chain'],
                self.config.min_whale_size
            )
            
            # Update known whales
            self.known_whales.update(whale_wallets)
            
            # Analyze movements
            analysis = await self._analyze_transfers(recent_transfers, whale_wallets)
            
            # Generate insights
            insights = self._generate_whale_insights(analysis)
            
            return {
                'impact_score': analysis['impact_score'],
                'whale_count': len(whale_wallets),
                'total_volume': analysis['total_volume'],
                'buy_pressure': analysis['buy_pressure'],
                'sell_pressure': analysis['sell_pressure'],
                'significant_moves': analysis['significant_moves'],
                'whale_confidence': analysis['confidence'],
                'insights': insights
            }
            
        except Exception as e:
            self.logger.error(f"Whale analysis failed: {str(e)}")
            return self._get_empty_analysis()
    
    async def _get_token_info(self, symbol: str) -> Optional[Dict]:
        """Get token contract details"""
        try:
            # Remove trading pair separator if present
            base_symbol = symbol.split('/')[0].lower()
            
            # Check cache first
            cache_key = f"token_info_{base_symbol}"
            cached = self.data_fetcher.cache.get(cache_key)
            if cached:
                return cached
            
            # Fetch from sources
            async with aiohttp.ClientSession() as session:
                # Try Ethereum first
                eth_url = f"https://api.etherscan.io/api?module=token&action=tokeninfo&contractaddress={base_symbol}"
                async with session.get(eth_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == '1':
                            token = {
                                'address': data['result']['contractAddress'],
                                'chain': 'ethereum',
                                'decimals': int(data['result']['decimals'])
                            }
                            self.data_fetcher.cache.set(cache_key, token)
                            return token
                
                # Try BSC
                bsc_url = f"https://api.bscscan.com/api?module=token&action=tokeninfo&contractaddress={base_symbol}"
                async with session.get(bsc_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == '1':
                            token = {
                                'address': data['result']['contractAddress'],
                                'chain': 'bsc',
                                'decimals': int(data['result']['decimals'])
                            }
                            self.data_fetcher.cache.set(cache_key, token)
                            return token
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting token info: {str(e)}")
            return None
    
    async def _analyze_transfers(
        self,
        transfers: List[Dict],
        whale_wallets: Set[str]
    ) -> Dict:
        """Analyze whale transfers"""
        if not transfers:
            return self._get_empty_analysis()
        
        total_volume = sum(t['amount'] for t in transfers)
        buy_volume = sum(
            t['amount'] for t in transfers
            if t['to'] in whale_wallets
        )
        sell_volume = sum(
            t['amount'] for t in transfers
            if t['from'] in whale_wallets
        )
        
        # Calculate net pressure
        net_pressure = (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
        
        # Find significant moves
        significant_moves = [
            t for t in transfers
            if t['amount'] >= self.config.min_whale_size * 2  # Extra large moves
        ]
        
        # Calculate confidence based on participation and consensus
        whale_participation = len(set(
            t['from'] for t in transfers if t['from'] in whale_wallets
        ) | set(
            t['to'] for t in transfers if t['to'] in whale_wallets
        )) / len(whale_wallets) if whale_wallets else 0
        
        direction_consensus = abs(net_pressure)
        confidence = (whale_participation + direction_consensus) / 2 * 100
        
        return {
            'impact_score': net_pressure,
            'total_volume': total_volume,
            'buy_pressure': buy_volume,
            'sell_pressure': sell_volume,
            'significant_moves': significant_moves,
            'confidence': confidence
        }
    
    def _generate_whale_insights(self, analysis: Dict) -> List[str]:
        """Generate insights from whale activity"""
        insights = []
        
        # Activity level
        if analysis['total_volume'] > 0:
            if abs(analysis['impact_score']) > 0.3:
                direction = "accumulation" if analysis['impact_score'] > 0 else "distribution"
                insights.append(f"Strong whale {direction} detected")
                
                if analysis['confidence'] > 70:
                    insights.append("High confidence in whale direction")
            
            # Significant moves
            if analysis['significant_moves']:
                moves = len(analysis['significant_moves'])
                insights.append(f"Detected {moves} large whale transactions")
        else:
            insights.append("No significant whale activity detected")
        
        return insights
    
    def _get_empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'impact_score': 0,
            'whale_count': 0,
            'total_volume': 0,
            'buy_pressure': 0,
            'sell_pressure': 0,
            'significant_moves': [],
            'whale_confidence': 0,
            'insights': ["No whale activity data available"]
        }

    async def cleanup(self):
        if hasattr(self, 'data_fetcher'):
            await self.data_fetcher.cleanup()

class PatternRecognizer:
    """Recognizes technical chart patterns"""
    
    def __init__(self):
        self.window_size = 20  # Default window for pattern detection
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """Analyze price data for patterns"""
        patterns = []
        
        # Check various patterns
        if self._check_double_bottom(data):
            patterns.append(('double_bottom', 'Buy'))
        if self._check_double_top(data):
            patterns.append(('double_top', 'Sell'))
        if self._check_head_shoulders(data):
            patterns.append(('head_shoulders', 'Sell'))
        if self._check_inverse_head_shoulders(data):
            patterns.append(('inverse_head_shoulders', 'Buy'))
            
        # Check triangle patterns
        triangle = self._check_triangle(data)
        if triangle:
            if triangle == 'ascending':
                patterns.append(('ascending_triangle', 'Buy'))
            elif triangle == 'descending':
                patterns.append(('descending_triangle', 'Sell'))
            else:
                patterns.append(('symmetrical_triangle', 'Hold'))
        
        return {
            'patterns': patterns,
            'confidence': self._calculate_pattern_confidence(patterns, data) if patterns else 0,
            'action': self._get_consensus_action(patterns) if patterns else 'Hold'
        }
    
    def _check_double_bottom(self, data: pd.DataFrame) -> bool:
        """Check for double bottom pattern"""
        window = data.tail(self.window_size)
        lows = argrelextrema(window['low'].values, np.less)[0]
        
        if len(lows) < 2:
            return False
            
        # Get last two lows
        last_two_lows = window['low'].iloc[lows[-2:]]
        low_diff = abs(last_two_lows.iloc[0] - last_two_lows.iloc[1])
        
        # Check if lows are at similar levels
        if low_diff / last_two_lows.iloc[0] < 0.02:  # 2% tolerance
            # Check for higher high between lows
            between_highs = window['high'].iloc[lows[-2]:lows[-1]]
            if max(between_highs) > window['high'].iloc[-1]:
                return True
                
        return False
    
    def _check_double_top(self, data: pd.DataFrame) -> bool:
        """Check for double top pattern"""
        window = data.tail(self.window_size)
        highs = argrelextrema(window['high'].values, np.greater)[0]
        
        if len(highs) < 2:
            return False
            
        # Get last two highs
        last_two_highs = window['high'].iloc[highs[-2:]]
        high_diff = abs(last_two_highs.iloc[0] - last_two_highs.iloc[1])
        
        # Check if highs are at similar levels
        if high_diff / last_two_highs.iloc[0] < 0.02:  # 2% tolerance
            # Check for lower low between highs
            between_lows = window['low'].iloc[highs[-2]:highs[-1]]
            if min(between_lows) < window['low'].iloc[-1]:
                return True
                
        return False

class PredictionConsensus:
    """Generate consensus from multiple prediction sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.recent_predictions = []
        self.max_history = 1000
    
    def generate_consensus(self, predictions: List[Dict]) -> Dict:
        """Generate weighted consensus from multiple predictions"""
        try:
            # Validate predictions
            if not predictions:
                raise ValueError("No predictions provided")
            
            # Calculate initial consensus
            consensus = self._calculate_weighted_consensus(predictions)
            
            # Check for strong disagreement
            if self._has_strong_disagreement(predictions):
                consensus['confidence'] *= 0.8
                consensus['reasoning'].append("Reduced confidence due to agent disagreement")
            
            # Add historical context
            historical_bias = self._check_historical_bias()
            if historical_bias:
                consensus['reasoning'].append(
                    f"Historical bias detected: {historical_bias}"
                )
            
            # Store prediction
            self._store_prediction(consensus)
            
            return consensus
            
        except Exception as e:
            logger.error(f"Consensus generation error: {str(e)}")
            return {
                'action': "Hold",
                'confidence': 0,
                'reasoning': ["Consensus generation failed"]
            }
    
    def _calculate_weighted_consensus(self, predictions: List[Dict]) -> Dict:
        """Calculate weighted consensus from predictions"""
        action_scores = {'Buy': 0.0, 'Sell': 0.0, 'Hold': 0.0}
        total_weight = 0
        all_reasoning = []
        
        for pred in predictions:
            weight = self.config.agent_weights.get(pred['agent'], 1.0)
            confidence = pred['confidence'] / 100.0
            
            # Apply weight to action
            action_scores[pred['action']] += weight * confidence
            total_weight += weight
            
            # Collect reasoning
            if 'reasoning' in pred:
                all_reasoning.extend(pred['reasoning'])
        
        # Normalize scores
        if total_weight > 0:
            action_scores = {
                action: score/total_weight 
                for action, score in action_scores.items()
            }
        
        # Get final action
        final_action = max(action_scores.items(), key=lambda x: x[1])[0]
        confidence = action_scores[final_action] * 100
        
        return {
            'action': final_action,
            'confidence': confidence,
            'action_scores': action_scores,
            'reasoning': all_reasoning
        }
    
    def _has_strong_disagreement(self, predictions: List[Dict]) -> bool:
        """Check for strong disagreement between agents"""
        actions = [pred['action'] for pred in predictions]
        confident_predictions = [
            pred for pred in predictions 
            if pred['confidence'] > 70
        ]
        
        if len(confident_predictions) >= 2:
            confident_actions = set(pred['action'] for pred in confident_predictions)
            return len(confident_actions) > 1
        
        return False
    
    def _store_prediction(self, consensus: Dict):
        """Store prediction in history"""
        self.recent_predictions.append({
            'timestamp': datetime.datetime.now(timezone.utc),
            'action': consensus['action'],
            'confidence': consensus['confidence']
        })
        
        # Maintain history size
        if len(self.recent_predictions) > self.max_history:
            self.recent_predictions.pop(0)
    
    def _check_historical_bias(self) -> Optional[str]:
        """Check for bias in recent predictions"""
        if len(self.recent_predictions) < 10:
            return None
        
        recent_actions = [pred['action'] for pred in self.recent_predictions[-10:]]
        buy_ratio = recent_actions.count('Buy') / len(recent_actions)
        sell_ratio = recent_actions.count('Sell') / len(recent_actions)
        
        if buy_ratio > 0.8:
            return "Strong bullish bias"
        elif sell_ratio > 0.8:
            return "Strong bearish bias"
            
        return None

class StrategyGenerator:
    """Generate sophisticated trading strategies based on analysis"""
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_manager = RiskManager(config)
    
    def generate_strategy(
        self,
        consensus: Dict,
        market_data: pd.DataFrame,
        analysis: Dict
    ) -> Dict:
        """Generate complete trading strategy"""
        try:
            # Skip if no clear consensus
            if consensus['confidence'] < 60:
                return self._generate_hold_strategy(
                    "Low confidence in signals",
                    consensus['reasoning']
                )
            
            # Get current market state
            current_price = market_data['close'].iloc[-1]
            
            # Generate entry strategy
            entry_strategy = self._generate_entry_strategy(
                consensus['action'],
                current_price,
                analysis
            )
            
            # Generate exit strategy
            exit_strategy = self._generate_exit_strategy(
                entry_strategy,
                analysis
            )
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                entry_strategy,
                exit_strategy,
                analysis['risk_assessment']
            )
            
            # Generate monitoring rules
            monitoring_rules = self._generate_monitoring_rules(
                analysis,
                exit_strategy
            )
            
            return {
                'action': consensus['action'],
                'confidence': consensus['confidence'],
                'entry': entry_strategy,
                'exit': exit_strategy,
                'position_size': position_size,
                'monitoring_rules': monitoring_rules,
                'reasoning': consensus['reasoning']
            }
            
        except Exception as e:
            logger.error(f"Strategy generation error: {str(e)}")
            return self._generate_hold_strategy(
                "Strategy generation failed",
                [str(e)]
            )
    
    def _generate_entry_strategy(
        self,
        action: str,
        current_price: float,
        analysis: Dict
    ) -> Dict:
        """Generate sophisticated entry strategy"""
        volatility = analysis['volatility']['atr']
        volume_profile = analysis['volume_analysis']
        support_resistance = analysis['support_resistance']
        
        # Calculate base entry points
        if action == "Buy":
            main_entry = current_price
            entry_points = [
                main_entry,
                main_entry * 0.99,  # 1% below
                main_entry * 0.98   # 2% below
            ]
            
            # Adjust based on support levels
            for support in support_resistance['support_levels']:
                if support < main_entry and support > main_entry * 0.97:
                    entry_points.append(support)
                    
        else:  # Sell
            main_entry = current_price
            entry_points = [
                main_entry,
                main_entry * 1.01,  # 1% above
                main_entry * 1.02   # 2% above
            ]
            
            # Adjust based on resistance levels
            for resistance in support_resistance['resistance_levels']:
                if resistance > main_entry and resistance < main_entry * 1.03:
                    entry_points.append(resistance)
        
        # Sort and deduplicate entry points
        entry_points = sorted(set(entry_points))
        
        # Calculate allocations for each point
        allocations = self._calculate_entry_allocations(
            len(entry_points),
            volume_profile['volume_intensity']
        )
        
        return {
            'method': 'Scaled' if len(entry_points) > 1 else 'Single',
            'points': {
                'price_levels': entry_points,
                'allocation': allocations
            },
            'conditions': self._generate_entry_conditions(action, analysis),
            'validity': '24h'  # Default validity period
        }
    
    def _generate_exit_strategy(
        self,
        entry_strategy: Dict,
        analysis: Dict
    ) -> Dict:
        """Generate comprehensive exit strategy"""
        entry_points = entry_strategy['points']['price_levels']
        avg_entry = sum(entry_points) / len(entry_points)
        
        # Calculate profit targets based on volatility and trend strength
        volatility = analysis['volatility']['atr']
        trend_strength = analysis['trend']['strength']
        
        # Adjust profit targets based on market conditions
        if trend_strength > 2:  # Strong trend
            profit_targets = [
                1.5 * volatility,
                2.5 * volatility,
                4.0 * volatility
            ]
        else:  # Normal market
            profit_targets = [
                1.0 * volatility,
                2.0 * volatility,
                3.0 * volatility
            ]
        
        # Calculate take profit levels
        if entry_strategy['method'] == 'Buy':
            tp_levels = [avg_entry * (1 + target) for target in profit_targets]
            sl_level = avg_entry * (1 - volatility)
        else:
            tp_levels = [avg_entry * (1 - target) for target in profit_targets]
            sl_level = avg_entry * (1 + volatility)
        
        return {
            'take_profit': {
                'levels': tp_levels,
                'allocations': [0.4, 0.3, 0.3]  # Progressive profit taking
            },
            'stop_loss': {
                'main_stop': sl_level,
                'trailing_stop': {
                    'activation': f"{(volatility * 100):.1f}%",
                    'distance': f"{(volatility * 0.5 * 100):.1f}%"
                }
            },
            'time_based_exits': self._generate_time_based_exits(analysis)
        }
    
    def _calculate_entry_allocations(
        self,
        num_points: int,
        volume_intensity: str
    ) -> List[float]:
        """Calculate allocations for entry points"""
        if num_points == 1:
            return [1.0]
        
        if volume_intensity in ['High', 'Very High']:
            # More aggressive entry on high volume
            weights = [0.6, 0.3, 0.1]
        else:
            # More conservative entry on low volume
            weights = [0.4, 0.3, 0.3]
        
        # Adjust weights to match number of points
        weights = weights[:num_points]
        total = sum(weights)
        return [w/total for w in weights]
    
    def _generate_entry_conditions(
        self,
        action: str,
        analysis: Dict
    ) -> List[str]:
        """Generate entry conditions"""
        conditions = []
        
        # Volume conditions
        conditions.append(
            f"Volume above {analysis['volume_analysis']['volume_ma']:.2f}"
        )
        
        # Trend conditions
        if action == "Buy":
            conditions.extend([
                "No immediate resistance within 2%",
                "RSI not overbought",
                "Positive MACD momentum"
            ])
        else:
            conditions.extend([
                "No immediate support within 2%",
                "RSI not oversold",
                "Negative MACD momentum"
            ])
        
        return conditions
    
    def _generate_monitoring_rules(
        self,
        analysis: Dict,
        exit_strategy: Dict
    ) -> List[str]:
        """Generate monitoring rules"""
        return [
            "Monitor volume for divergences",
            f"Watch for trend reversal below {analysis['trend']['strength']}",
            "Track whale wallet movements",
            f"Monitor sentiment changes in {', '.join(analysis['sentiment']['sources'])}",
            "Check for unusual price action near S/R levels"
        ]
    
    def _generate_time_based_exits(self, analysis: Dict) -> Dict:
        """Generate time-based exit rules"""
        volatility_level = analysis['volatility']['volatility_level']
        
        if volatility_level == "High":
            max_hold = "24h"
            review_interval = "1h"
        elif volatility_level == "Medium":
            max_hold = "48h"
            review_interval = "4h"
        else:
            max_hold = "72h"
            review_interval = "6h"
        
        return {
            'max_hold_time': max_hold,
            'review_interval': review_interval,
            'extension_conditions': [
                "Strong trend continuation",
                "Volume supporting price action",
                "No reversal patterns"
            ]
        }
    
    def _generate_hold_strategy(
        self,
        reason: str,
        additional_reasons: List[str]
    ) -> Dict:
        """Generate hold strategy"""
        return {
            'action': "Hold",
            'confidence': 0,
            'entry': None,
            'exit': None,
            'position_size': 0,
            'reasoning': [reason] + additional_reasons
        }

class RiskManager:
    """Enhanced risk management with dynamic position sizing"""
    
    def __init__(self, config: Config):
        self.config = config
        self.position_history = []
        self.max_drawdown = 0.0
    
    def calculate_position_size(
        self,
        entry_strategy: Dict,
        exit_strategy: Dict,
        risk_assessment: Dict
    ) -> float:
        """Calculate optimal position size"""
        try:
            # Get base position size
            base_size = self._calculate_base_position(risk_assessment)
            
            # Calculate risk-adjusted size
            risk_adjusted_size = self._adjust_for_risk(
                base_size,
                entry_strategy,
                exit_strategy,
                risk_assessment
            )
            
            # Apply market impact adjustment
            final_size = self._adjust_for_market_impact(
                risk_adjusted_size,
                entry_strategy
            )
            
            # Enforce position limits
            return min(
                final_size,
                self.config.max_position_size
            )
            
        except Exception as e:
            logger.error(f"Position size calculation error: {str(e)}")
            return 0.0
    
    def _calculate_base_position(self, risk_assessment: Dict) -> float:
        """Calculate base position size"""
        risk_level = risk_assessment['risk_level']
        
        # Base size multipliers
        multipliers = {
            'Very Low': 1.0,
            'Low': 0.8,
            'Medium': 0.6,
            'High': 0.4,
            'Very High': 0.2
        }
        
        return self.config.max_position_size * multipliers.get(risk_level, 0.5)
    
    def _adjust_for_risk(
        self,
        base_size: float,
        entry_strategy: Dict,
        exit_strategy: Dict,
        risk_assessment: Dict
    ) -> float:
        """Adjust position size based on risk factors"""
        size = base_size
        
        # Calculate risk-reward ratio
        rr_ratio = self._calculate_risk_reward_ratio(
            entry_strategy,
            exit_strategy
        )
        
        # Adjust based on risk-reward
        if rr_ratio < 1.5:
            size *= 0.7
        elif rr_ratio > 3:
            size *= 1.2
        
        # Adjust for market conditions
        market_risk = risk_assessment['market_risk']
        size *= (1 - market_risk)
        
        # Consider current drawdown
        if self.max_drawdown > 0.1:  # 10% drawdown
            size *= 0.8
        
        return size
    
    def _adjust_for_market_impact(
        self,
        size: float,
        entry_strategy: Dict
    ) -> float:
        """Adjust position size for market impact"""
        # This would normally consider:
        # - Order book depth
        # - Recent volume
        # - Market liquidity
        # For this example, we'll use a simple adjustment
        if entry_strategy['method'] == 'Scaled':
            return size  # No adjustment needed for scaled entries
        return size * 0.9  # 10% reduction for single entries
    
    def _calculate_risk_reward_ratio(
        self,
        entry_strategy: Dict,
        exit_strategy: Dict
    ) -> float:
        """Calculate risk-reward ratio"""
        try:
            # Get average entry price
            entry_points = entry_strategy['points']['price_levels']
            avg_entry = sum(entry_points) / len(entry_points)
            
            # Get first take profit and stop loss
            tp_level = exit_strategy['take_profit']['levels'][0]
            sl_level = exit_strategy['stop_loss']['main_stop']
            
            # Calculate distances
            if entry_strategy['method'] == 'Buy':
                reward = tp_level - avg_entry
                risk = avg_entry - sl_level
            else:
                reward = avg_entry - tp_level
                risk = sl_level - avg_entry
            
            return reward / risk if risk > 0 else 0
            
        except Exception:
            return 1.0  # Default to 1:1 risk-reward
    
    def update_drawdown(self, profit_loss: float):
        """Update maximum drawdown"""
        self.max_drawdown = max(self.max_drawdown, -profit_loss)

class PerformanceMonitor:
    """Monitor and analyze strategy performance"""
    
    def __init__(self):
        self.trades = []
        self.performance_metrics = {}
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
    
    def add_trade(self, trade: Dict):
        """Add trade to history"""
        self.trades.append({
            'timestamp': datetime.datetime.now(timezone.utc),
            'action': trade['action'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'position_size': trade['position_size'],
            'profit_loss': trade['profit_loss'],
            'hold_time': trade['hold_time']
        })
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if not self.trades:
            return
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['profit_loss'] > 0])
        
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': profitable_trades / total_trades,
            'avg_profit': np.mean([t['profit_loss'] for t in self.trades]),
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'avg_hold_time': np.mean([t['hold_time'] for t in self.trades])
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of the strategy"""
        if not self.trades:
            return 0.0
        
        # Calculate returns
        returns = [t['profit_loss'] for t in self.trades]
        
        # Calculate excess returns over risk-free rate
        excess_returns = np.array(returns) - (self.config.risk_free_rate / 252)  # Daily adjustment
        
        # Calculate Sharpe ratio
        avg_excess_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns)
        
        if std_dev == 0:
            return 0.0
            
        # Annualize Sharpe ratio
        sharpe = (avg_excess_return / std_dev) * np.sqrt(252)  # Annualize
        
        return sharpe
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            'metrics': self.performance_metrics,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'recent_performance': self._calculate_recent_performance()
        }
    
    def _calculate_recent_performance(self) -> Dict:
        """Calculate recent trading performance"""
        if not self.trades:
            return {
                'win_rate_30d': 0.0,
                'avg_profit_30d': 0.0,
                'volatility_30d': 0.0
            }
        
        # Get trades from last 30 days
        thirty_days_ago = datetime.datetime.now(timezone.utc) - datetime.timedelta(days=30)
        recent_trades = [
            t for t in self.trades 
            if t['timestamp'] > thirty_days_ago
        ]
        
        if not recent_trades:
            return {
                'win_rate_30d': 0.0,
                'avg_profit_30d': 0.0,
                'volatility_30d': 0.0
            }
        
        # Calculate metrics
        profitable_trades = len([t for t in recent_trades if t['profit_loss'] > 0])
        total_trades = len(recent_trades)
        
        profits = [t['profit_loss'] for t in recent_trades]
        
        return {
            'win_rate_30d': profitable_trades / total_trades,
            'avg_profit_30d': np.mean(profits),
            'volatility_30d': np.std(profits)
        }

class CryptoAnalysisApp:
    """Enhanced main application class"""
    def __init__(self):
        self.config = Config()
        self.api_config = APIConfig()
        self.db_manager = DatabaseManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.ml_training_manager = MLTrainingManager(self.config, self.model_manager)
        self.secondary_manager = SecondaryDataManager(self.config, self.api_config)

        # Initialize components
        self.data_fetcher = EnhancedDataFetcher(self.config, self.api_config)
        self.agents = self._initialize_agents()
        self.consensus_engine = PredictionConsensus(self.config)
        self.strategy_generator = StrategyGenerator(self.config)
        self.backtester = BacktestEngine(self.config)
        
        # Initialize ML components for agents
        self._initialize_ml_components()
    
    def _initialize_agents(self) -> Dict:
        """Initialize AI agents"""
        return {
            'technical': TechnicalAnalysisAgent(
                weight=self.config.agent_weights['TechnicalAgent']
            ),
            'sentiment': SentimentAnalysisAgent(
                weight=self.config.agent_weights['SentimentAgent']
            ),
            'market_regime': MarketRegimeAgent(
                weight=self.config.agent_weights['MarketRegimeAgent']
            ),
            'whale': WhaleActivityAnalyzer(self.config)  # Pass config object instead of min_whale_size
        }
    
    def _initialize_ml_components(self):
        """Initialize ML components for agents"""
        for agent in self.agents.values():
            if hasattr(agent, 'initialize_ml_components'):
                agent.initialize_ml_components(
                    self.model_manager,
                    self.ml_training_manager
                )

    async def fetch_from_binance(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        try:
            exchange = ccxt.binance()
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe)
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
        except Exception as e:
            logger.error(f"Binance fetch failed: {str(e)}")
        return None

    async def fetch_from_secondary_sources(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        for source in ['cryptocompare', 'coingecko']:
            try:
                data = await self.data_sources[source].get_market_data(symbol, timeframe)
                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched from {source}")
                    return data
            except Exception as e:
                logger.error(f"{source} fetch failed: {str(e)}")
        return None

    async def analyze_market(self, symbol: str, timeframe: str = '1d') -> Dict:
        """Perform market analysis"""
        try:
            # Fetch market data with proper async handling
            market_data = await self.data_fetcher.fetch_market_data(symbol, timeframe)
            if market_data is None:
                raise DataError("Failed to fetch market data")

            # Process agent predictions
            predictions = []
            for agent_name, agent in self.agents.items():
                try:
                    prediction = await agent.analyze(market_data, {
                        'symbol': symbol,
                        'timeframe': timeframe
                    })
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    logger.error(f"{agent_name} analysis failed: {e}")

            # Generate trading signals
            consensus = self.consensus_engine.generate_consensus(predictions)
            strategy = self.strategy_generator.generate_strategy(
                consensus,
                market_data,
                {'predictions': predictions}
            )

            return {
                'market_data': market_data.to_dict(),
                'predictions': predictions,
                'consensus': consensus,
                'strategy': strategy
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    async def fetch_market_data(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Fetch market data with fallback sources"""
        try:
            # Try Binance first
            data = await self._fetch_from_binance(symbol, timeframe)
            if data is not None:
                return data

            # Try secondary sources
            data = await self._fetch_from_secondary(symbol, timeframe)
            if data is not None:
                return data

            raise DataError("No data available from any source")

        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            raise

    def _ensure_models_trained(self, market_data: pd.DataFrame):
        """Ensure all models are trained"""
        for agent in self.agents.values():
            if hasattr(agent, 'train'):
                try:
                    if self.model_manager.needs_training(agent.name):
                        agent.train(market_data)
                except Exception as e:
                    logger.error(f"Training failed for {agent.name}: {str(e)}")

    async def cleanup(self):
        """Cleanup all resources"""
        cleanup_tasks = []
        
        # Cleanup data fetchers
        if hasattr(self, 'data_fetcher'):
            cleanup_tasks.append(self.data_fetcher.cleanup())
        if hasattr(self, 'secondary_manager'):
            cleanup_tasks.append(self.secondary_manager.cleanup())
            
        # Cleanup agents
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                cleanup_tasks.append(agent.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

class OnChainDataFetcher:
    """Handles on-chain data fetching from multiple chains"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connections = self._initialize_blockchain_connections()
        self.cache = ExpiringCache(max_age_seconds=300)  # 5 minute cache
        
    def _initialize_blockchain_connections(self) -> Dict[str, Web3]:
        """Initialize connections to different blockchains"""
        connections = {}
        
        # Ethereum Mainnet
        eth_url = os.getenv('ETH_NODE_URL', 'https://mainnet.infura.io/v3/YOUR-PROJECT-ID')
        connections['ethereum'] = Web3(Web3.HTTPProvider(eth_url))
        
        # BSC
        bsc_url = os.getenv('BSC_NODE_URL', 'https://bsc-dataseed.binance.org/')
        connections['bsc'] = Web3(Web3.HTTPProvider(bsc_url))
        
        return connections
    
    async def fetch_token_transfers(
        self,
        token_address: str,
        chain: str,
        blocks_back: int = 1000
    ) -> List[Dict]:
        """Fetch recent token transfers"""
        try:
            cache_key = f"transfers_{chain}_{token_address}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            web3 = self.connections[chain]
            current_block = await web3.eth.block_number
            
            # Get token contract
            token_contract = web3.eth.contract(
                address=Web3.toChecksumAddress(token_address),
                abi=ERC20_ABI
            )
            
            # Get transfer events
            transfer_filter = token_contract.events.Transfer.createFilter(
                fromBlock=current_block - blocks_back
            )
            events = transfer_filter.get_all_entries()
            
            # Process events
            transfers = []
            for event in events:
                amount = event.args.value / (10 ** await token_contract.functions.decimals().call())
                if amount >= self.config.min_whale_size:
                    transfers.append({
                        'from': event.args['from'],
                        'to': event.args.to,
                        'amount': amount,
                        'block': event.blockNumber,
                        'tx_hash': event.transactionHash.hex(),
                        'timestamp': web3.eth.getBlock(event.blockNumber).timestamp
                    })
            
            self.cache.set(cache_key, transfers)
            return transfers
            
        except Exception as e:
            logger.error(f"Error fetching transfers: {str(e)}")
            return []
    
    async def get_whale_wallets(
        self,
        token_address: str,
        chain: str,
        min_balance: float
    ) -> Set[str]:
        """Get wallets with significant token holdings"""
        try:
            cache_key = f"whales_{chain}_{token_address}"
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            web3 = self.connections[chain]
            token_contract = web3.eth.contract(
                address=Web3.toChecksumAddress(token_address),
                abi=ERC20_ABI
            )
            
            # Get Transfer events to identify holders
            transfer_filter = token_contract.events.Transfer.createFilter(
                fromBlock=0
            )
            events = transfer_filter.get_all_entries()
            
            # Track balances
            balances = {}
            for event in events:
                from_addr = event.args['from']
                to_addr = event.args.to
                amount = event.args.value / (10 ** await token_contract.functions.decimals().call())
                
                balances[from_addr] = balances.get(from_addr, 0) - amount
                balances[to_addr] = balances.get(to_addr, 0) + amount
            
            # Filter whale wallets
            whales = {
                addr for addr, balance in balances.items()
                if balance >= min_balance
                and addr != Web3.toChecksumAddress('0x0000000000000000000000000000000000000000')
            }
            
            self.cache.set(cache_key, whales)
            return whales
            
        except Exception as e:
            logger.error(f"Error getting whale wallets: {str(e)}")
            return set()

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()

class ExpiringCache:
    """Cache with expiration time for entries"""
    
    def __init__(self, max_age_seconds: int = 300, max_len: int = 1000):
        self.max_age = max_age_seconds
        self.max_len = max_len
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str) -> Any:
        """Get item from cache if not expired"""
        if key in self._cache:
            if time.time() - self._timestamps[key] > self.max_age:
                del self._cache[key]
                del self._timestamps[key]
                return None
            return self._cache[key]
        return None

    def set(self, key: str, value: Any):
        """Add item to cache"""
        if len(self._cache) >= self.max_len:
            oldest_key = min(self._timestamps.items(), key=lambda x: x[1])[0]
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        """Clear cache"""
        self._cache.clear()
        self._timestamps.clear()

class ModelRegistry:
    """Manages model persistence, versioning, and metadata"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_path = Path(config.model_save_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Dict] = {}
        self._load_registry()
    
    def save_model(
        self,
        model: BaseEstimator,
        model_id: str,
        metadata: Dict,
        version: Optional[str] = None
    ) -> str:
        """Save model with metadata and versioning"""
        try:
            # Generate version if not provided
            version = version or datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            
            # Create model directory
            model_dir = self.base_path / model_id / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model file
            model_path = model_dir / 'model.joblib'
            joblib.dump(model, model_path)
            
            # Save metadata
            metadata.update({
                'version': version,
                'saved_at': datetime.now(timezone.utc).isoformat(),
                'model_type': type(model).__name__
            })
            
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update registry
            if model_id not in self.models:
                self.models[model_id] = {}
            self.models[model_id][version] = metadata
            self._save_registry()
            
            return version
            
        except Exception as e:
            logger.error(f"Error saving model {model_id}: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def load_model(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Tuple[BaseEstimator, Dict]:
        """Load model and its metadata"""
        try:
            # Get latest version if not specified
            if version is None:
                version = self._get_latest_version(model_id)
                if not version:
                    raise ModelError(f"No versions found for model {model_id}")
            
            # Load model
            model_path = self.base_path / model_id / version / 'model.joblib'
            if not model_path.exists():
                raise ModelError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            
            # Load metadata
            metadata_path = self.base_path / model_id / version / 'metadata.json'
            if not metadata_path.exists():
                raise ModelError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get model information and available versions"""
        if model_id not in self.models:
            return {}
        
        versions = self.models[model_id]
        latest_version = self._get_latest_version(model_id)
        
        return {
            'model_id': model_id,
            'versions': versions,
            'latest_version': latest_version,
            'version_count': len(versions)
        }
    
    def _get_latest_version(self, model_id: str) -> Optional[str]:
        """Get latest version of a model"""
        if model_id not in self.models:
            return None
            
        versions = list(self.models[model_id].keys())
        return max(versions) if versions else None
    
    def _load_registry(self):
        """Load model registry from disk"""
        registry_path = self.base_path / 'registry.json'
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.models = json.load(f)
    
    def _save_registry(self):
        """Save model registry to disk"""
        registry_path = self.base_path / 'registry.json'
        with open(registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)

class ModelTrainingPipeline:
    """Manages model training and evaluation"""
    
    def __init__(
        self,
        config: Config,
        model_registry: ModelRegistry,
        data_validator: Optional[Any] = None
    ):
        self.config = config
        self.registry = model_registry
        self.data_validator = data_validator
        self.logger = logging.getLogger(__name__)
    
    async def train_model(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        model_params: Dict,
        feature_columns: List[str],
        target_column: str
    ) -> Tuple[str, Dict]:
        """Train and evaluate a model"""
        try:
            # Validate input data
            if self.data_validator:
                self.data_validator.validate_training_data(
                    training_data,
                    feature_columns,
                    target_column
                )
            
            # Prepare data
            X = training_data[feature_columns]
            y = training_data[target_column]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.2,
                shuffle=False  # Time series data
            )
            
            # Initialize and train model
            model = self._initialize_model(model_params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            metrics = self._evaluate_model(model, X_val, y_val)
            
            # Save model
            metadata = {
                'parameters': model_params,
                'features': feature_columns,
                'metrics': metrics,
                'data_info': {
                    'training_size': len(X_train),
                    'validation_size': len(X_val),
                    'feature_count': len(feature_columns)
                }
            }
            
            version = self.registry.save_model(model, model_id, metadata)
            
            return version, metadata
            
        except Exception as e:
            self.logger.error(f"Training failed for {model_id}: {str(e)}")
            raise ModelError(f"Model training failed: {str(e)}")
    
    def _initialize_model(self, params: Dict) -> BaseEstimator:
        """Initialize model based on parameters"""
        model_type = params.pop('model_type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict:
        """Evaluate model performance"""
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)
        
        return {
            'accuracy': accuracy_score(y_val, predictions),
            'precision': precision_score(y_val, predictions, average='weighted'),
            'recall': recall_score(y_val, predictions, average='weighted'),
            'f1': f1_score(y_val, predictions, average='weighted'),
            'log_loss': log_loss(y_val, probabilities)
        }

class StreamManager:
    """Manages real-time data streams from multiple sources"""
    
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.api_config = api_config
        self.active_streams: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.callbacks: Dict[str, List[callable]] = {}
        self.reconnect_delays = [1, 2, 5, 10, 30]  # Exponential backoff
        self._stream_tasks: Set[asyncio.Task] = set()
        self.logger = logging.getLogger(__name__)
    
    async def start_stream(
        self,
        symbol: str,
        callbacks: List[callable],
        stream_types: List[str] = ['trades', 'orderbook', 'ticker']
    ):
        """Start streaming data for a symbol"""
        try:
            stream_id = f"{symbol}_{'_'.join(stream_types)}"
            
            if stream_id in self.active_streams:
                self.logger.info(f"Stream already active for {stream_id}")
                return
            
            # Register callbacks
            self.callbacks[stream_id] = callbacks
            
            # Create and start stream tasks
            for exchange in ['binance', 'ftx', 'kraken']:
                task = asyncio.create_task(
                    self._maintain_stream(
                        symbol,
                        stream_types,
                        exchange,
                        stream_id
                    )
                )
                self._stream_tasks.add(task)
                task.add_done_callback(self._stream_tasks.discard)
            
            self.logger.info(f"Started streaming for {stream_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting stream: {str(e)}")
            raise StreamError(f"Failed to start stream: {str(e)}")
    
    async def stop_stream(self, symbol: str, stream_types: List[str] = None):
        """Stop streaming data for a symbol"""
        try:
            stream_id = f"{symbol}_{'_'.join(stream_types or ['trades', 'orderbook', 'ticker'])}"
            
            if stream_id not in self.active_streams:
                return
            
            # Close websocket
            ws = self.active_streams[stream_id]
            await ws.close()
            
            # Clean up
            del self.active_streams[stream_id]
            del self.callbacks[stream_id]
            
            # Cancel tasks
            for task in self._stream_tasks:
                if not task.done():
                    task.cancel()
            
            self.logger.info(f"Stopped streaming for {stream_id}")
            
        except Exception as e:
            self.logger.error(f"Error stopping stream: {str(e)}")
    
    async def _maintain_stream(
        self,
        symbol: str,
        stream_types: List[str],
        exchange: str,
        stream_id: str
    ):
        """Maintain websocket connection with reconnection"""
        retry_count = 0
        
        while True:
            try:
                uri = self._get_websocket_uri(symbol, stream_types, exchange)
                
                async with websockets.connect(uri) as websocket:
                    self.active_streams[stream_id] = websocket
                    retry_count = 0  # Reset retry count on successful connection
                    
                    # Subscribe to channels
                    await self._subscribe(websocket, symbol, stream_types, exchange)
                    
                    # Process messages
                    async for message in websocket:
                        await self._process_message(message, stream_id, exchange)
                        
            except websockets.ConnectionClosed:
                self.logger.warning(f"Connection closed for {stream_id}")
                await self._handle_reconnection(retry_count, stream_id)
                retry_count += 1
                
            except Exception as e:
                self.logger.error(f"Stream error: {str(e)}")
                await self._handle_reconnection(retry_count, stream_id)
                retry_count += 1
    
    def _get_websocket_uri(self, symbol: str, stream_types: List[str], exchange: str) -> str:
        """Get websocket URI for exchange"""
        if exchange == 'binance':
            streams = [f"{symbol.lower()}@{stype}" for stype in stream_types]
            return f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        elif exchange == 'ftx':
            return "wss://ftx.com/ws/"
        elif exchange == 'kraken':
            return "wss://ws.kraken.com"
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
    
    async def _subscribe(
        self,
        websocket: websockets.WebSocketClientProtocol,
        symbol: str,
        stream_types: List[str],
        exchange: str
    ):
        """Subscribe to channels"""
        if exchange == 'binance':
            # Binance auto-subscribes based on URI
            pass
        elif exchange == 'ftx':
            for stype in stream_types:
                subscribe_message = {
                    "op": "subscribe",
                    "channel": stype,
                    "market": symbol
                }
                await websocket.send(json.dumps(subscribe_message))
        elif exchange == 'kraken':
            subscribe_message = {
                "event": "subscribe",
                "pair": [symbol],
                "subscription": {"name":stream_types}
                }
            await websocket.send(json.dumps(subscribe_message))
    
    async def _process_message(
        self,
        message: str,
        stream_id: str,
        exchange: str
    ):
        """Process incoming websocket message"""
        try:
            data = json.loads(message)
            
            # Normalize data format
            normalized = self._normalize_message(data, exchange)
            
            # Call registered callbacks
            for callback in self.callbacks.get(stream_id, []):
                try:
                    await callback(normalized)
                except Exception as e:
                    self.logger.error(f"Callback error: {str(e)}")
                    
        except json.JSONDecodeError:
            self.logger.error("Invalid JSON message")
        except Exception as e:
            self.logger.error(f"Message processing error: {str(e)}")
    
    def _normalize_message(self, data: Dict, exchange: str) -> Dict:
        """Normalize message format across exchanges"""
        if exchange == 'binance':
            return self._normalize_binance(data)
        elif exchange == 'ftx':
            return self._normalize_ftx(data)
        elif exchange == 'kraken':
            return self._normalize_kraken(data)
        return data
    
    async def _handle_reconnection(self, retry_count: int, stream_id: str):
        """Handle reconnection with exponential backoff"""
        delay = self.reconnect_delays[min(retry_count, len(self.reconnect_delays)-1)]
        self.logger.info(f"Reconnecting to {stream_id} in {delay} seconds...")
        await asyncio.sleep(delay)

class SocialMediaIntegration:
    """Comprehensive social media integration"""
    
    def __init__(self, config: Config, api_config: APIConfig):
        self.config = config
        self.api_config = api_config
        self.session = aiohttp.ClientSession()
        self.rate_limiters = self._initialize_rate_limiters()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_rate_limiters(self) -> Dict[str, TokenBucketLimiter]:
        """Initialize rate limiters for each platform"""
        return {
            'twitter': TokenBucketLimiter(
                rate_limit=300,
                per_second=900
            ),
            'reddit': TokenBucketLimiter(
                rate_limit=30,
                per_second=60
            ),
            'telegram': TokenBucketLimiter(
                rate_limit=20,
                per_second=60
            )
        }
    
    async def fetch_social_data(
        self,
        symbol: str,
        platforms: List[str] = ['twitter', 'reddit', 'telegram']
    ) -> Dict:
        """Fetch data from multiple social platforms"""
        tasks = []
        for platform in platforms:
            if platform == 'twitter':
                tasks.append(self._fetch_twitter_data(symbol))
            elif platform == 'reddit':
                tasks.append(self._fetch_reddit_data(symbol))
            elif platform == 'telegram':
                tasks.append(self._fetch_telegram_data(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            platform: result for platform, result in zip(platforms, results)
            if not isinstance(result, Exception)
        }
    
    async def _fetch_twitter_data(self, symbol: str) -> Dict:
        """Fetch Twitter data using v2 API"""
        try:
            async with self.rate_limiters['twitter']:
                headers = {
                    'Authorization': f"Bearer {self.api_config.twitter_bearer_token}",
                    'Content-Type': 'application/json'
                }
                
                # Search recent tweets
                search_url = "https://api.twitter.com/2/tweets/search/recent"
                params = {
                    'query': f"#{symbol} OR ${symbol} -is:retweet",
                    'tweet.fields': 'created_at,public_metrics,entities',
                    'max_results': 100
                }
                
                async with self.session.get(search_url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    tweets_data = await response.json()
                
                return self._process_twitter_data(tweets_data)
                
        except Exception as e:
            self.logger.error(f"Twitter fetch error: {str(e)}")
            return {}
    
    async def _fetch_reddit_data(self, symbol: str) -> Dict:
        """Fetch Reddit data"""
        try:
            async with self.rate_limiters['reddit']:
                headers = {
                    'User-Agent': 'CryptoAssistant/1.0',
                    'Authorization': f"Bearer {self.api_config.reddit_access_token}"
                }
                
                # Search crypto subreddits
                subreddits = ['cryptocurrency', 'cryptomarkets', f"{symbol}"]
                posts = []
                
                for subreddit in subreddits:
                    url = f"https://oauth.reddit.com/r/{subreddit}/search"
                    params = {
                        'q': symbol,
                        't': 'day',
                        'limit': 100
                    }
                    
                    async with self.session.get(url, headers=headers, params=params) as response:
                        response.raise_for_status()
                        data = await response.json()
                        posts.extend(data['data']['children'])
                
                return self._process_reddit_data(posts)
                
        except Exception as e:
            self.logger.error(f"Reddit fetch error: {str(e)}")
            return {}
    
    async def _fetch_telegram_data(self, symbol: str) -> Dict:
        """Fetch Telegram data from monitoring channels"""
        try:
            async with self.rate_limiters['telegram']:
                return {}  # Implement based on specific requirements
                
        except Exception as e:
            self.logger.error(f"Telegram fetch error: {str(e)}")
            return {}
    
    def _process_twitter_data(self, data: Dict) -> Dict:
        """Process Twitter API response"""
        if not data or 'data' not in data:
            return {}
            
        tweets = data['data']
        
        # Calculate metrics
        sentiment_scores = []
        total_engagement = 0
        
        for tweet in tweets:
            metrics = tweet.get('public_metrics', {})
            engagement = (
                metrics.get('retweet_count', 0) +
                metrics.get('reply_count', 0) +
                metrics.get('like_count', 0)
            )
            total_engagement += engagement
            
            # Calculate sentiment (implement using your preferred NLP model)
            # sentiment_scores.append(calculate_sentiment(tweet['text']))
        
        return {
            'tweet_count': len(tweets),
            'total_engagement': total_engagement,
            'avg_engagement': total_engagement / len(tweets) if tweets else 0,
            'tweets': tweets
        }
    
    def _process_reddit_data(self, posts: List[Dict]) -> Dict:
        """Process Reddit API response"""
        if not posts:
            return {}
            
        total_score = 0
        total_comments = 0
        
        for post in posts:
            data = post['data']
            total_score += data.get('score', 0)
            total_comments += data.get('num_comments', 0)
        
        return {
            'post_count': len(posts),
            'total_score': total_score,
            'total_comments': total_comments,
            'avg_score': total_score / len(posts),
            'posts': posts
        }

    async def cleanup(self):
        if hasattr(self, 'session'):
            await self.session.close()

class BaseFeatureGenerator:
    """Base class for feature generators"""
    
    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass

class TechnicalFeatureGenerator(BaseFeatureGenerator):
    """Generates technical analysis features"""
    
    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = data.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Momentum indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['willr'] = talib.WILLR(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['natr'] = talib.NATR(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Generate labels (future returns)
        labels = self._generate_labels(df)
        
        # Drop NaN values and unnecessary columns
        df = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
        df = df.dropna()
        
        return df, labels
    
    def _generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading labels based on future returns"""
        future_returns = df['close'].pct_change(5).shift(-5)  # 5-period forward returns
        
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_returns > 0.02] = 1  # Buy
        labels[future_returns < -0.02] = -1  # Sell
        labels[(-0.02 <= future_returns) & (future_returns <= 0.02)] = 0  # Hold
        
        return labels.iloc[:-5]  # Remove last 5 rows due to shift

class SentimentFeatureGenerator(BaseFeatureGenerator):
    """Generates sentiment analysis features"""
    
    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Placeholder - implement sentiment feature generation
        return pd.DataFrame(), pd.Series()

class RegimeFeatureGenerator(BaseFeatureGenerator):
    """Generates market regime features"""
    
    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        # Placeholder - implement regime feature generation
        return pd.DataFrame(), pd.Series()

def display_market_overview(analysis: Dict):
    """Display market overview"""
    st.subheader("Market Overview")
    
    # Current price and changes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price",
            f"${analysis['market_data']['close'][-1]:.2f}",
            f"{analysis['market_data']['price_change_24h']:.2f}%"
        )
    
    with col2:
        st.metric(
            "24h Volume",
            f"${analysis['market_data']['volume_24h']:,.0f}"
        )
    
    with col3:
        st.metric(
            "Market Regime",
            analysis['agent_predictions']['market_regime']['regime']
        )
    
    # Market sentiment
    st.subheader("Market Sentiment")
    sentiment = analysis['agent_predictions']['sentiment']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Overall Sentiment",
            sentiment['sentiment'],
            f"Score: {sentiment['score']:.2f}"
        )
    
    with col2:
        st.metric(
            "Confidence",
            f"{sentiment['confidence']:.1f}%"
        )

def display_technical_analysis(analysis: Dict):
    """Display technical analysis"""
    st.subheader("Technical Analysis")
    
    # Technical indicators
    technical = analysis['agent_predictions']['technical']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Trend Direction",
            technical['trend']['direction'],
            f"Strength: {technical['trend']['strength']}/10"
        )
    
    with col2:
        st.metric(
            "Momentum",
            technical['momentum']['state'],
            f"RSI: {technical['momentum']['rsi']:.1f}"
        )
    
    with col3:
        st.metric(
            "Volatility",
            technical['volatility']['level'],
            f"ATR: {technical['volatility']['atr']:.2f}"
        )

def display_trading_strategy(analysis: Dict):
    """Display trading strategy"""
    st.subheader("Trading Strategy")
    
    strategy = analysis['strategy']
    
    # Action and confidence
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Recommended Action",
            strategy['action'],
            f"Confidence: {strategy['confidence']:.1f}%"
        )
    
    with col2:
        st.metric(
            "Position Size",
            f"{strategy['position_size']*100:.1f}%"
        )
    
    # Entry strategy
    st.subheader("Entry Strategy")
    if strategy['entry']:
        for i, (price, alloc) in enumerate(zip(
            strategy['entry']['points']['price_levels'],
            strategy['entry']['points']['allocation']
        )):
            st.write(f"Entry {i+1}: ${price:.2f} ({alloc*100:.0f}%)")

def display_performance_metrics(analysis: Dict):
    """Display performance metrics"""
    st.subheader("Performance Metrics")
    
    metrics = analysis['performance']['metrics']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate']*100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Average Profit",
            f"{metrics['avg_profit']:.2f}%"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}"
        )

async def run_analysis(app: CryptoAnalysisApp, symbol: str, timeframe: str) -> Dict:
    try:
        logger.info(f"Starting analysis for {symbol} on {timeframe}")
        analysis = await app.analyze_market(symbol, timeframe)
        
        if not analysis:
            raise ValueError("Analysis returned no data")
            
        return {
            'market_data': analysis.get('market_data', {}),
            'predictions': analysis.get('predictions', []),
            'strategy': analysis.get('strategy', {})
        }
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
    finally:
        await app.cleanup()

def display_manage_tab(analysis: Dict, app: CryptoAnalysisApp):
    """Display management interface"""
    st.subheader("Model Management")
    
    # Model training section
    st.write("### Model Training")
    if st.button("Retrain Models"):
        with st.spinner("Training models..."):
            # Add training logic here
            st.success("Models retrained successfully")
    
    # API Configuration
    st.write("### API Configuration")
    for api_name, status in app.api_config.active_apis.items():
        st.checkbox(f"{api_name} API", value=status, key=f"api_{api_name}")
    
    # Risk Management
    st.write("### Risk Management")
    position_size = st.slider("Max Position Size (%)", 0, 100, int(app.config.max_position_size * 100))
    stop_loss = st.slider("Default Stop Loss (%)", 0, 50, 10)
    
    if st.button("Save Settings"):
        app.config.max_position_size = position_size / 100
        st.success("Settings saved successfully")

def test_api_key_loading():
    load_dotenv()
    assert os.getenv("BINANCE_API_KEY") is not None, "BINANCE_API_KEY not loaded"
    assert os.getenv("MESSARI_API_KEY") is not None, "MESSARI_API_KEY not loaded"

async def test_fetch_market_data():
    config = Config()
    api_config = APIConfig()
    fetcher = DataFetcher(config, api_config)
    data = await fetcher.fetch_market_data("BTC/USDT", "1d")
    assert not data.empty, "Failed to fetch market data"
    assert set(data.columns) == {"open", "high", "low", "close", "volume"}

def test_async_execution():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(test_fetch_market_data())
    assert result is not None, "Async execution failed"

def main():
    # Validate API keys
    load_dotenv()
    api_config = APIConfig()
    if not all([
        api_config.binance_key,
        api_config.messari_key,
        api_config.santiment_key
    ]):
        st.error("API keys not loaded correctly. Please check your .env file.")
        return
    
    st.success("API keys validated successfully.")

    # Initialize app
    app = CryptoAnalysisApp()  # Make sure this is properly defined elsewhere
    
    st.sidebar.title("Configuration")
    
    # Symbol input
    symbol = st.sidebar.text_input(
        "Enter trading pair (e.g., BTC/USDT)",
        value="BTC/USDT"
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Select timeframe",
        options=['1m', '5m', '15m', '1h', '4h', '1d'],
        index=4  # Default to 4h
    )
    
    # Analysis button
    if st.sidebar.button("Analyze Market"):
        try:
            print("Starting analysis...")
            with st.spinner("Analyzing market..."):
                # Run async task within a synchronous wrapper
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                print(f"Analyzing {symbol} on {timeframe} timeframe") # Debug print
                analysis = loop.run_until_complete(run_analysis(app, symbol, timeframe))
                print("Analysis completed") 

                # Call updated run_analysis
                analysis = loop.run_until_complete(run_analysis(app, symbol, timeframe))
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Market Overview",
                    "Technical Analysis",
                    "Trading Strategy",
                    "Performance Metrics",
                    "Manage"
                ])
                
                with tab1:
                    display_market_overview(analysis)
                
                with tab2:
                    display_technical_analysis(analysis)
                
                with tab3:
                    display_trading_strategy(analysis)
                
                with tab4:
                    display_performance_metrics(analysis)
                
                with tab5:
                    display_manage_tab(analysis, app)
                    
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
        finally:
            loop.run_until_complete(app.cleanup())  # Ensure cleanup is called

if __name__ == "__main__":
    main()
