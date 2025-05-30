from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

import pandas as pd
from binance.client import Client as BinanceClient
from finnhub import Client as FinnhubClient
from polygon import RESTClient as PolygonClient
from twelvedata import TDClient
from alpha_vantage.timeseries import TimeSeries
import requests

from .data_types import MarketData, OrderBookData, normalize_ohlcv_data, normalize_orderbook

class BaseProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"Provider-{self.__class__.__name__}")
        
    @abstractmethod
    def get_market_data(self, symbol: str, interval: str = "1m") -> List[MarketData]:
        """Obtiene datos de mercado para un símbolo"""
        pass
        
    @abstractmethod
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Obtiene el order book para un símbolo"""
        pass

class BinanceProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = BinanceClient(
            config['api_key'],
            config['api_secret'],
            testnet=config.get('testnet', False)
        )
        
    def get_market_data(self, symbol: str, interval: str = "1m") -> List[MarketData]:
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=100
            )
            
            market_data = []
            for k in klines:
                data = {
                    'symbol': symbol,
                    'timestamp': k[0],
                    'open': k[1],
                    'high': k[2],
                    'low': k[3],
                    'close': k[4],
                    'volume': k[5],
                    'trades': k[8],
                    'vwap': float(k[7]) / float(k[5]) if float(k[5]) > 0 else None
                }
                market_data.append(normalize_ohlcv_data(data, 'binance'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Binance data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=20)
            return normalize_orderbook({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': depth['bids'],
                'asks': depth['asks']
            }, 'binance')
        except Exception as e:
            self.logger.error(f"Error fetching Binance orderbook: {str(e)}")
            return None

class FinnhubProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = FinnhubClient(api_key=config['api_key'])
        
    def get_market_data(self, symbol: str, interval: str = "1m") -> List[MarketData]:
        try:
            end = datetime.now()
            start = end - timedelta(days=1)
            
            candles = self.client.stock_candles(
                symbol,
                interval,
                int(start.timestamp()),
                int(end.timestamp())
            )
            
            if candles['s'] != 'ok':
                return []
                
            market_data = []
            for i in range(len(candles['t'])):
                data = {
                    'symbol': symbol,
                    'timestamp': candles['t'][i],
                    'open': candles['o'][i],
                    'high': candles['h'][i],
                    'low': candles['l'][i],
                    'close': candles['c'][i],
                    'volume': candles['v'][i]
                }
                market_data.append(normalize_ohlcv_data(data, 'finnhub'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Finnhub data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        # Finnhub no proporciona datos de order book
        return None

class PolygonProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = PolygonClient(config['api_key'])
        
    def get_market_data(self, symbol: str, interval: str = "1m") -> List[MarketData]:
        try:
            # Convertir intervalo a formato Polygon
            interval_map = {"1m": "minute", "1h": "hour", "1d": "day"}
            poly_interval = interval_map.get(interval, "minute")
            
            resp = self.client.get_aggs(
                symbol,
                1,
                poly_interval,
                datetime.now() - timedelta(days=1),
                datetime.now()
            )
            
            market_data = []
            for bar in resp:
                data = {
                    'symbol': symbol,
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'vwap': bar.vwap,
                    'trades': bar.transactions
                }
                market_data.append(normalize_ohlcv_data(data, 'polygon'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Polygon data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        try:
            book = self.client.get_level2_book(symbol)
            return normalize_orderbook({
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': [(b.price, b.size) for b in book.bids],
                'asks': [(a.price, a.size) for a in book.asks]
            }, 'polygon')
        except Exception as e:
            self.logger.error(f"Error fetching Polygon orderbook: {str(e)}")
            return None

class TwelveDataProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = TDClient(apikey=config['api_key'])
        
    def get_market_data(self, symbol: str, interval: str = "1min") -> List[MarketData]:
        try:
            ts = self.client.time_series(
                symbol=symbol,
                interval=interval,
                outputsize=100
            )
            
            market_data = []
            for bar in ts.as_pandas().itertuples():
                data = {
                    'symbol': symbol,
                    'timestamp': bar.datetime,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                market_data.append(normalize_ohlcv_data(data, 'twelvedata'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching TwelveData data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        # TwelveData no proporciona datos de order book
        return None

class AlphaVantageProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = TimeSeries(
            key=config['api_key'],
            output_format='pandas'
        )
        
    def get_market_data(self, symbol: str, interval: str = "1min") -> List[MarketData]:
        try:
            data, _ = self.client.get_intraday(
                symbol=symbol,
                interval=interval,
                outputsize='compact'
            )
            
            market_data = []
            for timestamp, row in data.iterrows():
                market_data.append(normalize_ohlcv_data({
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': row['1. open'],
                    'high': row['2. high'],
                    'low': row['3. low'],
                    'close': row['4. close'],
                    'volume': row['5. volume']
                }, 'alphavantage'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        # Alpha Vantage no proporciona datos de order book
        return None

class NinjaApisProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config['api_key']
        self.base_url = "https://api.ninjaapis.com/v1"
        
    def get_market_data(self, symbol: str, interval: str = "1m") -> List[MarketData]:
        try:
            response = requests.get(
                f"{self.base_url}/market-data/{symbol}",
                headers={"X-Api-Key": self.api_key},
                params={"interval": interval}
            )
            response.raise_for_status()
            
            data = response.json()
            market_data = []
            
            for bar in data['data']:
                market_data.append(normalize_ohlcv_data(bar, 'ninjaapis'))
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching NinjaAPIs data: {str(e)}")
            return []
            
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        try:
            response = requests.get(
                f"{self.base_url}/orderbook/{symbol}",
                headers={"X-Api-Key": self.api_key}
            )
            response.raise_for_status()
            
            data = response.json()
            return normalize_orderbook(data, 'ninjaapis')
            
        except Exception as e:
            self.logger.error(f"Error fetching NinjaAPIs orderbook: {str(e)}")
            return None 