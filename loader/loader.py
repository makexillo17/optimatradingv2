import yaml
import logging
import redis
import json
import os
import ccxt
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .data_types import MarketData, OrderBookData
from .providers import (
    BinanceProvider,
    FinnhubProvider,
    PolygonProvider,
    TwelveDataProvider,
    AlphaVantageProvider,
    NinjaApisProvider
)

class MarketDataLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else self._load_config_from_env()
        self.logger = self._setup_logging()
        self.cache = self._setup_cache()
        self.providers = self._setup_providers()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._load_config_from_env()
    
    def _load_config_from_env(self) -> Dict[str, Any]:
        """Carga configuración desde variables de entorno cuando no hay config.yaml"""
        return {
            'logging': {
                'level': os.environ.get('LOG_LEVEL', 'INFO'),
                'format': os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            },
            'cache': {
                'host': os.environ.get('REDIS_HOST', 'localhost'),
                'port': int(os.environ.get('REDIS_PORT', '6379')),
                'db': int(os.environ.get('REDIS_DB', '0')),
                'ttl': int(os.environ.get('CACHE_TTL', '3600'))
            },
            'apis': {
                'binance': {
                    'api_key': os.environ.get('BINANCE_API_KEY', ''),
                    'api_secret': os.environ.get('BINANCE_API_SECRET', '')
                },
                'finnhub': {
                    'api_key': os.environ.get('FINNHUB_API_KEY', '')
                },
                'polygon': {
                    'api_key': os.environ.get('POLYGON_API_KEY', '')
                },
                'twelvedata': {
                    'api_key': os.environ.get('TWELVEDATA_API_KEY', '')
                },
                'alphavantage': {
                    'api_key': os.environ.get('ALPHAVANTAGE_API_KEY', '')
                },
                'ninjaapis': {
                    'api_key': os.environ.get('NINJAAPIS_API_KEY', '')
                }
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('MarketDataLoader')
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Crear carpeta de logs si no existe
        log_dir = Path(__file__).resolve().parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Definir ruta absoluta al archivo de log
        log_file = log_dir / "optimatrading.log"
        # Configurar logging con archivo
        logging.basicConfig(
            filename=str(log_file),
            level=getattr(logging, log_level.upper(), logging.INFO),
            format=log_format
        )
        return logger
    
    def _setup_cache(self) -> redis.Redis:
        cache_config = self.config.get('cache', {})
        try:
            return redis.Redis(
                host=cache_config.get('host', 'localhost'),
                port=cache_config.get('port', 6379),
                db=cache_config.get('db', 0)
            )
        except Exception as e:
            # Si Redis no está disponible, crear un objeto mock o None
            logging.warning(f"Redis no disponible, funcionando sin caché: {str(e)}")
            return None
    
    def _setup_providers(self) -> Dict[str, Any]:
        providers = {}
        apis_config = self.config.get('apis', {})
        
        provider_mapping = {
            'binance': BinanceProvider,
            'finnhub': FinnhubProvider,
            'polygon': PolygonProvider,
            'twelvedata': TwelveDataProvider,
            'alphavantage': AlphaVantageProvider,
            'ninjaapis': NinjaApisProvider
        }
        
        for api_name, provider_class in provider_mapping.items():
            if api_name in apis_config:
                try:
                    providers[api_name] = provider_class(apis_config[api_name])
                    self.logger.info(f"Initialized provider: {api_name}")
                except Exception as e:
                    self.logger.error(f"Error initializing {api_name}: {str(e)}")
        
        return providers
    
    def _get_cached_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Intenta obtener datos del caché"""
        if self.cache is None:
            return None
        try:
            data = self.cache.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Error reading from cache: {str(e)}")
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]) -> None:
        """Guarda datos en el caché"""
        if self.cache is None:
            return
        try:
            ttl = self.config.get('cache', {}).get('ttl', 3600)
            self.cache.setex(
                key,
                ttl,
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Error writing to cache: {str(e)}")
    
    def load_market_data(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Carga datos de mercado de todas las fuentes disponibles
        
        Args:
            symbol: Símbolo del activo
            use_cache: Si se debe usar el caché
            
        Returns:
            Dict con datos de mercado consolidados
        """
        cache_key = f"market_data:{symbol}"
        
        # Intentar obtener del caché
        if use_cache:
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                self.logger.info(f"Cache hit for {symbol}")
                return cached_data
        
        consolidated_data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'market_data': {},
            'orderbooks': {},
            'metadata': {
                'sources': [],
                'success_count': 0,
                'error_count': 0
            }
        }
        
        # Usar ThreadPoolExecutor para paralelizar las llamadas
        with ThreadPoolExecutor(max_workers=len(self.providers)) as executor:
            # Preparar futures para market data
            market_futures = {
                executor.submit(
                    provider.get_market_data,
                    symbol
                ): name
                for name, provider in self.providers.items()
            }
            
            # Preparar futures para orderbooks
            orderbook_futures = {
                executor.submit(
                    provider.get_orderbook,
                    symbol
                ): name
                for name, provider in self.providers.items()
            }
            
            # Procesar resultados de market data
            for future in as_completed(market_futures):
                provider_name = market_futures[future]
                try:
                    data = future.result()
                    if data:
                        consolidated_data['market_data'][provider_name] = [
                            d.__dict__ for d in data
                        ]
                        consolidated_data['metadata']['sources'].append(provider_name)
                        consolidated_data['metadata']['success_count'] += 1
                except Exception as e:
                    self.logger.error(
                        f"Error getting market data from {provider_name}: {str(e)}"
                    )
                    consolidated_data['metadata']['error_count'] += 1
            
            # Procesar resultados de orderbooks
            for future in as_completed(orderbook_futures):
                provider_name = orderbook_futures[future]
                try:
                    data = future.result()
                    if data:
                        consolidated_data['orderbooks'][provider_name] = data.__dict__
                except Exception as e:
                    self.logger.error(
                        f"Error getting orderbook from {provider_name}: {str(e)}"
                    )
        
        # Guardar en caché si hay datos
        if consolidated_data['metadata']['success_count'] > 0:
            self._cache_data(cache_key, consolidated_data)
        
        return consolidated_data
    
    def get_provider_status(self) -> Dict[str, bool]:
        """Verifica el estado de cada proveedor"""
        status = {}
        
        for name, provider in self.providers.items():
            try:
                # Intentar obtener datos de un símbolo común
                data = provider.get_market_data("BTC/USD", "1m")
                status[name] = bool(data)
            except Exception:
                status[name] = False
                
        return status
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Limpia el caché para un símbolo o todo el caché"""
        if self.cache is None:
            return
        try:
            if symbol:
                self.cache.delete(f"market_data:{symbol}")
            else:
                self.cache.flushdb()
            self.logger.info(f"Cache cleared for {symbol if symbol else 'all symbols'}")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
    
    def load_broker_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Descarga velas OHLCV de Kraken usando ccxt y devuelve un DataFrame de Pandas.
        Usa Kraken en lugar de Binance para evitar geobloqueos desde EE.UU.
        
        Args:
            symbol: Símbolo del par (ej: 'BTC/USD' o 'BTC/USDT')
            timeframe: Timeframe de las velas (ej: '1m', '5m', '1h', '1d')
            limit: Número de velas a descargar (máximo 1000)
            
        Returns:
            DataFrame de Pandas con columnas: timestamp, open, high, low, close, volume
            
        Raises:
            Exception: Si falla la conexión o la descarga de datos
        """
        try:
            # Conectar a Kraken (API pública, permite conexiones desde EE.UU.)
            exchange = ccxt.kraken({
                'enableRateLimit': True
            })
            
            self.logger.info(f"Descargando {limit} velas de {symbol} en timeframe {timeframe} desde Kraken")
            
            # Descargar velas OHLCV
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv or len(ohlcv) == 0:
                raise Exception(f"No se obtuvieron datos para {symbol}")
            
            # Convertir a DataFrame de Pandas
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir timestamp a datetime legible
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Asegurar que todos los números sean float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            self.logger.info(f"Descargadas {len(df)} velas exitosamente para {symbol}")
            
            return df
            
        except ccxt.NetworkError as e:
            error_msg = f"Error de red al conectar con Kraken: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
        except ccxt.ExchangeError as e:
            error_msg = f"Error de la API de Kraken: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Error inesperado descargando datos de Kraken: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg) from e 