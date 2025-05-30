import yaml
import logging
import redis
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.cache = self._setup_cache()
        self.providers = self._setup_providers()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('MarketDataLoader')
        log_config = self.config['logging']
        
        logging.basicConfig(
            level=log_config['level'],
            format=log_config['format'],
            filename=log_config['file']
        )
        
        return logger
    
    def _setup_cache(self) -> redis.Redis:
        cache_config = self.config['cache']
        return redis.Redis(
            host=cache_config['host'],
            port=cache_config['port'],
            db=cache_config['db']
        )
    
    def _setup_providers(self) -> Dict[str, Any]:
        providers = {}
        apis_config = self.config['apis']
        
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
        try:
            data = self.cache.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Error reading from cache: {str(e)}")
        return None
    
    def _cache_data(self, key: str, data: Dict[str, Any]) -> None:
        """Guarda datos en el caché"""
        try:
            self.cache.setex(
                key,
                self.config['cache']['ttl'],
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
        try:
            if symbol:
                self.cache.delete(f"market_data:{symbol}")
            else:
                self.cache.flushdb()
            self.logger.info(f"Cache cleared for {symbol if symbol else 'all symbols'}")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}") 