from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class MarketData:
    """Estructura de datos estandarizada para datos de mercado"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: Optional[int] = None
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    source: str = ""
    extra_data: Dict[str, Any] = None

@dataclass
class OrderBookData:
    """Estructura de datos para el order book"""
    symbol: str
    timestamp: datetime
    bids: List[tuple[float, float]]  # precio, cantidad
    asks: List[tuple[float, float]]  # precio, cantidad
    source: str = ""

def normalize_ohlcv_data(data: Dict[str, Any], source: str) -> MarketData:
    """Normaliza datos OHLCV de diferentes fuentes al formato estándar"""
    normalized = MarketData(
        symbol=data.get('symbol', ''),
        timestamp=pd.to_datetime(data.get('timestamp')),
        open=float(data.get('open', 0)),
        high=float(data.get('high', 0)),
        low=float(data.get('low', 0)),
        close=float(data.get('close', 0)),
        volume=float(data.get('volume', 0)),
        trades=data.get('trades'),
        vwap=data.get('vwap'),
        source=source,
        extra_data={}
    )
    
    # Guardar datos adicionales específicos de la fuente
    for key, value in data.items():
        if key not in MarketData.__annotations__:
            normalized.extra_data[key] = value
            
    return normalized

def normalize_orderbook(data: Dict[str, Any], source: str) -> OrderBookData:
    """Normaliza datos del order book de diferentes fuentes"""
    return OrderBookData(
        symbol=data.get('symbol', ''),
        timestamp=pd.to_datetime(data.get('timestamp')),
        bids=[(float(p), float(q)) for p, q in data.get('bids', [])],
        asks=[(float(p), float(q)) for p, q in data.get('asks', [])],
        source=source
    ) 