"""
Modelos de validación de datos usando Pydantic
"""

from datetime import datetime
from typing import List, Dict, Optional, Union
from decimal import Decimal

from pydantic import BaseModel, Field, validator, constr

class MarketData(BaseModel):
    """Modelo para datos de mercado"""
    timestamp: datetime
    open: Decimal = Field(..., gt=0)
    high: Decimal = Field(..., gt=0)
    low: Decimal = Field(..., gt=0)
    close: Decimal = Field(..., gt=0)
    volume: Decimal = Field(..., ge=0)
    
    @validator('high')
    def high_greater_than_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('high debe ser mayor que low')
        return v
    
    @validator('high', 'low')
    def price_range_valid(cls, v, values):
        if 'open' in values:
            if v < values['open'] * Decimal('0.5') or v > values['open'] * Decimal('2.0'):
                raise ValueError('Precio fuera de rango razonable')
        return v

class ModuleConfig(BaseModel):
    """Modelo base para configuración de módulos"""
    enabled: bool = True
    weight: float = Field(1.0, gt=0, le=10)
    timeout: int = Field(30, gt=0)

class BrokerBehaviorConfig(ModuleConfig):
    """Configuración específica para módulo Broker Behavior"""
    timeframe: constr(regex='^[1-9][0-9]?[mhd]$')
    lookback_periods: int = Field(..., gt=0)
    min_volume: float = Field(..., gt=0)

class CarryTradeConfig(ModuleConfig):
    """Configuración específica para módulo Carry Trade"""
    min_rate_diff: float = Field(..., gt=0)
    funding_rate_threshold: float = Field(..., gt=0)
    update_interval: int = Field(..., gt=0)

class ModuleResult(BaseModel):
    """Resultado de análisis de un módulo"""
    recommendation: str = Field(..., regex='^(BUY|SELL|HOLD)$')
    confidence: float = Field(..., ge=0, le=1)
    justification: str
    timestamp: datetime
    metrics: Dict[str, float] = {}

class ConsensusConfig(BaseModel):
    """Configuración del sistema de consenso"""
    min_confidence: float = Field(..., gt=0, lt=1)
    min_agreement: float = Field(..., gt=0, lt=1)
    weighted_voting: bool = True
    required_modules: int = Field(..., gt=0)
    timeout: int = Field(..., gt=0)
    cache_results: bool = True
    cache_ttl: int = Field(300, gt=0)

class ValidationConfig(BaseModel):
    """Configuración de validación de datos"""
    required_fields: List[str]
    max_missing_values: float = Field(..., ge=0, lt=1)
    min_periods: int = Field(..., gt=0)
    check_data_quality: bool = True

class APICredentials(BaseModel):
    """Credenciales de API"""
    api_key: str
    api_secret: Optional[str] = None
    testnet: bool = False
    timeout: int = Field(30, gt=0)
    rate_limit: bool = True

class RedisConfig(BaseModel):
    """Configuración de Redis"""
    host: str
    port: int = Field(..., gt=0, lt=65536)
    db: int = Field(..., ge=0)
    default_ttl: int = Field(3600, gt=0)
    socket_timeout: int = Field(5, gt=0)
    socket_connect_timeout: int = Field(5, gt=0)
    retry_on_timeout: bool = True
    max_connections: int = Field(10, gt=0)

class LogConfig(BaseModel):
    """Configuración de logging"""
    level: str = Field(..., regex='^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$')
    format: str
    directory: Optional[str] = None
    max_size_mb: Optional[int] = Field(None, gt=0)
    backup_count: Optional[int] = Field(None, ge=0)

class MetricsConfig(BaseModel):
    """Configuración de métricas de rendimiento"""
    enabled: bool = True
    update_interval: int = Field(3600, gt=0)
    store_history: bool = True
    history_size: int = Field(1000, gt=0)
    calculate: Dict[str, bool] 