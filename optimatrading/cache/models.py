"""
Modelos de datos para el sistema de caché
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field

class CacheConfig(BaseModel):
    """Configuración del sistema de caché"""
    redis_host: str = "localhost"
    redis_port: int = Field(6379, gt=0, lt=65536)
    redis_db: int = Field(0, ge=0)
    default_ttl: Optional[int] = Field(3600, gt=0)
    socket_timeout: int = Field(5, gt=0)
    socket_connect_timeout: int = Field(5, gt=0)
    retry_on_timeout: bool = True
    max_memory_items: int = Field(10000, gt=0)
    max_pool_size: int = Field(10, gt=0)
    compression_threshold: int = Field(1024, gt=0)  # 1KB
    enable_memory_cache: bool = True
    memory_cache_type: str = Field("hybrid", regex="^(ttl|lru|hybrid)$")

class CacheStats(BaseModel):
    """Estadísticas de uso de caché"""
    memory_used_bytes: int
    memory_peak_bytes: int
    total_connections: int
    total_commands: int
    hits: int
    misses: int
    keys: int
    memory_cache_size: int
    
    @property
    def hit_ratio(self) -> float:
        """Calcula el ratio de hits"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0
        
    @property
    def memory_usage_ratio(self) -> float:
        """Calcula el ratio de uso de memoria"""
        return self.memory_used_bytes / self.memory_peak_bytes if self.memory_peak_bytes > 0 else 0 