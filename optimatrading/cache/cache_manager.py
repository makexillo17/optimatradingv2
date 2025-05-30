"""
Sistema de caché optimizado para Optimatrading
"""

import asyncio
import json
import zlib
from typing import Any, Optional, Dict, Union, List, Tuple
from datetime import datetime, timedelta
import threading
from functools import wraps
import time
import msgpack
from concurrent.futures import ThreadPoolExecutor

import redis
from prometheus_client import Counter, Histogram, Gauge
from cachetools import TTLCache, LRUCache

from ..exceptions import CacheError
from ..logging import LoggerManager
from .models import CacheConfig, CacheStats

# Métricas Prometheus
cache_hits = Counter(
    'optimatrading_cache_hits_total',
    'Total de hits en caché',
    ['backend', 'namespace']
)

cache_misses = Counter(
    'optimatrading_cache_misses_total',
    'Total de misses en caché',
    ['backend', 'namespace']
)

cache_latency = Histogram(
    'optimatrading_cache_operation_seconds',
    'Latencia de operaciones de caché',
    ['operation', 'backend']
)

cache_size = Gauge(
    'optimatrading_cache_size_bytes',
    'Tamaño total de datos en caché',
    ['backend']
)

class CacheManager:
    """
    Gestor de caché optimizado para Optimatrading.
    
    Proporciona múltiples backends de caché (memoria, Redis),
    compresión automática, serialización eficiente y monitoreo.
    """
    
    # Umbrales de compresión
    COMPRESSION_THRESHOLD = 1024  # 1KB
    COMPRESSION_LEVEL = 6  # Equilibrio entre velocidad y ratio
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        logger_manager: Optional[LoggerManager] = None,
        default_ttl: Optional[int] = None,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        retry_on_timeout: bool = True,
        max_memory_items: int = 10000,
        max_pool_size: int = 10,
        compression_threshold: int = COMPRESSION_THRESHOLD,
        enable_memory_cache: bool = True
    ):
        """
        Inicializa el gestor de caché.
        
        Args:
            host: Host de Redis
            port: Puerto de Redis
            db: Base de datos Redis
            logger_manager: Gestor de logging
            default_ttl: TTL por defecto en segundos
            socket_timeout: Timeout para operaciones
            socket_connect_timeout: Timeout para conexión
            retry_on_timeout: Si se debe reintentar en timeout
            max_memory_items: Máximo de items en caché de memoria
            max_pool_size: Tamaño máximo del pool de conexiones
            compression_threshold: Umbral para compresión en bytes
            enable_memory_cache: Si se debe usar caché en memoria
        """
        self.logger = logger_manager.get_logger("CacheManager") if logger_manager else None
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold
        
        # Inicializar Redis
        self.redis_pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            retry_on_timeout=retry_on_timeout,
            max_connections=max_pool_size
        )
        
        # Caché en memoria
        self.memory_cache = None
        if enable_memory_cache:
            self.memory_cache = {
                'ttl': TTLCache(
                    maxsize=max_memory_items,
                    ttl=default_ttl or 3600
                ),
                'lru': LRUCache(maxsize=max_memory_items)
            }
        
        # Pool de hilos para operaciones asíncronas
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_pool_size,
            thread_name_prefix="cache_worker"
        )
        
        # Locks para acceso concurrente
        self._redis_locks = {}
        self._memory_locks = {}
        self._lock_lock = threading.Lock()
        
    async def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None,
        use_memory: bool = True
    ) -> Any:
        """
        Obtiene un valor de la caché.
        
        Args:
            key: Clave a buscar
            namespace: Namespace para la clave
            default: Valor por defecto si no existe
            use_memory: Si se debe buscar en caché de memoria
            
        Returns:
            Valor almacenado o default
        """
        full_key = f"{namespace}:{key}"
        start_time = time.time()
        
        try:
            # Intentar caché en memoria primero
            if use_memory and self.memory_cache:
                value = await self._get_from_memory(full_key)
                if value is not None:
                    cache_hits.labels(backend='memory', namespace=namespace).inc()
                    return value
                    
            # Intentar Redis
            value = await self._get_from_redis(full_key)
            if value is not None:
                cache_hits.labels(backend='redis', namespace=namespace).inc()
                # Actualizar caché en memoria
                if use_memory and self.memory_cache:
                    await self._set_in_memory(full_key, value)
                return value
                
            cache_misses.labels(
                backend='memory' if use_memory else 'redis',
                namespace=namespace
            ).inc()
            return default
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "cache_get_error",
                    key=full_key,
                    error=str(e)
                )
            raise CacheError(f"Error obteniendo de caché: {str(e)}")
            
        finally:
            cache_latency.labels(
                operation='get',
                backend='memory' if use_memory else 'redis'
            ).observe(time.time() - start_time)
            
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl: Optional[int] = None,
        use_memory: bool = True,
        compression: bool = True
    ) -> bool:
        """
        Almacena un valor en caché.
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            namespace: Namespace para la clave
            ttl: Tiempo de vida en segundos
            use_memory: Si se debe usar caché en memoria
            compression: Si se debe comprimir
            
        Returns:
            True si se almacenó correctamente
        """
        full_key = f"{namespace}:{key}"
        start_time = time.time()
        
        try:
            # Serializar y comprimir si es necesario
            serialized = self._serialize(value)
            if compression and len(serialized) > self.compression_threshold:
                serialized = self._compress(serialized)
                
            # Almacenar en Redis
            success = await self._set_in_redis(full_key, serialized, ttl)
            
            # Almacenar en memoria si está habilitado
            if use_memory and self.memory_cache and success:
                await self._set_in_memory(full_key, value, ttl)
                
            cache_size.labels(
                backend='memory' if use_memory else 'redis'
            ).set(len(serialized))
            
            return success
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "cache_set_error",
                    key=full_key,
                    error=str(e)
                )
            raise CacheError(f"Error almacenando en caché: {str(e)}")
            
        finally:
            cache_latency.labels(
                operation='set',
                backend='memory' if use_memory else 'redis'
            ).observe(time.time() - start_time)
            
    async def delete(
        self,
        key: str,
        namespace: str = "default",
        use_memory: bool = True
    ) -> bool:
        """
        Elimina una clave de la caché.
        
        Args:
            key: Clave a eliminar
            namespace: Namespace de la clave
            use_memory: Si se debe eliminar de memoria
            
        Returns:
            True si se eliminó correctamente
        """
        full_key = f"{namespace}:{key}"
        start_time = time.time()
        
        try:
            # Eliminar de Redis
            success = await self._delete_from_redis(full_key)
            
            # Eliminar de memoria si está habilitado
            if use_memory and self.memory_cache:
                await self._delete_from_memory(full_key)
                
            return success
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "cache_delete_error",
                    key=full_key,
                    error=str(e)
                )
            raise CacheError(f"Error eliminando de caché: {str(e)}")
            
        finally:
            cache_latency.labels(
                operation='delete',
                backend='memory' if use_memory else 'redis'
            ).observe(time.time() - start_time)
            
    async def get_stats(self) -> CacheStats:
        """
        Obtiene estadísticas de la caché.
        
        Returns:
            Estadísticas de uso de caché
        """
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        info = redis_client.info()
        
        return CacheStats(
            memory_used_bytes=info['used_memory'],
            memory_peak_bytes=info['used_memory_peak'],
            total_connections=info['connected_clients'],
            total_commands=info['total_commands_processed'],
            hits=info['keyspace_hits'],
            misses=info['keyspace_misses'],
            keys=info['db0']['keys'] if 'db0' in info else 0,
            memory_cache_size=sum(
                len(cache) for cache in self.memory_cache.values()
            ) if self.memory_cache else 0
        )
        
    def _get_lock(self, key: str, backend: str) -> threading.Lock:
        """Obtiene un lock para una clave"""
        with self._lock_lock:
            locks = self._redis_locks if backend == 'redis' else self._memory_locks
            if key not in locks:
                locks[key] = threading.Lock()
            return locks[key]
            
    async def _get_from_memory(self, key: str) -> Optional[Any]:
        """Obtiene un valor de la caché en memoria"""
        if not self.memory_cache:
            return None
            
        with self._get_lock(key, 'memory'):
            # Intentar TTL cache primero
            value = self.memory_cache['ttl'].get(key)
            if value is not None:
                return value
                
            # Intentar LRU cache
            return self.memory_cache['lru'].get(key)
            
    async def _set_in_memory(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Almacena un valor en la caché en memoria"""
        if not self.memory_cache:
            return
            
        with self._get_lock(key, 'memory'):
            if ttl:
                self.memory_cache['ttl'].setdefault(key, value)
            else:
                self.memory_cache['lru'][key] = value
                
    async def _delete_from_memory(self, key: str) -> None:
        """Elimina un valor de la caché en memoria"""
        if not self.memory_cache:
            return
            
        with self._get_lock(key, 'memory'):
            self.memory_cache['ttl'].pop(key, None)
            self.memory_cache['lru'].pop(key, None)
            
    async def _get_from_redis(self, key: str) -> Optional[Any]:
        """Obtiene un valor de Redis"""
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        with self._get_lock(key, 'redis'):
            value = redis_client.get(key)
            if value is None:
                return None
                
            # Descomprimir si es necesario
            if self._is_compressed(value):
                value = self._decompress(value)
                
            return self._deserialize(value)
            
    async def _set_in_redis(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None
    ) -> bool:
        """Almacena un valor en Redis"""
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        with self._get_lock(key, 'redis'):
            if ttl:
                return redis_client.setex(key, ttl, value)
            return redis_client.set(key, value)
            
    async def _delete_from_redis(self, key: str) -> bool:
        """Elimina un valor de Redis"""
        redis_client = redis.Redis(connection_pool=self.redis_pool)
        with self._get_lock(key, 'redis'):
            return bool(redis_client.delete(key))
            
    def _serialize(self, value: Any) -> bytes:
        """Serializa un valor usando MessagePack"""
        try:
            return msgpack.packb(value, use_bin_type=True)
        except Exception as e:
            raise CacheError(f"Error serializando valor: {str(e)}")
            
    def _deserialize(self, value: bytes) -> Any:
        """Deserializa un valor usando MessagePack"""
        try:
            return msgpack.unpackb(value, raw=False)
        except Exception as e:
            raise CacheError(f"Error deserializando valor: {str(e)}")
            
    def _compress(self, data: bytes) -> bytes:
        """Comprime datos usando zlib"""
        return zlib.compress(data, level=self.COMPRESSION_LEVEL)
        
    def _decompress(self, data: bytes) -> bytes:
        """Descomprime datos usando zlib"""
        return zlib.decompress(data)
        
    def _is_compressed(self, data: bytes) -> bool:
        """Verifica si los datos están comprimidos"""
        return data.startswith(b'x\x9c') or data.startswith(b'x\xda') 