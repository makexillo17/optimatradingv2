import redis
from typing import Any, Optional, Dict, Union
import json
import pickle
from datetime import datetime, timedelta
import hashlib
from logging.logger_manager import LoggerManager

class CacheManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        logger_manager: Optional[LoggerManager] = None
    ):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.logger = logger_manager.get_logger("CacheManager") if logger_manager else None
        
    def get(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> Optional[Any]:
        """
        Obtiene un valor de la caché
        
        Args:
            key: Clave del valor
            namespace: Namespace opcional para agrupar claves
            
        Returns:
            Valor almacenado o None si no existe
        """
        try:
            full_key = self._get_full_key(key, namespace)
            value = self.redis_client.get(full_key)
            
            if value is None:
                return None
                
            return pickle.loads(value)
            
        except Exception as e:
            if self.logger:
                self.logger.error("cache_get_error", error=str(e), key=key)
            return None
            
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Almacena un valor en la caché
        
        Args:
            key: Clave para almacenar
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
            namespace: Namespace opcional
            
        Returns:
            True si se almacenó correctamente
        """
        try:
            full_key = self._get_full_key(key, namespace)
            pickled_value = pickle.dumps(value)
            
            if ttl:
                success = self.redis_client.setex(
                    full_key,
                    ttl,
                    pickled_value
                )
            else:
                success = self.redis_client.set(full_key, pickled_value)
                
            if self.logger:
                self.logger.info(
                    "cache_set",
                    key=key,
                    namespace=namespace,
                    ttl=ttl,
                    success=bool(success)
                )
                
            return bool(success)
            
        except Exception as e:
            if self.logger:
                self.logger.error("cache_set_error", error=str(e), key=key)
            return False
            
    def delete(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Elimina un valor de la caché
        
        Args:
            key: Clave a eliminar
            namespace: Namespace opcional
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            full_key = self._get_full_key(key, namespace)
            success = self.redis_client.delete(full_key)
            
            if self.logger:
                self.logger.info(
                    "cache_delete",
                    key=key,
                    namespace=namespace,
                    success=bool(success)
                )
                
            return bool(success)
            
        except Exception as e:
            if self.logger:
                self.logger.error("cache_delete_error", error=str(e), key=key)
            return False
            
    def clear_namespace(self, namespace: str) -> int:
        """
        Elimina todos los valores de un namespace
        
        Args:
            namespace: Namespace a limpiar
            
        Returns:
            Número de claves eliminadas
        """
        try:
            pattern = f"{namespace}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return 0
                
            deleted = self.redis_client.delete(*keys)
            
            if self.logger:
                self.logger.info(
                    "cache_clear_namespace",
                    namespace=namespace,
                    keys_deleted=deleted
                )
                
            return deleted
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "cache_clear_namespace_error",
                    error=str(e),
                    namespace=namespace
                )
            return 0
            
    def get_or_set(
        self,
        key: str,
        value_func: callable,
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> Any:
        """
        Obtiene un valor de la caché o lo calcula y almacena
        
        Args:
            key: Clave del valor
            value_func: Función para calcular el valor si no existe
            ttl: Tiempo de vida en segundos
            namespace: Namespace opcional
            
        Returns:
            Valor almacenado o calculado
        """
        value = self.get(key, namespace)
        
        if value is not None:
            return value
            
        value = value_func()
        self.set(key, value, ttl, namespace)
        return value
        
    def mget(
        self,
        keys: list,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Obtiene múltiples valores de la caché
        
        Args:
            keys: Lista de claves
            namespace: Namespace opcional
            
        Returns:
            Diccionario con valores encontrados
        """
        try:
            full_keys = [self._get_full_key(k, namespace) for k in keys]
            values = self.redis_client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = pickle.loads(value)
                    
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error("cache_mget_error", error=str(e), keys=keys)
            return {}
            
    def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Almacena múltiples valores en la caché
        
        Args:
            mapping: Diccionario de clave-valor
            ttl: Tiempo de vida en segundos
            namespace: Namespace opcional
            
        Returns:
            True si se almacenaron correctamente
        """
        try:
            pickled_mapping = {
                self._get_full_key(k, namespace): pickle.dumps(v)
                for k, v in mapping.items()
            }
            
            success = self.redis_client.mset(pickled_mapping)
            
            if ttl and success:
                pipe = self.redis_client.pipeline()
                for key in pickled_mapping.keys():
                    pipe.expire(key, ttl)
                pipe.execute()
                
            if self.logger:
                self.logger.info(
                    "cache_mset",
                    keys=list(mapping.keys()),
                    namespace=namespace,
                    ttl=ttl,
                    success=success
                )
                
            return success
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "cache_mset_error",
                    error=str(e),
                    keys=list(mapping.keys())
                )
            return False
            
    def _get_full_key(
        self,
        key: str,
        namespace: Optional[str] = None
    ) -> str:
        """
        Genera la clave completa con namespace
        
        Args:
            key: Clave base
            namespace: Namespace opcional
            
        Returns:
            Clave completa
        """
        if namespace:
            return f"{namespace}:{key}"
        return key
        
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """
        Obtiene estadísticas de la caché
        
        Returns:
            Diccionario con estadísticas
        """
        try:
            info = self.redis_client.info()
            return {
                'used_memory': info['used_memory'],
                'hits': info['keyspace_hits'],
                'misses': info['keyspace_misses'],
                'keys': info['db0']['keys'],
                'expires': info['db0']['expires']
            }
        except Exception as e:
            if self.logger:
                self.logger.error("cache_stats_error", error=str(e))
            return {} 