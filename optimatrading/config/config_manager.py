"""
Sistema de configuración dinámica para Optimatrading
"""

# Librerías estándar
import os
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import shutil
from pathlib import Path
import threading
from copy import deepcopy

# Librerías externas
import yaml
from yaml.parser import ParserError

# Módulos locales
from ..logging import LoggerManager
from ..cache import CacheManager

class ConfigValidationError(Exception):
    """Error de validación de configuración"""
    pass

class ConfigManager:
    """
    Gestor de configuración dinámica para Optimatrading.
    
    Proporciona funcionalidades de configuración basada en YAML con
    soporte para actualizaciones en tiempo real y validación de esquemas.
    """
    
    # Esquema de validación para la configuración
    REQUIRED_SECTIONS = {
        'apis': ['binance'],
        'redis': ['host', 'port', 'db'],
        'logging': ['console', 'file'],
        'modules': [],  # Al menos un módulo debe estar habilitado
        'consensus': ['min_confidence', 'min_agreement'],
        'validation': ['required_fields'],
        'metrics': ['enabled'],
        'system': ['environment']
    }
    
    def __init__(
        self,
        config_dir: str = "config",
        config_file: str = "config.yaml",
        logger_manager: Optional[LoggerManager] = None,
        cache_manager: Optional[CacheManager] = None,
        backup_dir: Optional[str] = None,
        auto_reload: bool = True,
        validate_schema: bool = True,
        env_prefix: str = "OPTIMATRADING_"
    ):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_dir: Directorio de archivos de configuración
            config_file: Nombre del archivo de configuración
            logger_manager: Instancia del gestor de logging
            cache_manager: Instancia del gestor de caché
            backup_dir: Directorio para backups
            auto_reload: Si se debe recargar automáticamente
            validate_schema: Si se debe validar el esquema
            env_prefix: Prefijo para variables de entorno
        """
        self.config_dir = Path(config_dir)
        self.config_file = self.config_dir / config_file
        self.logger = logger_manager.get_logger("ConfigManager") if logger_manager else None
        self.cache = cache_manager
        self.backup_dir = Path(backup_dir) if backup_dir else self.config_dir / "backups"
        self.auto_reload = auto_reload
        self.validate_schema = validate_schema
        self.env_prefix = env_prefix
        self.config_cache = {}
        self._config_lock = threading.Lock()
        self._last_modified = None
        
        # Crear directorios necesarios
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar configuración inicial
        self._load_config()
        
    def get(
        self,
        path: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Obtiene un valor de configuración por su ruta.
        
        Args:
            path: Ruta del valor (e.g. "redis.host")
            default: Valor por defecto si no existe
            required: Si el valor es requerido
            
        Returns:
            Valor de configuración
            
        Raises:
            ConfigValidationError: Si el valor es requerido y no existe
        """
        try:
            value = self._get_nested(self.config_cache, path.split('.'))
            if value is not None:
                return value
                
            # Buscar en variables de entorno
            env_value = self._get_from_env(path)
            if env_value is not None:
                return env_value
                
            if required:
                raise ConfigValidationError(f"Valor requerido no encontrado: {path}")
                
            return default
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_get_error",
                    path=path,
                    error=str(e)
                )
            if required:
                raise
            return default
            
    def get_all(self) -> Dict[str, Any]:
        """
        Obtiene toda la configuración.
        
        Returns:
            Copia de la configuración completa
        """
        with self._config_lock:
            return deepcopy(self.config_cache)
            
    def reload(self) -> bool:
        """
        Recarga la configuración desde el archivo.
        
        Returns:
            True si se recargó correctamente
        """
        try:
            with self._config_lock:
                self._load_config()
            return True
        except Exception as e:
            if self.logger:
                self.logger.error("config_reload_error", error=str(e))
            return False
            
    def _load_config(self):
        """Carga la configuración desde el archivo"""
        try:
            if not self.config_file.exists():
                example_file = self.config_dir / "config.yaml.example"
                if example_file.exists():
                    shutil.copy(example_file, self.config_file)
                else:
                    raise FileNotFoundError(
                        f"No se encontró el archivo de configuración: {self.config_file}"
                    )
                    
            # Verificar si el archivo ha cambiado
            current_mtime = os.path.getmtime(self.config_file)
            if (
                not self.auto_reload and
                self._last_modified and
                current_mtime <= self._last_modified
            ):
                return
                
            # Cargar configuración
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validar esquema
            if self.validate_schema:
                self._validate_config(config)
                
            # Crear backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"config_{timestamp}.yaml"
            shutil.copy(self.config_file, backup_file)
            
            # Actualizar caché
            self.config_cache = config
            self._last_modified = current_mtime
            
            if self.logger:
                self.logger.info("config_loaded", filename=str(self.config_file))
                
        except Exception as e:
            if self.logger:
                self.logger.error("config_load_error", error=str(e))
            raise
            
    def _validate_config(self, config: Dict[str, Any]):
        """
        Valida la configuración contra el esquema.
        
        Args:
            config: Configuración a validar
            
        Raises:
            ConfigValidationError: Si la configuración es inválida
        """
        errors = []
        
        # Validar secciones requeridas
        for section, required_fields in self.REQUIRED_SECTIONS.items():
            if section not in config:
                errors.append(f"Sección requerida no encontrada: {section}")
                continue
                
            for field in required_fields:
                if field not in config[section]:
                    errors.append(
                        f"Campo requerido no encontrado: {section}.{field}"
                    )
                    
        # Validar que al menos un módulo esté habilitado
        if 'modules' in config:
            enabled_modules = [
                name for name, settings in config['modules'].items()
                if settings.get('enabled', False)
            ]
            if not enabled_modules:
                errors.append("Al menos un módulo debe estar habilitado")
                
        if errors:
            raise ConfigValidationError("\n".join(errors))
            
    def _get_nested(
        self,
        data: Dict[str, Any],
        keys: List[str]
    ) -> Any:
        """
        Obtiene un valor anidado del diccionario.
        
        Args:
            data: Diccionario de datos
            keys: Lista de claves para acceder al valor
            
        Returns:
            Valor encontrado o None
        """
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                return None
            data = data[key]
        return data
        
    def _get_from_env(self, path: str) -> Optional[Any]:
        """
        Obtiene un valor de las variables de entorno.
        
        Args:
            path: Ruta del valor
            
        Returns:
            Valor encontrado o None
        """
        env_key = self.env_prefix + path.upper().replace('.', '_')
        value = os.environ.get(env_key)
        
        if value is None:
            return None
            
        # Intentar convertir a tipo apropiado
        try:
            # Intentar como número
            if value.isdigit():
                return int(value)
            try:
                return float(value)
            except ValueError:
                pass
                
            # Intentar como booleano
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
                
            # Intentar como lista o diccionario
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
            # Devolver como string
            return value
            
        except Exception:
            return value 