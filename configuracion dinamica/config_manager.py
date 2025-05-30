import yaml
from typing import Dict, Any, Optional
import os
from datetime import datetime
import json
from logging.logger_manager import LoggerManager
from cache.cache_manager import CacheManager

class ConfigManager:
    def __init__(
        self,
        config_dir: str = "config",
        logger_manager: Optional[LoggerManager] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        self.config_dir = config_dir
        self.logger = logger_manager.get_logger("ConfigManager") if logger_manager else None
        self.cache = cache_manager
        self.config_cache = {}
        self._load_all_configs()
        
    def _load_all_configs(self):
        """Carga todas las configuraciones iniciales"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            
            # Cargar configuraciones base
            self._load_config_file("base_config.yml")
            
            # Cargar configuraciones específicas
            config_files = [
                f for f in os.listdir(self.config_dir)
                if f.endswith('.yml') and f != "base_config.yml"
            ]
            
            for config_file in config_files:
                self._load_config_file(config_file)
                
            if self.logger:
                self.logger.info(
                    "configs_loaded",
                    config_count=len(config_files) + 1
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error("config_load_error", error=str(e))
            raise
            
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """
        Carga un archivo de configuración
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Configuración cargada
        """
        try:
            filepath = os.path.join(self.config_dir, filename)
            
            if not os.path.exists(filepath):
                if self.logger:
                    self.logger.warning(
                        "config_file_not_found",
                        filename=filename
                    )
                return {}
                
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
                
            # Almacenar en caché local
            config_name = os.path.splitext(filename)[0]
            self.config_cache[config_name] = config
            
            # Almacenar en Redis si está disponible
            if self.cache:
                self.cache.set(
                    f"config:{config_name}",
                    config,
                    namespace="config"
                )
                
            return config
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_file_load_error",
                    filename=filename,
                    error=str(e)
                )
            return {}
            
    def get_config(
        self,
        config_name: str,
        default: Any = None
    ) -> Dict[str, Any]:
        """
        Obtiene una configuración
        
        Args:
            config_name: Nombre de la configuración
            default: Valor por defecto
            
        Returns:
            Configuración solicitada
        """
        try:
            # Intentar obtener de caché Redis
            if self.cache:
                config = self.cache.get(
                    f"config:{config_name}",
                    namespace="config"
                )
                if config:
                    return config
                    
            # Intentar obtener de caché local
            config = self.config_cache.get(config_name)
            if config:
                return config
                
            # Intentar cargar del archivo
            config = self._load_config_file(f"{config_name}.yml")
            if config:
                return config
                
            return default
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_get_error",
                    config_name=config_name,
                    error=str(e)
                )
            return default
            
    def update_config(
        self,
        config_name: str,
        updates: Dict[str, Any],
        persist: bool = True
    ) -> bool:
        """
        Actualiza una configuración
        
        Args:
            config_name: Nombre de la configuración
            updates: Actualizaciones a aplicar
            persist: Si se debe persistir a disco
            
        Returns:
            True si se actualizó correctamente
        """
        try:
            current_config = self.get_config(config_name, {})
            
            # Actualizar configuración
            self._deep_update(current_config, updates)
            
            # Actualizar caché local
            self.config_cache[config_name] = current_config
            
            # Actualizar Redis si está disponible
            if self.cache:
                self.cache.set(
                    f"config:{config_name}",
                    current_config,
                    namespace="config"
                )
                
            # Persistir si se solicita
            if persist:
                self._save_config_file(config_name, current_config)
                
            if self.logger:
                self.logger.info(
                    "config_updated",
                    config_name=config_name,
                    updates=updates,
                    persisted=persist
                )
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_update_error",
                    config_name=config_name,
                    error=str(e)
                )
            return False
            
    def _save_config_file(
        self,
        config_name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Guarda una configuración a disco
        
        Args:
            config_name: Nombre de la configuración
            config: Configuración a guardar
            
        Returns:
            True si se guardó correctamente
        """
        try:
            filepath = os.path.join(self.config_dir, f"{config_name}.yml")
            
            # Crear backup
            if os.path.exists(filepath):
                backup_path = f"{filepath}.{datetime.now():%Y%m%d_%H%M%S}.bak"
                os.rename(filepath, backup_path)
                
            # Guardar nueva configuración
            with open(filepath, 'w') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_save_error",
                    config_name=config_name,
                    error=str(e)
                )
            return False
            
    def _deep_update(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> None:
        """
        Actualiza un diccionario de forma recursiva
        
        Args:
            base: Diccionario base
            updates: Actualizaciones a aplicar
        """
        for key, value in updates.items():
            if (
                key in base and
                isinstance(base[key], dict) and
                isinstance(value, dict)
            ):
                self._deep_update(base[key], value)
            else:
                base[key] = value
                
    def validate_config(
        self,
        config_name: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida una configuración contra un esquema
        
        Args:
            config_name: Nombre de la configuración
            schema: Esquema de validación
            
        Returns:
            Diccionario con errores de validación
        """
        try:
            config = self.get_config(config_name)
            if not config:
                return {"error": "Configuración no encontrada"}
                
            errors = {}
            self._validate_dict(config, schema, "", errors)
            
            if errors and self.logger:
                self.logger.warning(
                    "config_validation_errors",
                    config_name=config_name,
                    errors=errors
                )
                
            return errors
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "config_validation_error",
                    config_name=config_name,
                    error=str(e)
                )
            return {"error": str(e)}
            
    def _validate_dict(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
        errors: Dict[str, str]
    ) -> None:
        """
        Valida un diccionario contra un esquema de forma recursiva
        
        Args:
            config: Configuración a validar
            schema: Esquema de validación
            path: Ruta actual en la validación
            errors: Diccionario para almacenar errores
        """
        for key, value_schema in schema.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in config:
                if value_schema.get("required", False):
                    errors[current_path] = "Campo requerido no encontrado"
                continue
                
            value = config[key]
            
            # Validar tipo
            expected_type = value_schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                errors[current_path] = f"Tipo inválido. Esperado: {expected_type.__name__}"
                continue
                
            # Validar rango numérico
            if isinstance(value, (int, float)):
                min_val = value_schema.get("min")
                max_val = value_schema.get("max")
                
                if min_val is not None and value < min_val:
                    errors[current_path] = f"Valor menor al mínimo ({min_val})"
                if max_val is not None and value > max_val:
                    errors[current_path] = f"Valor mayor al máximo ({max_val})"
                    
            # Validar longitud de strings
            if isinstance(value, str):
                min_len = value_schema.get("min_length")
                max_len = value_schema.get("max_length")
                
                if min_len is not None and len(value) < min_len:
                    errors[current_path] = f"Longitud menor a la mínima ({min_len})"
                if max_len is not None and len(value) > max_len:
                    errors[current_path] = f"Longitud mayor a la máxima ({max_len})"
                    
            # Validar valores permitidos
            allowed_values = value_schema.get("allowed_values")
            if allowed_values is not None and value not in allowed_values:
                errors[current_path] = f"Valor no permitido. Permitidos: {allowed_values}"
                
            # Validar subdiccionarios
            if isinstance(value, dict) and "properties" in value_schema:
                self._validate_dict(
                    value,
                    value_schema["properties"],
                    current_path,
                    errors
                ) 