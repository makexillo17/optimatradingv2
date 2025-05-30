"""
Sistema de logging centralizado para Optimatrading
"""

# Librerías estándar
import os
import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import threading
from logging.handlers import RotatingFileHandler

# Librerías externas
import structlog
from prometheus_client import Counter, Histogram

# Módulos locales
from ..exceptions import OptimatradingError
from ..validation.models import LogConfig

# Métricas Prometheus
log_entries = Counter(
    'optimatrading_log_entries_total',
    'Total de entradas de log',
    ['level', 'module']
)

log_processing_time = Histogram(
    'optimatrading_log_processing_seconds',
    'Tiempo de procesamiento de logs',
    ['level']
)

class StructuredFormatter(logging.Formatter):
    """Formateador para logs estructurados"""
    
    def format(self, record):
        """Formatea el registro de log en formato JSON estructurado"""
        # Datos base
        data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Agregar datos de excepción si existen
        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)
            
        # Agregar datos extra
        if hasattr(record, 'data'):
            data.update(record.data)
            
        return json.dumps(data)

class LoggerManager:
    """
    Gestor centralizado de logging para Optimatrading.
    
    Proporciona funcionalidades de logging estructurado, métricas
    y múltiples handlers para diferentes niveles y destinos.
    """
    
    def __init__(
        self,
        app_name: str = "optimatrading",
        log_dir: str = "logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        max_size_mb: int = 100,
        backup_count: int = 5,
        environment: str = "production"
    ):
        """
        Inicializa el gestor de logging.
        
        Args:
            app_name: Nombre de la aplicación
            log_dir: Directorio para archivos de log
            console_level: Nivel de logging para consola
            file_level: Nivel de logging para archivos
            max_size_mb: Tamaño máximo del archivo de log en MB
            backup_count: Número de backups a mantener
            environment: Entorno de ejecución
        """
        self.app_name = app_name
        self.log_dir = Path(log_dir)
        self.environment = environment
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging estructurado
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.render_to_log_kwargs,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configurar handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        if environment == "development":
            console_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        else:
            console_handler.setFormatter(StructuredFormatter())
            
        # Configurar handler de archivo
        log_file = self.log_dir / f"{app_name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(StructuredFormatter())
        
        # Configurar logger raíz
        self.root_logger = logging.getLogger(app_name)
        self.root_logger.setLevel(logging.DEBUG)
        self.root_logger.addHandler(console_handler)
        self.root_logger.addHandler(file_handler)
        
        # Diccionario para cachear loggers
        self._loggers = {}
        self._logger_lock = threading.Lock()
        
    def get_logger(self, name: str) -> logging.Logger:
        """
        Obtiene un logger configurado.
        
        Args:
            name: Nombre del logger
            
        Returns:
            Logger configurado
        """
        with self._logger_lock:
            if name not in self._loggers:
                logger = self.root_logger.getChild(name)
                self._loggers[name] = logger
            return self._loggers[name]
            
    @staticmethod
    def add_context(logger: logging.Logger, **kwargs) -> None:
        """
        Agrega contexto al logger.
        
        Args:
            logger: Logger a modificar
            **kwargs: Datos de contexto
        """
        adapter = logging.LoggerAdapter(logger, kwargs)
        return adapter
        
    def log_exception(
        self,
        logger: logging.Logger,
        exc: Exception,
        level: str = "ERROR",
        **kwargs
    ) -> None:
        """
        Registra una excepción con contexto adicional.
        
        Args:
            logger: Logger a utilizar
            exc: Excepción a registrar
            level: Nivel de log
            **kwargs: Contexto adicional
        """
        exc_info = (type(exc), exc, exc.__traceback__)
        context = {
            'exception_type': exc.__class__.__name__,
            'exception_message': str(exc)
        }
        context.update(kwargs)
        
        logger.log(
            getattr(logging, level.upper()),
            str(exc),
            exc_info=exc_info,
            extra={'data': context}
        )
        
        # Incrementar contador de métricas
        log_entries.labels(
            level=level,
            module=logger.name
        ).inc()
        
    @classmethod
    def from_config(cls, config: LogConfig, app_name: str = "optimatrading") -> 'LoggerManager':
        """
        Crea una instancia desde configuración.
        
        Args:
            config: Configuración de logging
            app_name: Nombre de la aplicación
            
        Returns:
            Instancia de LoggerManager
        """
        return cls(
            app_name=app_name,
            log_dir=config.directory or "logs",
            console_level=config.level,
            file_level=config.level,
            max_size_mb=config.max_size_mb or 100,
            backup_count=config.backup_count or 5
        ) 