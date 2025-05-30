import logging
import structlog
from typing import Optional, Dict, Any
import os
from datetime import datetime
import json
from prometheus_client import Counter, Histogram

class LoggerManager:
    def __init__(self, app_name: str = "optimatrading"):
        self.app_name = app_name
        self.log_dir = "logs"
        self.metrics = self._setup_metrics()
        self._setup_logging()
        
    def _setup_metrics(self) -> Dict[str, Any]:
        """Configura métricas para monitoreo"""
        return {
            'log_entries': Counter(
                'log_entries_total',
                'Total number of log entries',
                ['level', 'module']
            ),
            'processing_time': Histogram(
                'processing_time_seconds',
                'Time spent processing requests',
                ['module', 'operation']
            )
        }
        
    def _setup_logging(self):
        """Configura el sistema de logging"""
        # Crear directorio de logs si no existe
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configurar procesadores de structlog
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            self._add_app_info,
            self._count_log_entries,
            structlog.processors.JSONRenderer()
        ]
        
        # Configurar logger
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configurar handler de archivo
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f"{self.app_name}_{datetime.now():%Y%m%d}.log")
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Configurar handler de consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configurar formato
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Obtener logger raíz y configurar
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    def _add_app_info(self, logger, method_name, event_dict):
        """Agrega información de la aplicación al log"""
        event_dict["app"] = self.app_name
        return event_dict
        
    def _count_log_entries(self, logger, method_name, event_dict):
        """Cuenta entradas de log para métricas"""
        self.metrics['log_entries'].labels(
            level=event_dict.get('level', 'UNKNOWN'),
            module=event_dict.get('module', 'unknown')
        ).inc()
        return event_dict
        
    def get_logger(self, module_name: str) -> structlog.BoundLogger:
        """
        Obtiene un logger configurado para un módulo específico
        
        Args:
            module_name: Nombre del módulo que solicita el logger
            
        Returns:
            Logger configurado
        """
        return structlog.get_logger(module_name)
        
    def log_error(
        self,
        module: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Registra un error con contexto adicional
        
        Args:
            module: Módulo donde ocurrió el error
            error: Excepción capturada
            context: Contexto adicional opcional
        """
        logger = self.get_logger(module)
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'module': module
        }
        
        if context:
            error_data.update(context)
            
        logger.error("error_occurred", **error_data)
        
    def log_metric(
        self,
        module: str,
        operation: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra una métrica de rendimiento
        
        Args:
            module: Módulo que generó la métrica
            operation: Operación realizada
            duration: Duración en segundos
            metadata: Metadatos adicionales opcionales
        """
        self.metrics['processing_time'].labels(
            module=module,
            operation=operation
        ).observe(duration)
        
        if metadata:
            logger = self.get_logger(module)
            logger.info(
                "metric_recorded",
                operation=operation,
                duration=duration,
                **metadata
            )
            
    def log_validation_error(
        self,
        module: str,
        validation_errors: Dict[str, Any]
    ):
        """
        Registra errores de validación
        
        Args:
            module: Módulo donde ocurrió el error
            validation_errors: Diccionario con errores de validación
        """
        logger = self.get_logger(module)
        logger.warning(
            "validation_error",
            module=module,
            errors=validation_errors
        )
        
    def log_api_request(
        self,
        module: str,
        api_name: str,
        endpoint: str,
        response_time: float,
        status_code: int,
        success: bool
    ):
        """
        Registra una llamada a API
        
        Args:
            module: Módulo que realizó la llamada
            api_name: Nombre de la API
            endpoint: Endpoint llamado
            response_time: Tiempo de respuesta en segundos
            status_code: Código de estado HTTP
            success: Si la llamada fue exitosa
        """
        logger = self.get_logger(module)
        logger.info(
            "api_request",
            module=module,
            api_name=api_name,
            endpoint=endpoint,
            response_time=response_time,
            status_code=status_code,
            success=success
        ) 