"""
Utilidades para monitoreo de rendimiento
"""

import time
import functools
import threading
from typing import Any, Callable, Dict, Optional
from datetime import datetime, timedelta

from prometheus_client import Counter, Histogram, Gauge
from structlog import get_logger

# Métricas Prometheus
function_calls = Counter(
    'optimatrading_function_calls_total',
    'Total de llamadas a funciones',
    ['module', 'function']
)

function_errors = Counter(
    'optimatrading_function_errors_total',
    'Total de errores en funciones',
    ['module', 'function', 'error_type']
)

function_duration = Histogram(
    'optimatrading_function_duration_seconds',
    'Duración de ejecución de funciones',
    ['module', 'function']
)

memory_usage = Gauge(
    'optimatrading_memory_usage_bytes',
    'Uso de memoria por módulo',
    ['module']
)

class PerformanceMonitor:
    """Monitor de rendimiento para módulos"""
    
    def __init__(self):
        self.logger = get_logger()
        self._stats = {}
        self._lock = threading.Lock()
        
    def monitor(
        self,
        module: str,
        threshold_ms: Optional[float] = None,
        log_args: bool = False
    ) -> Callable:
        """
        Decorador para monitorear rendimiento de funciones.
        
        Args:
            module: Nombre del módulo
            threshold_ms: Umbral para alertar en ms
            log_args: Si se deben loguear argumentos
            
        Returns:
            Función decorada
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                
                # Incrementar contador de llamadas
                function_calls.labels(
                    module=module,
                    function=func.__name__
                ).inc()
                
                try:
                    # Ejecutar función
                    result = func(*args, **kwargs)
                    
                    # Registrar duración
                    duration = time.time() - start_time
                    function_duration.labels(
                        module=module,
                        function=func.__name__
                    ).observe(duration)
                    
                    # Verificar umbral
                    if threshold_ms and duration * 1000 > threshold_ms:
                        self.logger.warning(
                            "function_duration_exceeded_threshold",
                            module=module,
                            function=func.__name__,
                            duration_ms=duration * 1000,
                            threshold_ms=threshold_ms
                        )
                        
                    # Actualizar estadísticas
                    with self._lock:
                        if module not in self._stats:
                            self._stats[module] = {
                                'calls': 0,
                                'errors': 0,
                                'total_time': 0,
                                'min_time': float('inf'),
                                'max_time': 0
                            }
                            
                        stats = self._stats[module]
                        stats['calls'] += 1
                        stats['total_time'] += duration
                        stats['min_time'] = min(stats['min_time'], duration)
                        stats['max_time'] = max(stats['max_time'], duration)
                        
                    # Loguear detalles si está habilitado
                    if log_args:
                        self.logger.debug(
                            "function_call_completed",
                            module=module,
                            function=func.__name__,
                            duration_ms=duration * 1000,
                            args=args,
                            kwargs=kwargs
                        )
                        
                    return result
                    
                except Exception as e:
                    # Registrar error
                    function_errors.labels(
                        module=module,
                        function=func.__name__,
                        error_type=type(e).__name__
                    ).inc()
                    
                    with self._lock:
                        if module in self._stats:
                            self._stats[module]['errors'] += 1
                            
                    self.logger.error(
                        "function_call_error",
                        module=module,
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    raise
                    
            return wrapper
        return decorator
        
    def get_stats(self, module: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene estadísticas de rendimiento.
        
        Args:
            module: Módulo específico o None para todos
            
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            if module:
                return self._stats.get(module, {}).copy()
            return self._stats.copy()
            
    def reset_stats(self, module: Optional[str] = None) -> None:
        """
        Reinicia estadísticas de rendimiento.
        
        Args:
            module: Módulo específico o None para todos
        """
        with self._lock:
            if module:
                self._stats.pop(module, None)
            else:
                self._stats.clear()
                
class ResourceMonitor:
    """Monitor de recursos del sistema"""
    
    def __init__(
        self,
        check_interval: float = 60.0,
        memory_threshold: float = 0.9
    ):
        """
        Inicializa el monitor.
        
        Args:
            check_interval: Intervalo de chequeo en segundos
            memory_threshold: Umbral de uso de memoria (0-1)
        """
        self.logger = get_logger()
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self._stop_event = threading.Event()
        self._monitor_thread = None
        
    def start(self) -> None:
        """Inicia el monitoreo"""
        if self._monitor_thread is not None:
            return
            
        def monitor_loop():
            while not self._stop_event.is_set():
                try:
                    self._check_resources()
                except Exception as e:
                    self.logger.error(
                        "resource_check_error",
                        error=str(e)
                    )
                finally:
                    self._stop_event.wait(self.check_interval)
                    
        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            name="resource_monitor",
            daemon=True
        )
        self._monitor_thread.start()
        
    def stop(self) -> None:
        """Detiene el monitoreo"""
        if self._monitor_thread is None:
            return
            
        self._stop_event.set()
        self._monitor_thread.join()
        self._monitor_thread = None
        
    def _check_resources(self) -> None:
        """Verifica uso de recursos"""
        import psutil
        
        # Verificar memoria
        memory = psutil.virtual_memory()
        memory_usage.set(memory.used)
        
        if memory.percent / 100 > self.memory_threshold:
            self.logger.warning(
                "high_memory_usage",
                used_percent=memory.percent,
                threshold_percent=self.memory_threshold * 100
            )
            
        # Verificar CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            self.logger.warning(
                "high_cpu_usage",
                cpu_percent=cpu_percent
            ) 