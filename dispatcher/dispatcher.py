import logging
import yaml
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from queue import Queue, Empty
from threading import Event, Lock

from .data_types import (
    ModuleType,
    ModuleConfig,
    ModuleResult,
    DispatchResult
)

class ModuleDispatcher:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.modules: Dict[ModuleType, ModuleConfig] = self._load_module_configs()
        self.callbacks: Dict[ModuleType, Callable] = {}
        self.queues: Dict[ModuleType, Queue] = {}
        self.running = False
        self.worker_threads = []
        self.last_execution: Dict[ModuleType, datetime] = {}
        self.results_lock = Lock()
        self.stop_event = Event()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ModuleDispatcher')
        log_config = self.config.get('logging', {})
        
        logging.basicConfig(
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            filename=log_config.get('file', 'dispatcher.log')
        )
        
        return logger
    
    def _load_module_configs(self) -> Dict[ModuleType, ModuleConfig]:
        """Carga la configuración de los módulos desde el archivo de configuración"""
        modules = {}
        module_configs = self.config.get('modules', {})
        
        for module_name, config in module_configs.items():
            try:
                module_type = ModuleType[module_name.upper()]
                modules[module_type] = ModuleConfig(
                    module_type=module_type,
                    required_data=config.get('required_data', []),
                    update_interval=config.get('update_interval', 60),
                    priority=config.get('priority', 5),
                    timeout=config.get('timeout', 30),
                    enabled=config.get('enabled', True)
                )
            except KeyError:
                self.logger.error(f"Módulo desconocido en configuración: {module_name}")
                
        return modules
    
    def register_module(
        self,
        module_type: ModuleType,
        callback: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        """Registra un módulo y su callback"""
        if module_type not in self.modules:
            raise ValueError(f"Tipo de módulo no configurado: {module_type}")
            
        self.callbacks[module_type] = callback
        self.queues[module_type] = Queue()
        self.last_execution[module_type] = datetime.min
        self.logger.info(f"Módulo registrado: {module_type.name}")
    
    def _should_process_module(self, module_type: ModuleType, current_time: datetime) -> bool:
        """Determina si un módulo debe procesar datos basado en su intervalo"""
        if not self.modules[module_type].enabled:
            return False
            
        last_exec = self.last_execution.get(module_type, datetime.min)
        interval = self.modules[module_type].update_interval
        
        return (current_time - last_exec).total_seconds() >= interval
    
    def _process_module(
        self,
        module_type: ModuleType,
        data: Dict[str, Any]
    ) -> ModuleResult:
        """Procesa datos para un módulo específico"""
        start_time = time.time()
        config = self.modules[module_type]
        
        try:
            # Verificar que todos los datos requeridos estén presentes
            missing_data = [
                key for key in config.required_data
                if key not in data
            ]
            
            if missing_data:
                raise ValueError(f"Datos faltantes: {missing_data}")
                
            # Ejecutar el callback del módulo con timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.callbacks[module_type], data)
                result = future.result(timeout=config.timeout)
                
            execution_time = time.time() - start_time
            
            with self.results_lock:
                self.last_execution[module_type] = datetime.now()
                
            return ModuleResult(
                module_type=module_type,
                timestamp=datetime.now(),
                success=True,
                data=result,
                execution_time=execution_time,
                metadata={'priority': config.priority}
            )
            
        except TimeoutError:
            self.logger.error(f"Timeout en módulo {module_type.name}")
            return ModuleResult(
                module_type=module_type,
                timestamp=datetime.now(),
                success=False,
                error="Timeout",
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error en módulo {module_type.name}: {str(e)}")
            return ModuleResult(
                module_type=module_type,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def dispatch_to_modules(self, data: Dict[str, Any]) -> DispatchResult:
        """
        Distribuye datos a todos los módulos registrados y recolecta resultados
        
        Args:
            data: Datos a procesar
            
        Returns:
            DispatchResult con los resultados de todos los módulos
        """
        start_time = time.time()
        current_time = datetime.now()
        results = []
        
        # Filtrar módulos que deben ejecutarse
        active_modules = [
            module_type for module_type in self.callbacks.keys()
            if self._should_process_module(module_type, current_time)
        ]
        
        # Ordenar por prioridad
        active_modules.sort(
            key=lambda m: self.modules[m].priority,
            reverse=True
        )
        
        # Procesar módulos en paralelo
        with ThreadPoolExecutor(max_workers=len(active_modules)) as executor:
            future_to_module = {
                executor.submit(self._process_module, module_type, data): module_type
                for module_type in active_modules
            }
            
            for future in as_completed(future_to_module):
                result = future.result()
                results.append(result)
                
                if not result.success:
                    self.logger.warning(
                        f"Módulo {result.module_type.name} falló: {result.error}"
                    )
        
        total_time = time.time() - start_time
        
        return DispatchResult(
            timestamp=current_time,
            results=results,
            metadata={
                'total_time': total_time,
                'active_modules': len(active_modules),
                'total_modules': len(self.callbacks)
            }
        )
    
    def start(self) -> None:
        """Inicia el dispatcher"""
        if not self.callbacks:
            raise RuntimeError("No hay módulos registrados")
            
        self.running = True
        self.stop_event.clear()
        self.logger.info("Dispatcher iniciado")
    
    def stop(self) -> None:
        """Detiene el dispatcher"""
        self.running = False
        self.stop_event.set()
        
        for thread in self.worker_threads:
            thread.join()
            
        self.logger.info("Dispatcher detenido")
    
    def get_module_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de los módulos"""
        stats = {
            'total_modules': len(self.callbacks),
            'enabled_modules': sum(1 for m in self.modules.values() if m.enabled),
            'last_execution': {
                m.name: t.isoformat()
                for m, t in self.last_execution.items()
            },
            'queue_sizes': {
                m.name: q.qsize()
                for m, q in self.queues.items()
            }
        }
        return stats 