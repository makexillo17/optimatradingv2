import logging
import yaml
import time
import os
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
    def __init__(self, config_path: Optional[str] = None):
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
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga config desde archivo o usa defaults si no existe."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Error leyendo config: {e}, usando defaults.")
        
        # Configuración por defecto (Salvavidas para la Nube)
        return {
            'logging': {'level': 'INFO'},
            'modules': {
                'broker_behavior': {'enabled': True, 'priority': 5},
                'carry_trade': {'enabled': True, 'priority': 5},
                'dynamic_hedging': {'enabled': True, 'priority': 8},
                'liquidity_provision': {'enabled': True, 'priority': 5},
                'market_making': {'enabled': True, 'priority': 5},
                'pairs_trading': {'enabled': True, 'priority': 5},
                'smc_ict': {'enabled': True, 'priority': 5},
                'stat_arb': {'enabled': True, 'priority': 5},
                'volatility_arb': {'enabled': True, 'priority': 5},  # Se mapea a VOLATILITY
                'yield_anomaly': {'enabled': True, 'priority': 5},  # Se mapea a MOMENTUM si no existe
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('ModuleDispatcher')
        # Configuración básica si no hay config de log
        logging.basicConfig(level=logging.INFO)
        return logger
    
    def _load_module_configs(self) -> Dict[ModuleType, ModuleConfig]:
        modules = {}
        module_configs = self.config.get('modules', {})
        
        # Mapeo de nombres de config a nombres de enum
        name_mapping = {
            'volatility_arb': 'VOLATILITY',
            'yield_anomaly': 'MOMENTUM',  # Usar MOMENTUM como fallback si no existe YIELD_ANOMALY
        }
        
        for module_name, config in module_configs.items():
            try:
                # Aplicar mapeo si existe
                enum_name = name_mapping.get(module_name, module_name).upper()
                
                # Intenta matchear string con Enum
                if hasattr(ModuleType, enum_name):
                    module_type = ModuleType[enum_name]
                    modules[module_type] = ModuleConfig(
                        module_type=module_type,
                        required_data=config.get('required_data', []),
                        update_interval=config.get('update_interval', 60),
                        priority=config.get('priority', 5),
                        timeout=config.get('timeout', 30),
                        enabled=config.get('enabled', True)
                    )
                else:
                    self.logger.warning(f"No se encontró enum para módulo: {module_name} (buscado como {enum_name})")
            except KeyError:
                self.logger.error(f"Módulo desconocido: {module_name}")
                
        return modules
    
    def register_module(
        self,
        module_type: ModuleType,
        callback: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> None:
        if module_type not in self.modules:
            # Si no estaba en config, lo registramos dinámicamente con defaults
            self.modules[module_type] = ModuleConfig(
                module_type=module_type,
                required_data=[],
                update_interval=60,
                priority=5,
                timeout=30,
                enabled=True
            )
            
        self.callbacks[module_type] = callback
        self.queues[module_type] = Queue()
        self.last_execution[module_type] = datetime.min
        self.logger.info(f"Módulo registrado: {module_type.name}")

    def _should_process_module(self, module_type: ModuleType, current_time: datetime) -> bool:
        if not self.modules[module_type].enabled:
            return False
        last_exec = self.last_execution.get(module_type, datetime.min)
        interval = self.modules[module_type].update_interval
        return (current_time - last_exec).total_seconds() >= interval

    def _process_module(self, module_type: ModuleType, data: Dict[str, Any]) -> ModuleResult:
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
            
            # Ejecutar callback
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
            self.logger.error(f"Error en {module_type.name}: {e}")
            return ModuleResult(
                module_type=module_type,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def dispatch_to_modules(self, data: Dict[str, Any]) -> DispatchResult:
        start_time = time.time()
        current_time = datetime.now()
        results = []
        
        # Filtrar módulos activos
        active_modules = [
            m_type for m_type in self.callbacks.keys()
            if self._should_process_module(m_type, current_time)
        ]
        
        # Ordenar por prioridad
        active_modules.sort(
            key=lambda m: self.modules[m].priority,
            reverse=True
        )
        
        # Ejecutar en paralelo
        with ThreadPoolExecutor(max_workers=len(active_modules) or 1) as executor:
            future_to_module = {
                executor.submit(self._process_module, m_type, data): m_type
                for m_type in active_modules
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
