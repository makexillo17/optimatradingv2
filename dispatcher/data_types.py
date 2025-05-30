from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum, auto

class ModuleType(Enum):
    """Tipos de módulos soportados"""
    BROKER_BEHAVIOR = auto()
    CARRY_TRADE = auto()
    DYNAMIC_HEDGING = auto()
    LIQUIDITY_PROVISION = auto()
    MARKET_MAKING = auto()
    PAIRS_TRADING = auto()
    SMC_ICT = auto()
    STAT_ARB = auto()
    MOMENTUM = auto()
    VOLATILITY = auto()

@dataclass
class ModuleConfig:
    """Configuración de un módulo"""
    module_type: ModuleType
    required_data: List[str]  # Lista de tipos de datos requeridos
    update_interval: int  # Intervalo de actualización en segundos
    priority: int  # Prioridad del módulo (1-10)
    timeout: int  # Timeout en segundos
    enabled: bool = True

@dataclass
class ModuleResult:
    """Resultado del análisis de un módulo"""
    module_type: ModuleType
    timestamp: datetime
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

@dataclass
class DispatchResult:
    """Resultado completo de un ciclo de dispatch"""
    timestamp: datetime
    results: List[ModuleResult]
    metadata: Dict[str, Any]
    
    @property
    def successful_modules(self) -> int:
        """Número de módulos que ejecutaron correctamente"""
        return sum(1 for r in self.results if r.success)
    
    @property
    def failed_modules(self) -> int:
        """Número de módulos que fallaron"""
        return sum(1 for r in self.results if not r.success)
    
    @property
    def total_execution_time(self) -> float:
        """Tiempo total de ejecución"""
        return sum(r.execution_time for r in self.results)
    
    def get_result_by_module(self, module_type: ModuleType) -> Optional[ModuleResult]:
        """Obtiene el resultado de un módulo específico"""
        for result in self.results:
            if result.module_type == module_type:
                return result
        return None 