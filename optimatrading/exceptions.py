"""
Excepciones personalizadas para Optimatrading
"""

class OptimatradingError(Exception):
    """Clase base para excepciones de Optimatrading"""
    pass

class ValidationError(OptimatradingError):
    """Error en la validación de datos"""
    pass

class ConfigurationError(OptimatradingError):
    """Error en la configuración del sistema"""
    pass

class ConnectionError(OptimatradingError):
    """Error de conexión con servicios externos"""
    pass

class APIError(OptimatradingError):
    """Error en llamadas a APIs externas"""
    def __init__(self, message: str, provider: str, status_code: int = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(f"{provider} API Error: {message} (Status: {status_code})")

class DataError(OptimatradingError):
    """Error en datos de entrada o procesamiento"""
    pass

class CacheError(OptimatradingError):
    """Error en operaciones de caché"""
    pass

class ModuleError(OptimatradingError):
    """Error en módulos de análisis"""
    def __init__(self, message: str, module_name: str):
        self.module_name = module_name
        super().__init__(f"Error en módulo {module_name}: {message}")

class ConsensusError(OptimatradingError):
    """Error en el sistema de consenso"""
    pass

class CalculationError(OptimatradingError):
    """Error en cálculos matemáticos"""
    pass 