"""
Sistema de validación de datos para Optimatrading
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from decimal import Decimal

from pydantic import ValidationError

from ..exceptions import ValidationError as OptimatradingValidationError
from ..logging import LoggerManager
from .models import (
    MarketData,
    ModuleConfig,
    ModuleResult,
    ConsensusConfig,
    ValidationConfig
)

class DataValidator:
    """
    Validador de datos para Optimatrading.
    
    Proporciona funcionalidades de validación para diferentes
    tipos de datos y estructuras utilizadas en el sistema.
    """
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None,
        config: Optional[ValidationConfig] = None
    ):
        """
        Inicializa el validador.
        
        Args:
            logger_manager: Gestor de logging
            config: Configuración de validación
        """
        self.logger = logger_manager.get_logger("DataValidator") if logger_manager else None
        self.config = config
        
    def validate_market_data(
        self,
        data: Union[Dict[str, Any], pd.DataFrame],
        required_fields: Optional[List[str]] = None,
        max_missing_values: float = 0.1
    ) -> List[str]:
        """
        Valida datos de mercado.
        
        Args:
            data: Datos a validar
            required_fields: Campos requeridos
            max_missing_values: Máximo de valores faltantes permitidos
            
        Returns:
            Lista de errores encontrados
        """
        errors = []
        
        try:
            # Convertir DataFrame a diccionario si es necesario
            if isinstance(data, pd.DataFrame):
                if len(data) == 0:
                    raise OptimatradingValidationError("DataFrame vacío")
                data = data.to_dict('records')[0]
                
            # Validar usando Pydantic
            try:
                MarketData(**data)
            except ValidationError as e:
                for error in e.errors():
                    errors.append(f"Error en {error['loc']}: {error['msg']}")
                    
            # Validar campos requeridos
            required = required_fields or self.config.required_fields if self.config else [
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ]
            
            for field in required:
                if field not in data:
                    errors.append(f"Campo requerido faltante: {field}")
                    
            # Validar valores faltantes
            missing_count = sum(1 for v in data.values() if pd.isna(v))
            if missing_count / len(data) > max_missing_values:
                errors.append(
                    f"Demasiados valores faltantes: {missing_count/len(data):.2%}"
                )
                
            # Validaciones específicas
            self._validate_price_consistency(data, errors)
            self._validate_timestamp(data, errors)
            
            if errors and self.logger:
                self.logger.warning(
                    "market_data_validation_errors",
                    errors=errors,
                    data=data
                )
                
            return errors
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "market_data_validation_error",
                    error=str(e),
                    data=data
                )
            raise OptimatradingValidationError(f"Error validando datos: {str(e)}")
            
    def validate_module_result(
        self,
        result: Dict[str, Any],
        module_name: str
    ) -> None:
        """
        Valida el resultado de un módulo.
        
        Args:
            result: Resultado a validar
            module_name: Nombre del módulo
            
        Raises:
            OptimatradingValidationError: Si la validación falla
        """
        try:
            ModuleResult(**result)
        except ValidationError as e:
            errors = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
            raise OptimatradingValidationError(
                f"Resultado inválido del módulo {module_name}: {', '.join(errors)}"
            )
            
    def validate_numeric_array(
        self,
        arr: Union[List[float], np.ndarray],
        name: str,
        min_length: int = 1,
        allow_zeros: bool = True,
        allow_negatives: bool = True
    ) -> None:
        """
        Valida un array numérico.
        
        Args:
            arr: Array a validar
            name: Nombre del array para mensajes
            min_length: Longitud mínima requerida
            allow_zeros: Si se permiten ceros
            allow_negatives: Si se permiten negativos
            
        Raises:
            OptimatradingValidationError: Si la validación falla
        """
        try:
            if len(arr) < min_length:
                raise OptimatradingValidationError(
                    f"{name} debe tener al menos {min_length} elementos"
                )
                
            if not allow_zeros and np.any(np.array(arr) == 0):
                raise OptimatradingValidationError(
                    f"{name} no puede contener ceros"
                )
                
            if not allow_negatives and np.any(np.array(arr) < 0):
                raise OptimatradingValidationError(
                    f"{name} no puede contener valores negativos"
                )
                
        except Exception as e:
            if not isinstance(e, OptimatradingValidationError):
                raise OptimatradingValidationError(
                    f"Error validando {name}: {str(e)}"
                )
            raise
            
    def _validate_price_consistency(
        self,
        data: Dict[str, Any],
        errors: List[str]
    ) -> None:
        """Valida consistencia de precios"""
        try:
            high = Decimal(str(data.get('high', 0)))
            low = Decimal(str(data.get('low', 0)))
            open_price = Decimal(str(data.get('open', 0)))
            close = Decimal(str(data.get('close', 0)))
            
            if high < low:
                errors.append(f"High ({high}) menor que low ({low})")
                
            if high < open_price or high < close:
                errors.append("High debe ser mayor o igual que open y close")
                
            if low > open_price or low > close:
                errors.append("Low debe ser menor o igual que open y close")
                
        except (TypeError, ValueError) as e:
            errors.append(f"Error en conversión de precios: {str(e)}")
            
    def _validate_timestamp(
        self,
        data: Dict[str, Any],
        errors: List[str]
    ) -> None:
        """Valida timestamp"""
        try:
            ts = data.get('timestamp')
            if isinstance(ts, str):
                datetime.fromisoformat(ts.replace('Z', '+00:00'))
            elif not isinstance(ts, (datetime, pd.Timestamp)):
                errors.append("Timestamp en formato inválido")
                
            # Validar que no sea futuro
            if isinstance(ts, (datetime, pd.Timestamp)) and ts > datetime.now():
                errors.append("Timestamp en el futuro")
                
        except ValueError as e:
            errors.append(f"Error en timestamp: {str(e)}")
            
    @classmethod
    def from_config(
        cls,
        config: ValidationConfig,
        logger_manager: Optional[LoggerManager] = None
    ) -> 'DataValidator':
        """
        Crea una instancia desde configuración.
        
        Args:
            config: Configuración de validación
            logger_manager: Gestor de logging
            
        Returns:
            Instancia de DataValidator
        """
        return cls(
            logger_manager=logger_manager,
            config=config
        ) 