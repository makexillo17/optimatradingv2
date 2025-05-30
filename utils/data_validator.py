from typing import Dict, Any, List, Optional, Union, Type
import numpy as np
from datetime import datetime
import pandas as pd
from logging.logger_manager import LoggerManager

class DataValidator:
    def __init__(self, logger_manager: Optional[LoggerManager] = None):
        self.logger = logger_manager.get_logger("DataValidator") if logger_manager else None
        
    def validate_market_data(
        self,
        data: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Valida datos de mercado
        
        Args:
            data: Datos a validar
            required_fields: Campos requeridos
            
        Returns:
            Diccionario con errores encontrados
        """
        errors = {}
        
        if required_fields is None:
            required_fields = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ]
            
        # Validar campos requeridos
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]
        
        if missing_fields:
            errors['missing_fields'] = missing_fields
            
        # Validar tipos de datos
        type_errors = []
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in numeric_fields:
            if field in data and not isinstance(data[field], (int, float)):
                type_errors.append(f"{field} debe ser numérico")
                
        if type_errors:
            errors['type_errors'] = type_errors
            
        # Validar valores lógicos
        logic_errors = []
        
        if all(field in data for field in ['high', 'low', 'open', 'close']):
            if data['high'] < data['low']:
                logic_errors.append("high no puede ser menor que low")
            if data['open'] > data['high'] or data['open'] < data['low']:
                logic_errors.append("open debe estar entre high y low")
            if data['close'] > data['high'] or data['close'] < data['low']:
                logic_errors.append("close debe estar entre high y low")
                
        if logic_errors:
            errors['logic_errors'] = logic_errors
            
        if errors and self.logger:
            self.logger.warning(
                "market_data_validation_errors",
                errors=errors
            )
            
        return errors
        
    def validate_recommendation(
        self,
        recommendation: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Valida una recomendación de trading
        
        Args:
            recommendation: Recomendación a validar
            
        Returns:
            Diccionario con errores encontrados
        """
        errors = {}
        
        # Validar campos requeridos
        required_fields = [
            'recommendation', 'confidence', 'timestamp',
            'justification', 'module_results'
        ]
        
        missing_fields = [
            field for field in required_fields
            if field not in recommendation
        ]
        
        if missing_fields:
            errors['missing_fields'] = missing_fields
            
        # Validar valores permitidos
        if 'recommendation' in recommendation:
            allowed_values = ['LONG', 'SHORT', 'NEUTRAL']
            if recommendation['recommendation'] not in allowed_values:
                errors['invalid_recommendation'] = [
                    f"Valor debe ser uno de: {allowed_values}"
                ]
                
        # Validar rango de confianza
        if 'confidence' in recommendation:
            confidence = recommendation['confidence']
            if not isinstance(confidence, (int, float)):
                errors['invalid_confidence'] = ["Debe ser numérico"]
            elif confidence < 0 or confidence > 1:
                errors['invalid_confidence'] = ["Debe estar entre 0 y 1"]
                
        # Validar resultados de módulos
        if 'module_results' in recommendation:
            module_errors = self._validate_module_results(
                recommendation['module_results']
            )
            if module_errors:
                errors['module_errors'] = module_errors
                
        if errors and self.logger:
            self.logger.warning(
                "recommendation_validation_errors",
                errors=errors
            )
            
        return errors
        
    def _validate_module_results(
        self,
        module_results: Dict[str, Any]
    ) -> List[str]:
        """Valida resultados de módulos individuales"""
        errors = []
        
        for module, result in module_results.items():
            if not isinstance(result, dict):
                errors.append(f"Resultado de {module} debe ser un diccionario")
                continue
                
            required_fields = [
                'recommendation', 'confidence', 'justification'
            ]
            
            missing_fields = [
                field for field in required_fields
                if field not in result
            ]
            
            if missing_fields:
                errors.append(
                    f"Campos faltantes en {module}: {missing_fields}"
                )
                
        return errors
        
    def validate_time_series(
        self,
        data: Union[pd.DataFrame, Dict[str, List[Any]]],
        required_fields: Optional[List[str]] = None,
        min_length: int = 1
    ) -> Dict[str, List[str]]:
        """
        Valida una serie temporal
        
        Args:
            data: Serie temporal a validar
            required_fields: Campos requeridos
            min_length: Longitud mínima requerida
            
        Returns:
            Diccionario con errores encontrados
        """
        errors = {}
        
        # Convertir a DataFrame si es necesario
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                return {'conversion_error': [str(e)]}
                
        # Validar campos requeridos
        if required_fields:
            missing_fields = [
                field for field in required_fields
                if field not in data.columns
            ]
            if missing_fields:
                errors['missing_fields'] = missing_fields
                
        # Validar longitud mínima
        if len(data) < min_length:
            errors['insufficient_length'] = [
                f"Se requieren al menos {min_length} registros"
            ]
            
        # Validar valores faltantes
        na_columns = data.columns[data.isna().any()].tolist()
        if na_columns:
            errors['missing_values'] = [
                f"Valores faltantes en columnas: {na_columns}"
            ]
            
        # Validar tipos de datos
        type_errors = []
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if data[col].dtype not in [np.float64, np.int64]:
                type_errors.append(f"{col} debe ser numérico")
                
        if type_errors:
            errors['type_errors'] = type_errors
            
        if errors and self.logger:
            self.logger.warning(
                "time_series_validation_errors",
                errors=errors
            )
            
        return errors
        
    def validate_input_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Valida datos contra un esquema
        
        Args:
            data: Datos a validar
            schema: Esquema de validación
            
        Returns:
            Diccionario con errores encontrados
        """
        errors = {}
        
        for field, field_schema in schema.items():
            field_errors = []
            
            # Validar presencia si es requerido
            if field_schema.get('required', False) and field not in data:
                field_errors.append("Campo requerido no encontrado")
                errors[field] = field_errors
                continue
                
            if field not in data:
                continue
                
            value = data[field]
            
            # Validar tipo
            expected_type = field_schema.get('type')
            if expected_type and not isinstance(value, expected_type):
                field_errors.append(
                    f"Tipo inválido. Esperado: {expected_type.__name__}"
                )
                
            # Validar rango numérico
            if isinstance(value, (int, float)):
                min_val = field_schema.get('min')
                max_val = field_schema.get('max')
                
                if min_val is not None and value < min_val:
                    field_errors.append(f"Valor menor al mínimo ({min_val})")
                if max_val is not None and value > max_val:
                    field_errors.append(f"Valor mayor al máximo ({max_val})")
                    
            # Validar longitud de strings
            if isinstance(value, str):
                min_len = field_schema.get('min_length')
                max_len = field_schema.get('max_length')
                
                if min_len is not None and len(value) < min_len:
                    field_errors.append(
                        f"Longitud menor a la mínima ({min_len})"
                    )
                if max_len is not None and len(value) > max_len:
                    field_errors.append(
                        f"Longitud mayor a la máxima ({max_len})"
                    )
                    
            # Validar valores permitidos
            allowed_values = field_schema.get('allowed_values')
            if allowed_values is not None and value not in allowed_values:
                field_errors.append(
                    f"Valor no permitido. Permitidos: {allowed_values}"
                )
                
            # Validar formato de fecha
            if field_schema.get('format') == 'datetime':
                try:
                    if isinstance(value, str):
                        datetime.strptime(value, field_schema.get('datetime_format', '%Y-%m-%d %H:%M:%S'))
                    elif not isinstance(value, datetime):
                        field_errors.append("Formato de fecha inválido")
                except ValueError:
                    field_errors.append("Formato de fecha inválido")
                    
            if field_errors:
                errors[field] = field_errors
                
        if errors and self.logger:
            self.logger.warning(
                "schema_validation_errors",
                errors=errors
            )
            
        return errors
        
    def validate_output_format(
        self,
        data: Any,
        expected_format: Union[Type, Dict[str, Type]]
    ) -> Dict[str, List[str]]:
        """
        Valida el formato de salida
        
        Args:
            data: Datos a validar
            expected_format: Formato esperado
            
        Returns:
            Diccionario con errores encontrados
        """
        errors = {}
        
        if isinstance(expected_format, type):
            if not isinstance(data, expected_format):
                errors['format_error'] = [
                    f"Tipo inválido. Esperado: {expected_format.__name__}"
                ]
                
        elif isinstance(expected_format, dict):
            if not isinstance(data, dict):
                errors['format_error'] = ["Se esperaba un diccionario"]
            else:
                for key, expected_type in expected_format.items():
                    if key not in data:
                        errors[key] = ["Campo requerido no encontrado"]
                    elif not isinstance(data[key], expected_type):
                        errors[key] = [
                            f"Tipo inválido. Esperado: {expected_type.__name__}"
                        ]
                        
        if errors and self.logger:
            self.logger.warning(
                "output_format_validation_errors",
                errors=errors
            )
            
        return errors 