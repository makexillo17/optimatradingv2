from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging

class BaseAnalysisModule(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"Module-{self.name}")
        return logger
        
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método principal de análisis que debe ser implementado por cada módulo
        
        Args:
            data: Diccionario con los datos necesarios para el análisis
            
        Returns:
            Dict con recomendación, confianza y justificación
        """
        pass
        
    def validate_data(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Valida que los datos contengan todos los campos requeridos"""
        missing = [field for field in required_fields if field not in data]
        if missing:
            self.logger.error(f"Campos faltantes: {missing}")
            return False
        return True
        
    def calculate_confidence(self, signals: List[float], weights: Optional[List[float]] = None) -> float:
        """Calcula el nivel de confianza basado en múltiples señales"""
        if weights is None:
            weights = [1.0] * len(signals)
            
        # Normalizar pesos
        weights = np.array(weights) / sum(weights)
        
        # Calcular confianza ponderada
        confidence = np.sum(np.array(signals) * weights)
        
        # Limitar entre 0 y 1
        return float(np.clip(confidence, 0, 1))
        
    def format_result(self, recommendation: str, confidence: float, justification: str) -> Dict[str, Any]:
        """Formatea el resultado en el formato estándar"""
        return {
            'timestamp': datetime.now().isoformat(),
            'module': self.name,
            'recommendation': recommendation,
            'confidence': confidence,
            'justification': justification
        } 