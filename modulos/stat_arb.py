import numpy as np
import pandas as pd

from typing import Dict, Any
from .base_module import BaseAnalysisModule

class StatArbModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("stat_arb")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de arbitraje estadístico usando Oscilador Estocástico (para un solo activo)"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 14:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            from ta.momentum import StochasticOscillator

            # Calcular Oscilador Estocástico
            stoch_indicator = StochasticOscillator(high=market_data['high'], low=market_data['low'], close=market_data['close'])
            stoch_k = stoch_indicator.stoch()
            
            if stoch_k is None or len(stoch_k) == 0:
                return self.format_result("neutral", 0.0, "Error obteniendo Estocástico %K")
            
            current_stoch = stoch_k.iloc[-1]
            
            # Lógica de trading (igual que pairs_trading)
            if current_stoch < 20:
                recommendation = "long"
                confidence = 0.75
                justification = f"Estocástico en sobreventa ({current_stoch:.2f} < 20). Oportunidad de arbitraje estadístico"
            elif current_stoch > 80:
                recommendation = "short"
                confidence = 0.75
                justification = f"Estocástico en sobrecompra ({current_stoch:.2f} > 80). Oportunidad de arbitraje estadístico"
            else:
                recommendation = "neutral"
                confidence = 0.4
                justification = f"Estocástico en zona neutral ({current_stoch:.2f}). Sin oportunidad de arbitraje clara"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
