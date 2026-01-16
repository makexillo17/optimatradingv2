import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class PairsTradingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("pairs_trading")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis usando Oscilador Estocástico (para un solo activo)"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 14:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            # Calcular Oscilador Estocástico
            stoch = ta.stoch(market_data['high'], market_data['low'], market_data['close'])
            
            if stoch is None or len(stoch) == 0:
                return self.format_result("neutral", 0.0, "Error calculando Estocástico")
            
            # Obtener valor actual del %K (primer componente)
            if isinstance(stoch, pd.DataFrame):
                stoch_k = stoch.iloc[:, 0] if len(stoch.columns) > 0 else None
            else:
                stoch_k = stoch
            
            if stoch_k is None or len(stoch_k) == 0:
                return self.format_result("neutral", 0.0, "Error obteniendo Estocástico %K")
            
            current_stoch = stoch_k.iloc[-1]
            
            # Lógica de trading
            if current_stoch < 20:
                recommendation = "long"
                confidence = 0.75
                justification = f"Estocástico en sobreventa ({current_stoch:.2f} < 20). Oportunidad de compra"
            elif current_stoch > 80:
                recommendation = "short"
                confidence = 0.75
                justification = f"Estocástico en sobrecompra ({current_stoch:.2f} > 80). Oportunidad de venta"
            else:
                recommendation = "neutral"
                confidence = 0.4
                justification = f"Estocástico en zona neutral ({current_stoch:.2f}). Sin señal clara"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
