import numpy as np
import pandas as pd

from typing import Dict, Any
from .base_module import BaseAnalysisModule

class MarketMakingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("market_making")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de mercados laterales para Market Making"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 14:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            from ta.trend import ADXIndicator
            from ta.momentum import RSIIndicator

            # Calcular ADX (14)
            adx_indicator = ADXIndicator(high=market_data['high'], low=market_data['low'], close=market_data['close'], window=14)
            current_adx = adx_indicator.adx().iloc[-1]
            
            # Calcular RSI
            current_rsi = RSIIndicator(close=market_data['close'], window=14).rsi().iloc[-1]
            
            # Lógica: Si ADX < 20 (tendencia débil) y RSI entre 40 y 60 (neutral)
            if current_adx < 20 and 40 <= current_rsi <= 60:
                recommendation = "neutral"
                confidence = 0.8
                justification = f"Mercado lateral estable. ADX={current_adx:.2f} (<20), RSI={current_rsi:.2f} (40-60). Ideal para Market Making"
            else:
                if current_adx >= 20:
                    recommendation = "neutral"
                    confidence = 0.5
                    justification = f"Tendencia detectada (ADX={current_adx:.2f}). No es ideal para Market Making"
                else:
                    recommendation = "neutral"
                    confidence = 0.5
                    justification = f"RSI fuera del rango neutral ({current_rsi:.2f}). Condiciones no ideales para Market Making"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
