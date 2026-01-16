import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class DynamicHedgingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("dynamic_hedging")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de volatilidad para gestión de riesgo y hedging dinámico"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 50:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (se necesitan 50)")
            
            # Calcular ATR (14)
            atr = ta.atr(market_data['high'], market_data['low'], market_data['close'], length=14)
            
            if atr is None or len(atr) == 0:
                return self.format_result("neutral", 0.0, "Error calculando ATR")
            
            current_atr = atr.iloc[-1]
            
            # Calcular percentil 90 de ATR en las últimas 50 velas
            atr_last_50 = atr.tail(50)
            percentile_90 = atr_last_50.quantile(0.90)
            
            # Si ATR está en el 10% más alto (percentil 90+)
            if current_atr >= percentile_90:
                recommendation = "short"
                confidence = 0.75
                justification = f"Alta volatilidad detectada (ATR={current_atr:.2f} en percentil 90+). Hedging Necesario / High Volatility Alert"
            else:
                recommendation = "neutral"
                confidence = 0.5
                justification = f"Volatilidad normal (ATR={current_atr:.2f}). Sin necesidad inmediata de hedging"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
