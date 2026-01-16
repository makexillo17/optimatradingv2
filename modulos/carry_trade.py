import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class CarryTradeModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("carry_trade")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de tendencia fuerte usando EMA 50 y EMA 200"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 200:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (se necesitan 200)")
            
            # Calcular EMA 50 y EMA 200
            ema_50 = ta.ema(market_data['close'], length=50)
            ema_200 = ta.ema(market_data['close'], length=200)
            
            if ema_50 is None or ema_200 is None or len(ema_50) < 2 or len(ema_200) < 2:
                return self.format_result("neutral", 0.0, "Error calculando EMAs")
            
            current_ema50 = ema_50.iloc[-1]
            current_ema200 = ema_200.iloc[-1]
            previous_ema50 = ema_50.iloc[-2]
            
            # Verificar Golden Cross (EMA 50 > EMA 200)
            if current_ema50 > current_ema200:
                # Verificar pendiente positiva (EMA 50 creciente)
                if current_ema50 > previous_ema50:
                    recommendation = "long"
                    confidence = 0.8
                    justification = f"Golden Cross detectado (EMA50={current_ema50:.2f} > EMA200={current_ema200:.2f}) con pendiente positiva. Tendencia alcista fuerte"
                else:
                    recommendation = "neutral"
                    confidence = 0.5
                    justification = f"Golden Cross presente pero EMA50 sin pendiente positiva"
            elif current_ema50 < current_ema200:
                # Death Cross - tendencia bajista
                if current_ema50 < previous_ema50:
                    recommendation = "short"
                    confidence = 0.7
                    justification = f"Death Cross detectado (EMA50={current_ema50:.2f} < EMA200={current_ema200:.2f}) con pendiente negativa. Tendencia bajista"
                else:
                    recommendation = "neutral"
                    confidence = 0.4
                    justification = f"Death Cross presente pero sin pendiente negativa clara"
            else:
                recommendation = "neutral"
                confidence = 0.3
                justification = f"EMAs muy cercanas. Sin señal clara de tendencia"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
