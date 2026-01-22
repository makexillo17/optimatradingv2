import numpy as np
import pandas as pd

from typing import Dict, Any
from .base_module import BaseAnalysisModule

class VolatilityArbModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("volatility_arb")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de Squeeze Radar usando Bollinger Bandwidth"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            # Necesitamos al menos 30 velas para BB(20) y confirmar tendencias previas
            if len(market_data) < 30:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (Min 30)")
            
            from ta.volatility import BollingerBands

            # Calcular Bandas de Bollinger (20, 2)
            bb_indicator = BollingerBands(close=market_data['close'], window=20, window_dev=2)
            
            upper_band = bb_indicator.bollinger_hband()
            lower_band = bb_indicator.bollinger_lband()
            middle_band = bb_indicator.bollinger_mavg()
            
            if upper_band.iloc[-1] is None or lower_band.iloc[-1] is None or middle_band.iloc[-1] is None:
                 return self.format_result("neutral", 0.0, "Error datos insuficientes para Bandas de Bollinger")

            # Calcular Bandwidth (Ancho de Banda Normalizado)
            # Formula: (Upper - Lower) / Middle
            # Evitar división por cero
            bandwidth = (upper_band - lower_band) / middle_band.replace(0, np.nan)
            
            # Obtener el ancho de banda actual
            current_bw = bandwidth.iloc[-1]
            
            # Determinar el mínimo de los últimos 20 periodos (incluyendo actual)
            # Usamos tail(20)
            min_bw_last_20 = bandwidth.tail(20).min()
            
            # Margen de tolerancia para considerar "cerca del mínimo" (5%)
            margin = min_bw_last_20 * 0.05
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Volatilidad normal. Bandwidth: {current_bw:.4f}"
            
            # Condición de Squeeze
            if current_bw <= (min_bw_last_20 + margin):
                signal = "neutral" # Dirección desconocida
                confidence = 0.8   # Alta probabilidad de explosión
                justification = "⚠️ ALERTA DE SQUEEZE: Alta tensión detectada. Explosión de volatilidad inminente."
                
            return self.format_result(signal, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Volatility Arbritrage: {str(e)}")
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
