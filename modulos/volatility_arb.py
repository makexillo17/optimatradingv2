import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class VolatilityArbModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("volatility_arb")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de squeezes usando Bandas de Bollinger"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 100:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (se necesitan 100)")
            
            # Calcular Bandas de Bollinger
            bbands = ta.bbands(market_data['close'], length=20, std=2)
            
            if bbands is None or len(bbands) == 0:
                return self.format_result("neutral", 0.0, "Error calculando Bandas de Bollinger")
            
            # Calcular ancho de las bandas
            if isinstance(bbands, pd.DataFrame):
                upper_band = bbands.iloc[:, 2] if len(bbands.columns) > 2 else None
                lower_band = bbands.iloc[:, 0] if len(bbands.columns) > 0 else None
            else:
                upper_band = lower_band = None
            
            if upper_band is None or lower_band is None:
                return self.format_result("neutral", 0.0, "Error obteniendo bandas de Bollinger")
            
            # Calcular ancho de banda (diferencia entre superior e inferior)
            band_width = upper_band - lower_band
            
            # Verificar si el ancho actual es el mínimo de los últimos 100 periodos
            current_width = band_width.iloc[-1]
            min_width_last_100 = band_width.tail(100).min()
            
            # Tolerancia del 1% para considerar "mínimo"
            tolerance = min_width_last_100 * 0.01
            
            if current_width <= (min_width_last_100 + tolerance):
                recommendation = "neutral"
                confidence = 0.85
                justification = f"Squeeze Detectado - Explosión inminente. Ancho de banda ({current_width:.2f}) es mínimo en 100 periodos"
            else:
                recommendation = "neutral"
                confidence = 0.3
                justification = f"Sin squeeze detectado. Ancho de banda normal ({current_width:.2f})"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
