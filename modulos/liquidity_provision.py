import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class LiquidityProvisionModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("liquidity_provision")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de reversión a la media usando Bandas de Bollinger"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 20:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            # Calcular Bandas de Bollinger (20, 2)
            bbands = ta.bbands(market_data['close'], length=20, std=2)
            
            if bbands is None or len(bbands) == 0:
                return self.format_result("neutral", 0.0, "Error calculando Bandas de Bollinger")
            
            # Obtener valores actuales
            current_price = market_data['close'].iloc[-1]
            lower_band = bbands.iloc[-1, 0] if isinstance(bbands, pd.DataFrame) else bbands.iloc[-1]
            upper_band = bbands.iloc[-1, 2] if isinstance(bbands, pd.DataFrame) else bbands.iloc[-1]
            
            # Verificar si el precio toca las bandas
            price_tolerance = (upper_band - lower_band) * 0.02  # 2% de tolerancia
            
            if current_price <= (lower_band + price_tolerance):
                # Precio toca banda inferior - Proveedor de liquidez absorbe ventas
                recommendation = "long"
                confidence = 0.7
                justification = f"Precio en banda inferior de Bollinger ({current_price:.2f} <= {lower_band:.2f}). Reversión esperada"
            elif current_price >= (upper_band - price_tolerance):
                # Precio toca banda superior
                recommendation = "short"
                confidence = 0.7
                justification = f"Precio en banda superior de Bollinger ({current_price:.2f} >= {upper_band:.2f}). Reversión esperada"
            else:
                recommendation = "neutral"
                confidence = 0.4
                justification = f"Precio dentro de las bandas de Bollinger. Sin señal de reversión"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
