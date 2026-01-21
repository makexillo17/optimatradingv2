import numpy as np
import pandas as pd

from typing import Dict, Any
from .base_module import BaseAnalysisModule

class BrokerBehaviorModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("broker_behavior")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el comportamiento de los brokers basado en volumen institucional"""
        try:
            # Obtener el DataFrame de market_data
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 20:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            from ta.trend import SMAIndicator

            # Calcular promedio de volumen de 20 periodos
            volume_ma = SMAIndicator(close=market_data['volume'], window=20).sma_indicator().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            
            # Verificar si hay actividad institucional (volumen > 2.5x promedio)
            if current_volume > (volume_ma * 2.5):
                # Determinar dirección basada en si la vela es verde o roja
                current_close = market_data['close'].iloc[-1]
                current_open = market_data['open'].iloc[-1]
                
                if current_close > current_open:
                    # Vela verde (alcista)
                    recommendation = "long"
                    confidence = 0.75
                    justification = f"Actividad institucional alcista detectada. Volumen actual ({current_volume:.0f}) es {current_volume/volume_ma:.2f}x el promedio"
                else:
                    # Vela roja (bajista)
                    recommendation = "short"
                    confidence = 0.75
                    justification = f"Actividad institucional bajista detectada. Volumen actual ({current_volume:.0f}) es {current_volume/volume_ma:.2f}x el promedio"
            else:
                recommendation = "neutral"
                confidence = 0.3
                justification = f"Volumen normal. Sin actividad institucional significativa ({current_volume/volume_ma:.2f}x promedio)"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
