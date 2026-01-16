import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class YieldAnomalyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("yield_anomaly")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Análisis de momentum usando MACD"""
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 26:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            # Calcular MACD
            macd = ta.macd(market_data['close'])
            
            if macd is None or len(macd) == 0:
                return self.format_result("neutral", 0.0, "Error calculando MACD")
            
            # Obtener valores actuales y anteriores
            if isinstance(macd, pd.DataFrame):
                macd_line = macd.iloc[:, 0] if len(macd.columns) > 0 else None
                signal_line = macd.iloc[:, 2] if len(macd.columns) > 2 else None
            else:
                macd_line = macd
                signal_line = None
            
            if macd_line is None or len(macd_line) < 2:
                return self.format_result("neutral", 0.0, "MACD insuficiente para detectar cruces")
            
            current_macd = macd_line.iloc[-1]
            previous_macd = macd_line.iloc[-2]
            
            if signal_line is not None and len(signal_line) >= 2:
                current_signal = signal_line.iloc[-1]
                previous_signal = signal_line.iloc[-2]
                
                # Detectar cruces
                if previous_macd <= previous_signal and current_macd > current_signal:
                    # Cruce hacia arriba (Golden Cross)
                    recommendation = "long"
                    confidence = 0.75
                    justification = f"MACD cruza hacia arriba la señal. Momentum alcista detectado"
                elif previous_macd >= previous_signal and current_macd < current_signal:
                    # Cruce hacia abajo (Death Cross)
                    recommendation = "short"
                    confidence = 0.75
                    justification = f"MACD cruza hacia abajo la señal. Momentum bajista detectado"
                else:
                    recommendation = "neutral"
                    confidence = 0.4
                    justification = f"MACD sin cruces significativos. Momentum neutral"
            else:
                # Si no hay línea de señal, usar solo MACD
                if current_macd > 0 and previous_macd <= 0:
                    recommendation = "long"
                    confidence = 0.65
                    justification = f"MACD cruza a positivo. Momentum alcista"
                elif current_macd < 0 and previous_macd >= 0:
                    recommendation = "short"
                    confidence = 0.65
                    justification = f"MACD cruza a negativo. Momentum bajista"
                else:
                    recommendation = "neutral"
                    confidence = 0.4
                    justification = f"MACD sin cambios significativos"
            
            return self.format_result(recommendation, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis: {str(e)}")
