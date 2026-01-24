import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class YieldAnomalyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("yield_anomaly")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis de Momentum Institucional: MACD Avanzado.
        Detecta fuerza de tendencia (Pendiente) y Agotamiento (Divergencias).
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 35:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 35)")
            
            df = market_data.copy()
            
            # --- 1. CÁLCULO DE MÉTRICAS (MACD) ---
            
            from ta.trend import MACD
            
            macd_indicator = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            
            df['macd'] = macd_indicator.macd()
            df['signal'] = macd_indicator.macd_signal()
            df['hist'] = macd_indicator.macd_diff()
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Momentum Slope (Pendiente del Histograma)
            # Diferencia entre barra actual y anterior
            slope = current['hist'] - prev['hist']
            
            # --- 2. DETECCIÓN DE DIVERGENCIAS ---
            
            divergence_detected = False
            div_type = "None"
            
            # Lookback de 5 velas para comparar picos recientes
            # Compararemos current vs hace 5 periodos
            if len(df) > 6:
                past = df.iloc[-6] # Hace 5 velas (index -1 is current, -6 is 5 ago)
                
                # Bearish Divergence (Agotamiento de Compra)
                # Precio sube, Histograma baja
                if current['close'] > past['close']:
                    # Verificamos si el histograma es menor (y positivo idealmente, o simplemente menor)
                    if current['hist'] < past['hist']:
                        divergence_detected = True
                        div_type = "Bearish"
                
                # Bullish Divergence (Agotamiento de Venta)
                # Precio baja, Histograma sube
                elif current['close'] < past['close']:
                    if current['hist'] > past['hist']:
                        divergence_detected = True
                        div_type = "Bullish"

            # --- 3. GENERACIÓN DE SEÑAL ---
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Momentum neutral. Slope: {slope:.4f}"
            
            # Prioridad 1: DIVERGENCIAS (Señales de Reversión - Alta Confianza)
            if divergence_detected:
                if div_type == "Bearish":
                    signal = "short" # Reversión a la baja
                    confidence = 0.8
                    justification = "⚠️ DIVERGENCIA BAJISTA: Precio subiendo con momentum cayendo. Reversión probable."
                elif div_type == "Bullish":
                    signal = "long" # Reversión al alza
                    confidence = 0.8
                    justification = "⚠️ DIVERGENCIA ALCISTA: Precio cayendo con momentum recuperando. Reversión probable."
            
            # Prioridad 2: IMPULSO (Tendencia - Media Confianza)
            # Solo si no hay divergencia
            else:
                # Impulso Alcista
                # MACD por encima de Signal Y Pendiente Positiva (Acelerando)
                if current['macd'] > current['signal'] and slope > 0:
                     signal = "long"
                     confidence = 0.7
                     justification = "🚀 IMPULSO ALCISTA: MACD > Signal con aceleración positiva."
                     
                # Impulso Bajista
                # MACD por debajo de Signal Y Pendiente Negativa (Acelerando a la baja)
                elif current['macd'] < current['signal'] and slope < 0:
                     signal = "short"
                     confidence = 0.7
                     justification = "🔻 IMPULSO BAJISTA: MACD < Signal con aceleración negativa."
                
                # Neutral/Consolidación (Slope muy bajo o contradicción)
                elif abs(slope) < 0.0001: # Histograma plano
                     signal = "neutral"
                     confidence = 0.4
                     justification = "Momentum plano (Consolidación)."

            result = self.format_result(signal, confidence, justification)
            result['momentum_score'] = float(current['hist'])
            result['slope'] = float(slope)
            result['divergence_detected'] = divergence_detected
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Yield Anomaly: {str(e)}")
