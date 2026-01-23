import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_module import BaseAnalysisModule
from ta.trend import SMAIndicator

class BrokerBehaviorModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("broker_behavior")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis VSA (Volume Spread Analysis) y Wyckoff.
        Detecta manipulación institucional: Absorción, Clímax, Springs y No Demand.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            # Necesitamos al menos 20 velas para las medias móviles
            if len(market_data) < 20:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (Min 20)")
            
            df = market_data.copy()
            
            # --- 1. CÁLCULO DE VARIABLES MAESTRAS ---
            
            # spread = high - low
            df['spread'] = df['high'] - df['low']
            
            # SMA(20) de Volumen y Spread
            df['vol_ma'] = df['volume'].rolling(window=20).mean()
            df['spread_ma'] = df['spread'].rolling(window=20).mean()
            
            # RVOL y R_SPREAD
            # Evitamos división por cero con replace o np.where si fuera necesario, 
            # pero en market data real es raro volumen 0 sostenido o spread 0 sostenido que de MA 0.
            current = df.iloc[-1]
            
            rvol = current['volume'] / current['vol_ma'] if current['vol_ma'] > 0 else 0
            r_spread = current['spread'] / current['spread_ma'] if current['spread_ma'] > 0 else 0
            
            # CLOSE_POS (0.0 = Low, 1.0 = High)
            range_len = current['high'] - current['low']
            close_pos = (current['close'] - current['low']) / range_len if range_len > 0 else 0.5
            
            # Datos auxiliares
            is_bullish = current['close'] > current['open']
            is_bearish = current['close'] < current['open']
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Comportamiento normal. RVOL: {rvol:.2f}, R_SPREAD: {r_spread:.2f}"
            analysis_type = "normal"
            
            # --- 2. DETECTOR DE ANOMALÍAS (Lógica Institucional) ---
            
            # Escenario A: ABSORPTION (La Trampa)
            # Mucho esfuerzo (Volumen) con poco resultado (Spread pequeño)
            if rvol > 2.0 and r_spread < 1.0:
                 # Si la vela es bajista o doji, sugiere que están frenando la caída (comprando)
                 # Si es alcista con cuerpo pequeño también puede ser absorción de venta, pero el caso clásico es frenado.
                 # El prompt dice: "Si la vela es bajista o Doji -> LONG"
                 if is_bearish or (abs(current['close'] - current['open']) < (current['spread'] * 0.1)):
                     signal = "long"
                     confidence = 0.85
                     justification = "🛡️ ABSORCIÓN DETECTADA: Volumen climático sin avance en precio (Ley Esfuerzo/Resultado)."
                     analysis_type = "absorption"

            # Escenario B: CLIMACTIC ACTION (El Fin de la Tendencia)
            # Volumen extremo y Rango extremo -> Euforia/Pánico
            elif rvol > 3.0 and r_spread > 2.0:
                signal = "neutral" # Reversal warning
                confidence = 0.8
                justification = "⚠️ VOLUMEN CLIMÁTICO: Posible agotamiento de tendencia por exceso."
                analysis_type = "climactic"
                
            # Escenario C: THE SPRING (La Sacudida Wyckoff)
            # Rompe mínimo reciente pero cierra alto con volumen
            # Condición: Rompe el mínimo de 10 velas anteriores
            else:
                # Chequeamos Spring solo si no es Climactic/Absorption para no sobreescribir (o según prioridad)
                # Calculamos mínimo de las 10 velas ANTERIORES a la actual (last 10 excluding current)
                # slice: iloc[-11:-1]
                if len(df) >= 12:
                    last_10_lows = df['low'].iloc[-11:-1].min()
                    
                    if current['low'] < last_10_lows:
                        # Rompió el mínimo
                        # Cierra en tercio superior
                        if close_pos > 0.6:
                            # Volumen Alto
                            if rvol > 1.5:
                                signal = "long" # STRONG_LONG en texto, mapeamos a 'long' con confianza alta
                                confidence = 0.95
                                justification = "🚀 WYCKOFF SPRING: Trampa bajista y recuperación rápida con volumen."
                                analysis_type = "spring"

            # Escenario D: NO DEMAND (La Debilidad)
            # Vela Alcista sin volumen
            if signal == "neutral" and analysis_type == "normal": # Solo si no hemos detectado otra cosa
                if is_bullish and rvol < 0.7:
                    signal = "short" # WEAK_SHORT
                    confidence = 0.6
                    justification = "📉 NO DEMAND: Subida sin respaldo institucional."
                    analysis_type = "no_demand"

            # Formatear resultado
            result = self.format_result(signal, confidence, justification)
            result['analysis_type'] = analysis_type
            result['metrics'] = {
                'rvol': float(rvol),
                'r_spread': float(r_spread),
                'close_pos': float(close_pos)
            }
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Broker Behavior: {str(e)}")
