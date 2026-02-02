import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class DynamicHedgingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("dynamic_hedging")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gestión de Riesgo Institucional: Volatility Sizing & Chandelier Exits.
        Calcula el tamaño de posición óptimo y stops dinámicos basados en ATR.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 50:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 50)")
            
            df = market_data.copy()
            
            # --- 1. CÁLCULO DE ATR (14) ---
            from ta.volatility import AverageTrueRange
            
            atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['atr'] = atr_indicator.average_true_range()
            
            # --- 2. VOLATILIDAD RELATIVA ---
            # Comparamos el ATR actual con su promedio de 20 periodos
            df['atr_ma'] = df['atr'].rolling(window=20).mean()
            
            current = df.iloc[-1]
            current_atr = current['atr']
            atr_ma = current['atr_ma'] if current['atr_ma'] > 0 else 1.0 # Evitar div/0
            
            rel_volatility = current_atr / atr_ma
            
            # --- 3. CHANDELIER EXITS (Stops Dinámicos) ---
            # Long Stop: Highest High (20) - 3 * ATR
            # Short Stop: Lowest Low (20) + 3 * ATR
            
            highest_high = df['high'].rolling(window=20).max().iloc[-1]
            lowest_low = df['low'].rolling(window=20).min().iloc[-1]
            
            long_stop = highest_high - (3.5 * current_atr)
            short_stop = lowest_low + (3.5 * current_atr)
            
            # --- 4. LÓGICA DE DIMENSIONAMIENTO (Position Sizing) ---
            
            risk_factor = 1.0
            hedging_status = "Safe"
            signal = "neutral"
            confidence = 0.5
            justification = f"Volatilidad controlada (Rel Vol: {rel_volatility:.2f}). Operativa normal."
            
            # Reglas de Volatilidad
            if rel_volatility < 1.0:
                # Mercado Calma
                risk_factor = 1.0
                hedging_status = "Safe"
                
            elif 1.0 <= rel_volatility <= 2.0:
                # Mercado Agitado (1.5 threshold mentioned in prompt, adjusting ranges conservatively)
                if rel_volatility > 1.5:
                    risk_factor = 0.5 # Reducir a la mitad
                    hedging_status = "Caution"
                    justification = f"⚠️ ALERTA DE VOLATILIDAD: Mercado agitado (Rel Vol: {rel_volatility:.2f}). Reducir riesgo al 50%."
                else:
                    risk_factor = 0.8 # Reducción leve
                    hedging_status = "Moderate"
                    
            elif rel_volatility > 2.0:
                # Pánico / Cisne Negro
                risk_factor = 0.0 # No operar / Hedging Total
                hedging_status = "Hedge_Mode"
                signal = "short" # Señal de protección (no necesariamente direccional, sino "risk off")
                confidence = 0.9
                justification = f"🚨 MODO PÁNICO: Volatilidad extrema (Rel Vol: {rel_volatility:.2f}). HEDGING TOTAL ACTIVADO / NO OPERAR."

            result = self.format_result(signal, confidence, justification)
            result.update({
                'risk_factor': float(risk_factor),
                'hedging_status': hedging_status,
                'current_atr': float(current_atr),
                'relative_volatility': float(rel_volatility),
                'suggested_stop_long': float(long_stop),
                'suggested_stop_short': float(short_stop)
            })
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Dynamic Hedging: {str(e)}")
