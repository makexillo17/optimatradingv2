import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class VolatilityArbModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("volatility_arb")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis de Volatilidad Avanzada: TTM SQUEEZE.
        Detecta compresión de volatilidad (Squeeze) y explosiones (Fired).
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 30:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 30)")
            
            df = market_data.copy()
            
            # --- 1. INDICADORES ---
            
            from ta.volatility import BollingerBands, AverageTrueRange
            from ta.trend import EMAIndicator
            
            # Bollinger Bands (20, 2.0)
            bb = BollingerBands(close=df['close'], window=20, window_dev=2.0)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            
            # Keltner Channels (20, 1.5 ATR)
            # KC Middle = EMA(20)
            ema20 = EMAIndicator(close=df['close'], window=20).ema_indicator()
            
            # ATR(20) - Note: standard is often 10 or 14, user asked for "Longitud 20"
            atr20 = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=20).average_true_range()
            
            df['kc_upper'] = ema20 + (atr20 * 1.5)
            df['kc_lower'] = ema20 - (atr20 * 1.5)
            
            # --- 2. LÓGICA DE SQUEEZE ---
            
            # Condición SQUEEZE ON: BB totalmente dentro de KC
            # UpperBB < UpperKC  AND  LowerBB > LowerKC
            df['squeeze_on'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
            
            # Condición SQUEEZE FIRED (Breakout):
            # Squeeze estaba ON hace 1 vela, y ahora está OFF
            df['squeeze_on_prev'] = df['squeeze_on'].shift(1)
            df['squeeze_fired'] = df['squeeze_on_prev'] & (~df['squeeze_on'])
            
            # Valores actuales
            current = df.iloc[-1]
            squeeze_on = bool(current['squeeze_on'])
            squeeze_fired = bool(current['squeeze_fired'])
            
            # --- 3. GENERACIÓN DE SEÑAL ---
            
            signal = "neutral"
            confidence = 0.0
            justification = "Volatilidad normal. Sin squeeze activo."
            squeeze_status = "Off"
            bandwidth = current['bb_upper'] - current['bb_lower'] # Simple width for info
            
            if squeeze_on:
                # Alerta de Compresión
                signal = "neutral"
                confidence = 0.8 # Alta confianza en que "algo va a pasar"
                justification = "⚠️ TTM SQUEEZE ACTIVO: Compresión de volatilidad extrema. Explosión inminente."
                squeeze_status = "On"
                
            elif squeeze_fired:
                # Explosión (Breakout)
                squeeze_status = "Fired"
                
                # Determinar dirección de la explosión
                # Precio rompe Upper BB?
                if current['close'] > current['bb_upper']:
                    signal = "long"
                    confidence = 0.85
                    justification = "💥 TTM SQUEEZE FIRED: Volatility Breakout Alcista (Close > UpperBB)."
                # Precio rompe Lower BB?
                elif current['close'] < current['bb_lower']:
                    signal = "short"
                    confidence = 0.85
                    justification = "💥 TTM SQUEEZE FIRED: Volatility Breakout Bajista (Close < LowerBB)."
                else:
                    # Salió del squeeze pero el precio no esta fuera de las bandas (raro pero posible si KC se expande)
                    # O dirección ambigua
                    signal = "neutral"
                    confidence = 0.5
                    justification = "💥 TTM SQUEEZE FIRED: Expansión detectada pero dirección no confirmada (Precio dentro de BB)."
            
            result = self.format_result(signal, confidence, justification)
            result['squeeze_status'] = squeeze_status
            result['bandwidth'] = float(bandwidth)
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Volatility: {str(e)}")
