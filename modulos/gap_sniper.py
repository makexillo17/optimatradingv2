import pandas as pd
from typing import Dict, Any, Optional
from .base_module import BaseAnalysisModule
from ta.volatility import AverageTrueRange

class GapSniperModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("gap_sniper")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta Fair Value Gaps (FVG) en las últimas velas.
        Estrategia de GAP SNIPER.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            # Necesitamos al menos 25 velas para ATR(14) y SMA(20) con solidez
            if len(market_data) < 25:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 25 requiredo para ATR/Vol)")
            
            # --- CÁLCULO DE INDICADORES ---
            # 1. ATR (14)
            indicator_atr = AverageTrueRange(high=market_data['high'], low=market_data['low'], close=market_data['close'], window=14)
            # El ATR actual es el último valor calculado
            current_atr = indicator_atr.average_true_range().iloc[-1]
            
            # 2. Volumen Promedio (20)
            # Calculamos la media móvil simple del volumen
            volume_ma = market_data['volume'].rolling(window=20).mean().iloc[-1]
            
            # --- DEFINICIÓN DE VELAS ---
            candle_c = market_data.iloc[-1] # Actual/Ultima (Signal Candle)
            candle_b = market_data.iloc[-2] # Penúltima
            candle_a = market_data.iloc[-3] # Antepenúltima
            
            signal = "neutral"
            confidence = 0.0
            justification = "No se detectaron Fair Value Gaps validados."
            
            # --- DETECCIÓN DE GAPS ---
            
            # Detectar FVG Alcista (Bullish)
            # Condición Base: Low(C) > High(A)
            if candle_c['low'] > candle_a['high']:
                gap_size = candle_c['low'] - candle_a['high']
                
                # --- FILTROS DE CALIDAD ---
                # 1. Filtro ATR: El gap debe ser relevante (> 0.5 * ATR)
                if gap_size > (0.5 * current_atr):
                    # 2. Filtro Volumen: La vela que deja el gap (C) debe tener fuerza
                    # Usamos 0.9 como factor de tolerancia leve
                    if candle_c['volume'] > (0.9 * volume_ma):
                        signal = "long" # BUY
                        confidence = 0.9
                        justification = f"FVG Alcista detectado y VALIDADO. Gap: {gap_size:.2f} (>0.5 ATR: {0.5*current_atr:.2f}). Vol: {candle_c['volume']:.0f} (>MA: {volume_ma:.0f}). Rango: {candle_a['high']:.2f} - {candle_c['low']:.2f}"
                    else:
                        justification = f"FVG Alcista descartado por bajo volumen ({candle_c['volume']:.0f} vs MA {volume_ma:.0f})"
                else:
                    justification = f"FVG Alcista descartado por tamaño insignificante ({gap_size:.2f} vs Min {0.5*current_atr:.2f})"
                
            # Detectar FVG Bajista (Bearish)
            # Condición Base: High(C) < Low(A)
            elif candle_c['high'] < candle_a['low']:
                gap_size = candle_a['low'] - candle_c['high']
                
                # --- FILTROS DE CALIDAD ---
                # 1. Filtro ATR
                if gap_size > (0.5 * current_atr):
                     # 2. Filtro Volumen
                     if candle_c['volume'] > (0.9 * volume_ma):
                        signal = "short" # SELL
                        confidence = 0.9
                        justification = f"FVG Bajista detectado y VALIDADO. Gap: {gap_size:.2f} (>0.5 ATR: {0.5*current_atr:.2f}). Vol: {candle_c['volume']:.0f} (>MA: {volume_ma:.0f}). Rango: {candle_a['low']:.2f} - {candle_c['high']:.2f}"
                     else:
                        justification = f"FVG Bajista descartado por bajo volumen ({candle_c['volume']:.0f} vs MA {volume_ma:.0f})"
                else:
                     justification = f"FVG Bajista descartado por tamaño insignificante ({gap_size:.2f} vs Min {0.5*current_atr:.2f})"
            
            # Nota: He removido la búsqueda iterativa en velas previas para simplificar y enfocar en la calidad
            # de la señal actual, como se solicitó "Validar que el movimiento tenga fuerza".
            # Si se desea reincorporar, debería aplicarse la misma lógica de filtros.

            return self.format_result(signal, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Gap Sniper: {str(e)}")
