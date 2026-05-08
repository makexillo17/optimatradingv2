import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class CarryTradeModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("carry_trade")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis de Fuerza de Tendencia (Trend Strength).
        Usa ADX para medir fuerza y SuperTrend para dirección.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 50:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 50)")
            
            df = market_data.copy()
            
            # --- 1. INDICADORES ---
            
            from ta.trend import ADXIndicator, EMAIndicator
            from ta.volatility import AverageTrueRange
            
            # ADX (14)
            adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_ind.adx()
            
            # EMAs (50, 200) - Referencia Macro
            df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            df['ema200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
            
            # SUPERTREND (10, 3.0) - Cálculo Manual Iterativo
            # Necesitamos ATR(10) para SuperTrend
            atr_st = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=10).average_true_range()
            
            # Multiplicador
            multiplier = 3.0
            
            # Basic Bands
            hl2 = (df['high'] + df['low']) / 2
            df['st_upper_basic'] = hl2 + (multiplier * atr_st)
            df['st_lower_basic'] = hl2 - (multiplier * atr_st)
            
            # Final Bands initialization — dtype=float64 explícito para
            # evitar errores de np_can_hold_element al asignar escalares.
            df['st_upper'] = df['st_upper_basic'].astype(np.float64).copy()
            df['st_lower'] = df['st_lower_basic'].astype(np.float64).copy()
            df['supertrend'] = np.zeros(len(df), dtype=np.float64)
            df['st_trend'] = np.ones(len(df), dtype=np.float64)  # 1: Bull, -1: Bear
            
            # Semilla inicial: las primeras 10 filas usan los básicos
            df.iloc[:10, df.columns.get_loc('supertrend')] = df['st_upper_basic'].iloc[:10].values
            
            # Iteración para SuperTrend (necesaria por la lógica recursiva)
            # Empezamos desde el índice 10 (tamaño de ventana ATR)
            for i in range(10, len(df)):
                curr_close = float(df['close'].iloc[i])
                prev_close = float(df['close'].iloc[i-1])
                
                curr_upper_basic = float(df['st_upper_basic'].iloc[i])
                curr_lower_basic = float(df['st_lower_basic'].iloc[i])
                
                prev_upper = float(df['st_upper'].iloc[i-1])
                prev_lower = float(df['st_lower'].iloc[i-1])
                prev_trend = float(df['st_trend'].iloc[i-1])
                
                # Guardia contra NaN en las primeras filas (medias móviles)
                if np.isnan(curr_upper_basic) or np.isnan(curr_lower_basic):
                    df.iloc[i, df.columns.get_loc('st_upper')] = prev_upper
                    df.iloc[i, df.columns.get_loc('st_lower')] = prev_lower
                    df.iloc[i, df.columns.get_loc('st_trend')] = prev_trend
                    df.iloc[i, df.columns.get_loc('supertrend')] = prev_upper if prev_trend == -1 else prev_lower
                    continue
                
                # Calculate Final Upper
                if (curr_upper_basic < prev_upper) or (prev_close > prev_upper):
                    df.iloc[i, df.columns.get_loc('st_upper')] = curr_upper_basic
                else:
                    df.iloc[i, df.columns.get_loc('st_upper')] = prev_upper
                    
                # Calculate Final Lower
                if (curr_lower_basic > prev_lower) or (prev_close < prev_lower):
                    df.iloc[i, df.columns.get_loc('st_lower')] = curr_lower_basic
                else:
                    df.iloc[i, df.columns.get_loc('st_lower')] = prev_lower
                
                # Determine Trend and Value
                curr_upper = float(df['st_upper'].iloc[i])
                curr_lower = float(df['st_lower'].iloc[i])
                
                if prev_trend == 1:  # Was Bullish
                    if curr_close < curr_lower:
                        df.iloc[i, df.columns.get_loc('st_trend')] = -1.0
                        df.iloc[i, df.columns.get_loc('supertrend')] = curr_upper
                    else:
                        df.iloc[i, df.columns.get_loc('st_trend')] = 1.0
                        df.iloc[i, df.columns.get_loc('supertrend')] = curr_lower
                else:  # Was Bearish
                    if curr_close > curr_upper:
                        df.iloc[i, df.columns.get_loc('st_trend')] = 1.0
                        df.iloc[i, df.columns.get_loc('supertrend')] = curr_lower
                    else:
                        df.iloc[i, df.columns.get_loc('st_trend')] = -1.0
                        df.iloc[i, df.columns.get_loc('supertrend')] = curr_upper
                        
            # Valores actuales
            current = df.iloc[-1]
            adx = current['adx']
            st_val = current['supertrend']
            st_trend = current['st_trend'] # 1 or -1
            price = current['close']
            
            # --- 2. LÓGICA DE FUERZA DE TENDENCIA ---
            
            trend_strength = "WEAK"
            if adx < 20:
                trend_strength = "WEAK"
            elif 20 <= adx <= 50:
                trend_strength = "STRONG"
            else:
                trend_strength = "EXTREME"
                
            # --- 3. GENERACIÓN DE SEÑAL ---
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Mercado sin tendencia clara (ADX={adx:.1f})."
            
            if trend_strength == "WEAK":
                signal = "neutral"
                confidence = 0.3
                justification = "😴 MERCADO SIN TENDENCIA: ADX bajo (<20). Operativa de tendencia desactivada."
            else:
                # Strong or Extreme Trend
                if st_trend == 1 and price > st_val:
                    # Bullish
                    signal = "long"
                    confidence = 0.8
                    justification = f"🚀 TENDENCIA ALCISTA FUERTE: Precio sobre SuperTrend con ADX activo ({adx:.1f})."
                    
                elif st_trend == -1 and price < st_val:
                    # Bearish
                    signal = "short"
                    confidence = 0.8
                    justification = f"📉 HARD TREND DOWN: Precio bajo SuperTrend con ADX activo ({adx:.1f})."
                else:
                    # Catch-all (e.g. price crossed but st not flipped yet? uncommon in strict logic)
                    signal = "neutral"
                    confidence = 0.5
                    justification = f"Conflicto Tren-Precio. ADX: {adx:.1f}"

            result = self.format_result(signal, confidence, justification)
            result.update({
                'adx_value': float(adx),
                'trend_strength': trend_strength,
                'supertrend': float(st_val)
            })
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Carry Trade: {str(e)}")
