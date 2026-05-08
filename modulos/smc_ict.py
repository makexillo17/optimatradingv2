import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .base_module import BaseAnalysisModule

class SmcIctModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("smc_ict")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SMC Institutional Grade Analysis.
        Detects: Market Structure, Order Blocks, Liquidity Sweeps, and Breakers.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            # Necesitamos historial suficiente para detectar estructura
            if len(market_data) < 50:
                 return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 50)")

            df = market_data.copy()
            df.reset_index(drop=True, inplace=True)
            
            # --- 1. IDENTIFICACIÓN DE PIVOTS (SWING POINTS) ---
            # Ventana: 2 izquierda, 2 derecha
            
            def is_pivot_high(idx):
                if idx < 2 or idx >= len(df) - 2: return False
                c = df['high'].iloc[idx]
                return (c > df['high'].iloc[idx-1] and 
                        c > df['high'].iloc[idx-2] and 
                        c > df['high'].iloc[idx+1] and 
                        c > df['high'].iloc[idx+2])

            def is_pivot_low(idx):
                if idx < 2 or idx >= len(df) - 2: return False
                c = df['low'].iloc[idx]
                return (c < df['low'].iloc[idx-1] and 
                        c < df['low'].iloc[idx-2] and 
                        c < df['low'].iloc[idx+1] and 
                        c < df['low'].iloc[idx+2])

            pivots = [] # List of tuples (index, type, price)
            for i in range(2, len(df) - 2):
                if is_pivot_high(i):
                    pivots.append((i, 'high', df['high'].iloc[i]))
                elif is_pivot_low(i):
                    pivots.append((i, 'low', df['low'].iloc[i]))
            
            # Definir Estructura de Mercado (Trend)
            structure = "Neutral"
            if len(pivots) >= 4:
                highs = [p[2] for p in pivots if p[1] == 'high']
                lows = [p[2] for p in pivots if p[1] == 'low']
                
                if len(highs) >= 2 and len(lows) >= 2:
                    if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                        structure = "Bullish Trend"
                    elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                        structure = "Bearish Trend"

            # --- 2. DETECTION DE ORDER BLOCKS & BREAKERS ---
            
            # Vamos a identificar los BOS más recientes y encontrar sus OBs
            # Simplificación: Escaneamos los últimos 2 BOS significativos
            
            zones = [] # {'type': 'bull_ob'/'bear_ob'/'bull_breaker'/'bear_breaker', 'top':, 'bottom':, 'quality':}
            
            # Iteramos pivots para encontrar quiebres
            # Buscamos un Pivot High que haya sido roto por el precio posteriormente
            
            # BUSCAR BULLISH OBs (Originan rotura de Highs)
            # 1. Encontrar Pivot High
            # 2. Encontrar si el precio cerró por encima después
            # 3. Encontrar el punto más bajo entre el Pivot y la rotura -> Ese es el Swing Low
            # 4. En ese Swing Low, la última vela bajista es el OB.
            
            # Estrategia simplificada:
            # Tomamos los últimos pivots y veremos si generan zonas activas
            
            # Analizar últimos 50 periodos
            subset = df.iloc[-50:]
            
            # Lógica ad-hoc simple para demostración institucional:
            # Detectar OBs "sin mitigar"
            # Un OB es válido si el precio no lo ha cruzado invalidándolo completamente
            
            # Función auxiliar para crear OB
            def find_ob_candle(start_idx, end_idx, direction):
                # direction 'bull': buscamos última vela bajista (red) en el fondo
                # direction 'bear': buscamos última vela alcista (green) en el tope
                slice_df = df.iloc[start_idx:end_idx+1]
                if direction == 'bull':
                    # Buscar el Low más bajo
                    min_idx = slice_df['low'].idxmin()
                    # Buscar desde min_idx hacia atrás la vela roja más cercana (o la misma si es roja)
                    # O típicamente: la vela roja ANTES del movimiento impulsivo. 
                    # Simplificación: Usamos la vela del mínimo si es roja, o la anterior.
                    return min_idx
                else:
                    max_idx = slice_df['high'].idxmax()
                    return max_idx

            # Generación de zonas basada en pivots recientes
            # Recorremos pivots
            for i in range(len(pivots) - 1):
                p_prev = pivots[i]
                
                # Check BULLISH BOS: Un Pivot High previo es roto al alza
                if p_prev[1] == 'high':
                    # Buscar si el precio rompió este High despues de p_prev[0]
                    # Buscamos cierre > p_prev[2]
                    break_candles = df[(df.index > p_prev[0]) & (df['close'] > p_prev[2])]
                    if not break_candles.empty:
                        break_idx = break_candles.index[0]
                        
                        # El movimiento se originó en el mínimo entre p_prev y break_idx
                        swing_low_range = df.iloc[p_prev[0]:break_idx]
                        swing_low_idx = swing_low_range['low'].idxmin()
                        
                        # Definir Bullish OB: La vela en swing_low_idx
                        # (Simplificación ICT: La vela create el Swing suele contener el OB)
                        ob_top = df['high'].iloc[swing_low_idx]
                        ob_bottom = df['low'].iloc[swing_low_idx]
                        
                        # Sweep Quality?
                        # Si este low rompió un pivot low anterior
                        is_sweep = False
                        prev_lows = [p for p in pivots if p[1] == 'low' and p[0] < swing_low_idx]
                        if prev_lows and df['low'].iloc[swing_low_idx] < prev_lows[-1][2]:
                            is_sweep = True
                            
                        zones.append({
                            'type': 'bull_ob',
                            'top': ob_top,
                            'bottom': ob_bottom,
                            'idx': swing_low_idx,
                            'quality': 'high' if is_sweep else 'normal'
                        })

                # Check BEARISH BOS: Un Pivot Low previo es roto a la baja
                if p_prev[1] == 'low':
                    break_candles = df[(df.index > p_prev[0]) & (df['close'] < p_prev[2])]
                    if not break_candles.empty:
                        break_idx = break_candles.index[0]
                        
                        swing_high_range = df.iloc[p_prev[0]:break_idx]
                        swing_high_idx = swing_high_range['high'].idxmax()
                        
                        ob_top = df['high'].iloc[swing_high_idx]
                        ob_bottom = df['low'].iloc[swing_high_idx]
                        
                        is_sweep = False
                        prev_highs = [p for p in pivots if p[1] == 'high' and p[0] < swing_high_idx]
                        if prev_highs and df['high'].iloc[swing_high_idx] > prev_highs[-1][2]:
                            is_sweep = True

                        zones.append({
                            'type': 'bear_ob',
                            'top': ob_top,
                            'bottom': ob_bottom,
                            'idx': swing_high_idx,
                            'quality': 'high' if is_sweep else 'normal'
                        })

            # --- 3. Lógica de BREAKERS y VALIDACIÓN ---
            
            # Analizar estado actual de las zonas (Active vs Broken)
            current_close = df['close'].iloc[-1]
            active_zones = []
            
            for z in zones:
                # Verificar si la zona fue invalidada (rota)
                # Bullish OB roto a la baja -> Bearish Breaker
                if z['type'] == 'bull_ob':
                    # Si el precio cerró por debajo del bottom DESPUES de la creación
                    # Buscamos cierres < bottom con indice > idx
                    breaks = df[(df.index > z['idx']) & (df['close'] < z['bottom'])]
                    if not breaks.empty:
                        # Se convierte en Bearish Breaker
                        z['type'] = 'bear_breaker'
                    
                    # Validar relevancia: solo nos importa si es reciente (ej: ultimas 100 velas)
                    if z['idx'] > len(df) - 100:
                        active_zones.append(z)
                        
                elif z['type'] == 'bear_ob':
                    # Si precio cerró por encima del top -> Bullish Breaker
                    breaks = df[(df.index > z['idx']) & (df['close'] > z['top'])]
                    if not breaks.empty:
                        z['type'] = 'bull_breaker'
                    
                    if z['idx'] > len(df) - 100:
                        active_zones.append(z)

            # --- 4. GENERACIÓN DE SEÑAL ---
            
            # --- VOLUME VALIDATION (Institutional Confirmation) ---
            # Calcular RVOL: Volumen relativo al promedio de 20 períodos
            has_volume = 'volume' in df.columns and df['volume'].sum() > 0
            if has_volume:
                vol_ma_20 = df['volume'].rolling(window=20).mean()
            else:
                vol_ma_20 = pd.Series([1.0] * len(df), index=df.index)
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Estructura: {structure}. Sin zonas de interacción."
            nearest_ob_type = "None"
            nearest_ob_price = 0.0
            ob_rvol = 0.0  # Diagnóstico: RVOL del OB que generó la señal
            
            # Ver si el precio actual está DENTRO de alguna zona activa
            for z in reversed(active_zones): # Prioridad a las más recientes
                
                # ── FILTRO DE VOLUMEN INSTITUCIONAL ──────────────────
                # Solo validar OBs con volumen >= 1.5x el promedio de 20 períodos
                if has_volume:
                    ob_idx = z['idx']
                    ob_volume = df['volume'].iloc[ob_idx]
                    ob_vol_ma = vol_ma_20.iloc[ob_idx]
                    if pd.notna(ob_vol_ma) and ob_vol_ma > 0:
                        zone_rvol = ob_volume / ob_vol_ma
                    else:
                        zone_rvol = 0.0
                    
                    if zone_rvol < 1.5:
                        continue  # OB sin volumen institucional → fantasma, ignorar
                else:
                    zone_rvol = 0.0  # Sin datos de volumen, no filtrar
                
                # Verificar cercanía o toque
                # Bullish Zone (OB or Breaker)
                if z['type'] in ['bull_ob', 'bull_breaker']:
                    # Precio dentro o muy cerca del rango de la zona
                    # Tolerancia superior (podemos entrar un poco antes)
                    if z['bottom'] <= current_close <= (z['top'] * 1.001):
                        signal = "long"
                        confidence = 0.85
                        quality_msg = " (SWEEP)" if z.get('quality') == 'high' else ""
                        breaker_msg = "BREAKER" if 'breaker' in z['type'] else "ORDER BLOCK"
                        nearest_ob_type = f"Bullish {breaker_msg}"
                        nearest_ob_price = z['top']
                        ob_rvol = zone_rvol
                        
                        justification = f"Precio en Zona {nearest_ob_type}{quality_msg}. RVOL: {zone_rvol:.2f}. Estructura: {structure}."
                        break # Encontramos la más relevante
                        
                # Bearish Zone
                elif z['type'] in ['bear_ob', 'bear_breaker']:
                    if (z['bottom'] * 0.999) <= current_close <= z['top']:
                        signal = "short"
                        confidence = 0.85
                        quality_msg = " (SWEEP)" if z.get('quality') == 'high' else ""
                        breaker_msg = "BREAKER" if 'breaker' in z['type'] else "ORDER BLOCK"
                        nearest_ob_type = f"Bearish {breaker_msg}"
                        nearest_ob_price = z['bottom']
                        ob_rvol = zone_rvol
                        
                        justification = f"Precio en Zona {nearest_ob_type}{quality_msg}. RVOL: {zone_rvol:.2f}. Estructura: {structure}."
                        break

            # Resultado final
            result = self.format_result(signal, confidence, justification)
            result.update({
                'structure': structure,
                'nearest_ob_type': nearest_ob_type,
                'nearest_ob_price': float(nearest_ob_price),
                'ob_rvol': float(ob_rvol),
                'pivots_count': len(pivots)
            })
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis SMC: {str(e)}")