"""
SMC ICT Module — Institutional Grade with OB State Machine

Implementa:
1. Market Structure Detection (BOS/CHoCH via pivot analysis)
2. Order Block State Machine: UNMITIGATED -> PARTIAL -> MITIGATED -> INVALIDATED
3. FVG Adjacency Priority (OBs con FVG adyacente tienen prioridad)
4. Institutional Volume Validation (RVOL >= 1.5x)
5. Take Profit 1:1.5 R:R matematico
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from .base_module import BaseAnalysisModule


# ── ORDER BLOCK STATES ──────────────────────────────────────────────
OB_UNMITIGATED = "UNMITIGATED"
OB_PARTIAL     = "PARTIAL_MITIGATION"
OB_MITIGATED   = "MITIGATED"
OB_INVALIDATED = "INVALIDATED"


class SmcIctModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("smc_ict")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        SMC Institutional Grade Analysis con OB State Machine.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes")
            
            if len(market_data) < 50:
                return self.format_result("neutral", 0.0,
                    f"Datos insuficientes: {len(market_data)} velas (Min 50)")

            df = market_data.copy()
            df.reset_index(drop=True, inplace=True)
            
            # ═══════════════════════════════════════════════════════
            # 1. PIVOTS (SWING POINTS)
            # ═══════════════════════════════════════════════════════
            pivots = self._detect_pivots(df)
            structure = self._determine_structure(pivots)
            
            # ═══════════════════════════════════════════════════════
            # 2. ORDER BLOCK DETECTION + STATE MACHINE
            # ═══════════════════════════════════════════════════════
            has_volume = 'volume' in df.columns and df['volume'].sum() > 0
            if has_volume:
                vol_ma_20 = df['volume'].rolling(window=20).mean()
            else:
                vol_ma_20 = pd.Series([1.0] * len(df), index=df.index)

            raw_zones = self._detect_order_blocks(df, pivots, has_volume, vol_ma_20)
            
            # Detect FVGs in data for adjacency scoring
            fvg_levels = self._detect_fvgs(df)
            
            # Apply state machine + FVG adjacency
            active_zones = self._apply_state_machine(df, raw_zones, fvg_levels)
            
            # ═══════════════════════════════════════════════════════
            # 3. SIGNAL GENERATION
            # ═══════════════════════════════════════════════════════
            current_close = df['close'].iloc[-1]
            current_price = current_close
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Estructura: {structure}. Sin zonas activas."
            nearest_ob_type = "None"
            nearest_ob_price = 0.0
            ob_rvol = 0.0
            ob_state = "N/A"
            has_fvg = False
            tp_price = 0.0
            sl_price = 0.0
            
            # Sort: Quality Score first (FVG-adjacent), then UNMITIGATED, then most recent
            active_zones.sort(key=lambda z: (
                -z.get('quality_score', 0.5),           # FVG priority (1.0 vs 0.5)
                0 if z['state'] == OB_UNMITIGATED else 1, # Fresh blocks first
                -z['idx']                                  # Most recent first
            ))
            
            for z in active_zones:
                # Skip mitigated (invalidated are already removed in state machine)
                if z['state'] in (OB_MITIGATED, OB_INVALIDATED):
                    continue
                
                # Check if price is in zone
                if z['type'] in ['bull_ob', 'bull_breaker']:
                    if z['bottom'] <= current_close <= (z['top'] * 1.001):
                        signal = "long"
                        fvg_tag = " +FVG" if z.get('has_fvg') else ""
                        state_tag = z['state']
                        quality_tag = " (SWEEP)" if z.get('quality') == 'high' else ""
                        breaker_tag = "BREAKER" if 'breaker' in z['type'] else "ORDER BLOCK"
                        
                        # Confidence based on state + FVG
                        if z.get('has_fvg') and z['state'] == OB_UNMITIGATED:
                            confidence = 0.95  # Best case
                        elif z['state'] == OB_UNMITIGATED:
                            confidence = 0.85
                        else:  # PARTIAL
                            confidence = 0.70
                        
                        nearest_ob_type = f"Bullish {breaker_tag}"
                        nearest_ob_price = z['top']
                        ob_rvol = z.get('rvol', 0.0)
                        ob_state = state_tag
                        has_fvg = z.get('has_fvg', False)
                        
                        # TP/SL matematico (1:1.5)
                        sl_price = z['bottom'] - (z['top'] - z['bottom']) * 0.1  # SL bajo el OB
                        sl_distance = current_close - sl_price
                        tp_price = current_close + (sl_distance * 1.5)
                        
                        justification = (
                            f"Precio en {nearest_ob_type}{quality_tag}{fvg_tag}. "
                            f"Estado: {state_tag}. RVOL: {zone_rvol:.2f}. "
                            f"Estructura: {structure}. "
                            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                        )
                        break
                        
                elif z['type'] in ['bear_ob', 'bear_breaker']:
                    if (z['bottom'] * 0.999) <= current_close <= z['top']:
                        signal = "short"
                        fvg_tag = " +FVG" if z.get('has_fvg') else ""
                        state_tag = z['state']
                        quality_tag = " (SWEEP)" if z.get('quality') == 'high' else ""
                        breaker_tag = "BREAKER" if 'breaker' in z['type'] else "ORDER BLOCK"
                        
                        if z.get('has_fvg') and z['state'] == OB_UNMITIGATED:
                            confidence = 0.95
                        elif z['state'] == OB_UNMITIGATED:
                            confidence = 0.85
                        else:
                            confidence = 0.70
                        
                        nearest_ob_type = f"Bearish {breaker_tag}"
                        nearest_ob_price = z['bottom']
                        ob_rvol = z.get('rvol', 0.0)
                        ob_state = state_tag
                        has_fvg = z.get('has_fvg', False)
                        
                        sl_price = z['top'] + (z['top'] - z['bottom']) * 0.1
                        sl_distance = sl_price - current_close
                        tp_price = current_close - (sl_distance * 1.5)
                        
                        justification = (
                            f"Precio en {nearest_ob_type}{quality_tag}{fvg_tag}. "
                            f"Estado: {state_tag}. RVOL: {zone_rvol:.2f}. "
                            f"Estructura: {structure}. "
                            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                        )
                        break

            result = self.format_result(signal, confidence, justification)
            result.update({
                'structure': structure,
                'nearest_ob_type': nearest_ob_type,
                'nearest_ob_price': float(nearest_ob_price),
                'ob_rvol': float(ob_rvol),
                'ob_state': ob_state,
                'has_fvg_adjacency': has_fvg,
                'poi_quality': 'INSTITUTIONAL_OB' if signal != 'neutral' else 'RETAIL_SIGNAL',
                'take_profit': float(tp_price),
                'stop_loss': float(sl_price),
                'pivots_count': len(pivots),
                'active_zones_count': len([z for z in active_zones if z['state'] != OB_INVALIDATED])
            })
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en SMC: {str(e)}")
    
    # ─── HELPERS ────────────────────────────────────────────────────
    
    def _detect_pivots(self, df: pd.DataFrame) -> List[Tuple]:
        """Detecta swing highs/lows con ventana 2-2."""
        pivots = []
        for i in range(2, len(df) - 2):
            h = df['high'].iloc[i]
            if (h > df['high'].iloc[i-1] and h > df['high'].iloc[i-2] and
                h > df['high'].iloc[i+1] and h > df['high'].iloc[i+2]):
                pivots.append((i, 'high', h))
            
            l = df['low'].iloc[i]
            if (l < df['low'].iloc[i-1] and l < df['low'].iloc[i-2] and
                l < df['low'].iloc[i+1] and l < df['low'].iloc[i+2]):
                pivots.append((i, 'low', l))
        return pivots
    
    def _determine_structure(self, pivots: List[Tuple]) -> str:
        """Determina tendencia basada en highs/lows."""
        if len(pivots) < 4:
            return "Neutral"
        highs = [p[2] for p in pivots if p[1] == 'high']
        lows = [p[2] for p in pivots if p[1] == 'low']
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
                return "Bullish Trend"
            elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
                return "Bearish Trend"
        return "Neutral"
    
    def _detect_order_blocks(self, df: pd.DataFrame, pivots: List[Tuple], has_volume: bool, vol_ma_20: pd.Series) -> List[Dict]:
        """Detecta Order Blocks raw (sin estado). Filtra por volumen institucional (Retail Noise ignorado)."""
        zones = []
        
        for i in range(len(pivots) - 1):
            p_prev = pivots[i]
            
            # Bullish BOS: pivot high roto al alza
            if p_prev[1] == 'high':
                break_candles = df[(df.index > p_prev[0]) & (df['close'] > p_prev[2])]
                if not break_candles.empty:
                    break_idx = break_candles.index[0]
                    swing_range = df.iloc[p_prev[0]:break_idx]
                    if len(swing_range) > 0:
                        swing_low_idx = swing_range['low'].idxmin()
                        
                        # --- VOLUME VALIDATION (Institutional Filter) ---
                        if has_volume:
                            ob_volume = df['volume'].iloc[swing_low_idx]
                            ob_vol_ma = vol_ma_20.iloc[swing_low_idx]
                            zone_rvol = (ob_volume / ob_vol_ma) if (pd.notna(ob_vol_ma) and ob_vol_ma > 0) else 0.0
                            if zone_rvol < 1.5:
                                continue  # Ignorar por Retail Noise
                        else:
                            zone_rvol = 0.0

                        is_sweep = False
                        prev_lows = [p for p in pivots if p[1] == 'low' and p[0] < swing_low_idx]
                        if prev_lows and df['low'].iloc[swing_low_idx] < prev_lows[-1][2]:
                            is_sweep = True
                        
                        zones.append({
                            'type': 'bull_ob',
                            'top': df['high'].iloc[swing_low_idx],
                            'bottom': df['low'].iloc[swing_low_idx],
                            'idx': swing_low_idx,
                            'quality': 'high' if is_sweep else 'normal',
                            'state': OB_UNMITIGATED,
                            'mitigation_count': 0,
                            'rvol': float(zone_rvol)
                        })
            
            # Bearish BOS: pivot low roto a la baja
            if p_prev[1] == 'low':
                break_candles = df[(df.index > p_prev[0]) & (df['close'] < p_prev[2])]
                if not break_candles.empty:
                    break_idx = break_candles.index[0]
                    swing_range = df.iloc[p_prev[0]:break_idx]
                    if len(swing_range) > 0:
                        swing_high_idx = swing_range['high'].idxmax()
                        
                        # --- VOLUME VALIDATION (Institutional Filter) ---
                        if has_volume:
                            ob_volume = df['volume'].iloc[swing_high_idx]
                            ob_vol_ma = vol_ma_20.iloc[swing_high_idx]
                            zone_rvol = (ob_volume / ob_vol_ma) if (pd.notna(ob_vol_ma) and ob_vol_ma > 0) else 0.0
                            if zone_rvol < 1.5:
                                continue  # Ignorar por Retail Noise
                        else:
                            zone_rvol = 0.0

                        is_sweep = False
                        prev_highs = [p for p in pivots if p[1] == 'high' and p[0] < swing_high_idx]
                        if prev_highs and df['high'].iloc[swing_high_idx] > prev_highs[-1][2]:
                            is_sweep = True
                        
                        zones.append({
                            'type': 'bear_ob',
                            'top': df['high'].iloc[swing_high_idx],
                            'bottom': df['low'].iloc[swing_high_idx],
                            'idx': swing_high_idx,
                            'quality': 'high' if is_sweep else 'normal',
                            'state': OB_UNMITIGATED,
                            'mitigation_count': 0,
                            'rvol': float(zone_rvol)
                        })
        
        return zones
    
    def _detect_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta Fair Value Gaps en los datos."""
        fvgs = []
        for i in range(2, len(df)):
            # Bullish FVG: low[i] > high[i-2]
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                fvgs.append({
                    'type': 'bullish',
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'idx': i
                })
            # Bearish FVG: high[i] < low[i-2]
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                fvgs.append({
                    'type': 'bearish',
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'idx': i
                })
        return fvgs
    
    def _apply_state_machine(self, df: pd.DataFrame, zones: List[Dict],
                              fvg_levels: List[Dict]) -> List[Dict]:
        """
        Aplica la maquina de estados a cada Order Block:
        UNMITIGATED -> PARTIAL -> MITIGATED -> INVALIDATED
        
        Tambien marca bloques con FVG adyacente.
        """
        active = []
        current_close = df['close'].iloc[-1]
        
        for z in zones:
            # Solo zonas recientes (ultimas 100 velas)
            if z['idx'] <= len(df) - 100:
                continue
            
            ob_top = z['top']
            ob_bottom = z['bottom']
            ob_mid = (ob_top + ob_bottom) / 2  # Mean Threshold (50%)
            
            # ── STATE MACHINE ─────────────────────────────────────
            mitigation_count = 0
            invalidated = False
            
            # Scan all candles AFTER the OB was created
            for j in range(z['idx'] + 1, len(df)):
                candle_high = df['high'].iloc[j]
                candle_low = df['low'].iloc[j]
                candle_close = df['close'].iloc[j]
                candle_body_top = max(df['open'].iloc[j], candle_close)
                candle_body_bottom = min(df['open'].iloc[j], candle_close)
                
                if z['type'] == 'bull_ob':
                    # Mecha toca el OB = mitigation event
                    if candle_low <= ob_top:
                        if candle_low <= ob_mid:
                            mitigation_count += 1
                        # INVALIDATED: body close below the OB bottom
                        if candle_body_bottom < ob_bottom:
                            invalidated = True
                            break
                
                elif z['type'] == 'bear_ob':
                    if candle_high >= ob_bottom:
                        if candle_high >= ob_mid:
                            mitigation_count += 1
                        if candle_body_top > ob_top:
                            invalidated = True
                            break
            
            # Determine state
            if invalidated:
                # Delete invalidated block from active memory
                continue
            elif mitigation_count == 0:
                z['state'] = OB_UNMITIGATED
            elif mitigation_count == 1:
                z['state'] = OB_PARTIAL
            else:
                z['state'] = OB_MITIGATED  # Touched 2+ times = used up
            
            z['mitigation_count'] = mitigation_count
            
            # ── FVG ADJACENCY CHECK ───────────────────────────────
            # Check if any FVG is within 1-3 candles of the OB
            z['has_fvg'] = False
            z['quality_score'] = 0.5
            for fvg in fvg_levels:
                distance = fvg['idx'] - z['idx']
                if 1 <= distance <= 3:
                    # Direction match
                    if z['type'] in ('bull_ob',) and fvg['type'] == 'bullish':
                        z['has_fvg'] = True
                        z['quality_score'] = 1.0
                        break
                    elif z['type'] in ('bear_ob',) and fvg['type'] == 'bearish':
                        z['has_fvg'] = True
                        z['quality_score'] = 1.0
                        break
            
            active.append(z)
        
        return active