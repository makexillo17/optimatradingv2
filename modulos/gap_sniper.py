"""
Gap Sniper Module — Institucional con WVDI + FVG + Liquidity Sweeps

Implementa:
1. WVDI (Wick-Volume Displacement Index) para validar sweeps
2. Fair Value Gap detection con filtro de volumen institucional
3. Liquidity Sweep detection (24h lookback)
4. Take Profit matematico 1:1.5 R:R
"""

import pandas as pd
import numpy as np
from modulos.microstructure import calculate_obi, is_flow_toxic
from typing import Dict, Any, Optional
from .base_module import BaseAnalysisModule
from ta.volatility import AverageTrueRange


class GapSniperModule(BaseAnalysisModule):
    def __init__(self, volume_threshold: float = 1.5, sweep_lookback_hours: int = 24,
                 wvdi_threshold: float = 1.5):
        super().__init__("gap_sniper")
        self.volume_threshold = volume_threshold
        self.sweep_lookback_hours = sweep_lookback_hours
        self.wvdi_threshold = wvdi_threshold
        
    def _calculate_wvdi(self, candle: pd.Series, volume_series: pd.Series,
                        liquidity_level: float, direction: str = 'bear') -> float:
        """
        Calcula el Wick-Volume Displacement Index (WVDI).
        
        WVDI = (Wick_Length / Total_Range) * Vol_ZScore * Phi(price > liq_level)
        
        Args:
            candle: Serie con open, high, low, close, volume
            volume_series: Ultimos 20+ volumenes para Z-score
            liquidity_level: Nivel de liquidez (swing high/low)
            direction: 'bear' (sweep bajista) o 'bull' (sweep alcista)
        
        Returns:
            WVDI score (>1.5 = sweep valido)
        """
        h = candle['high']
        l = candle['low']
        o = candle['open']
        c = candle['close']
        v = candle['volume']
        
        total_range = h - l
        if total_range <= 0:
            return 0.0
        
        # Wick ratio
        if direction == 'bear':
            # Mecha superior: precio subio al nivel y fue rechazado
            wick_length = h - max(o, c)
            phi = 1.0 if h > liquidity_level else 0.0
        else:
            # Mecha inferior: precio bajo al nivel y fue rechazado
            wick_length = min(o, c) - l
            phi = 1.0 if l < liquidity_level else 0.0
        
        wick_ratio = wick_length / total_range
        
        # Volume Z-Score
        vol_ma = volume_series.rolling(window=20).mean().iloc[-1]
        vol_std = volume_series.rolling(window=20).std().iloc[-1]
        
        if vol_std > 0 and not np.isnan(vol_std):
            vol_zscore = (v - vol_ma) / vol_std
        else:
            vol_zscore = 0.0
        
        # WVDI
        wvdi = wick_ratio * vol_zscore * phi
        return float(wvdi)
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta FVG + Liquidity Sweeps con WVDI.
        Institucional con confirmacion de volumen y matematica de riesgo.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 25:
                return self.format_result("neutral", 0.0,
                    f"Datos insuficientes: {len(market_data)} velas (Min 25)")
            
            # --- INDICADORES ---
            indicator_atr = AverageTrueRange(
                high=market_data['high'], low=market_data['low'],
                close=market_data['close'], window=14
            )
            current_atr = indicator_atr.average_true_range().iloc[-1]
            
            volume_ma = market_data['volume'].rolling(window=20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            rvol = (current_volume / volume_ma) if volume_ma > 0 else 0.0
            
            # --- VELAS ---
            candle_c = market_data.iloc[-1]
            candle_a = market_data.iloc[-3]
            
            signal = "neutral"
            confidence = 0.0
            justification = "No se detectaron Fair Value Gaps validados."
            sweep_detected = False
            tp_price = 0.0
            sl_price = 0.0
            wvdi_score = 0.0
            
            current_close = candle_c['close']
            current_low = candle_c['low']
            current_high = candle_c['high']
            
            # --- DETECCION DE LIQUIDITY SWEEPS ---
            lookback = min(self.sweep_lookback_hours, len(market_data) - 1)
            lookback_window = market_data.iloc[-lookback:]
            
            low_24h = lookback_window['low'].min()
            high_24h = lookback_window['high'].max()
            
            # Sweep Alcista: precio cruzo por debajo del minimo 24h y cerro por encima
            bullish_sweep = (current_low < low_24h) and (current_close > low_24h)
            # Sweep Bajista: precio cruzo por encima del maximo 24h y cerro por debajo
            bearish_sweep = (current_high > high_24h) and (current_close < high_24h)
            
            # --- CALCULO DE WVDI ---
            wvdi_bull = 0.0
            wvdi_bear = 0.0
            
            if bullish_sweep:
                wvdi_bull = self._calculate_wvdi(
                    candle_c, market_data['volume'], low_24h, direction='bull'
                )
            if bearish_sweep:
                wvdi_bear = self._calculate_wvdi(
                    candle_c, market_data['volume'], high_24h, direction='bear'
                )
            
            # --- FILTRO DE VOLUMEN INSTITUCIONAL ---
            volume_validated = rvol >= self.volume_threshold
            
            # ═══════════════════════════════════════════════════════
            # DETECCION DE FVG + WVDI
            # ═══════════════════════════════════════════════════════
            
            # FVG Alcista: Low(C) > High(A)
            if candle_c['low'] > candle_a['high']:
                gap_size = candle_c['low'] - candle_a['high']
                
                if gap_size > (0.5 * current_atr):
                    if volume_validated:
                        # WVDI bonus check
                        has_wvdi = bullish_sweep and wvdi_bull > self.wvdi_threshold
                        sweep_detected = bullish_sweep
                        wvdi_score = wvdi_bull
                        
                        signal = "long"
                        if has_wvdi:
                            confidence = 0.95  # WVDI confirmado = maximo
                            poi_quality = "SWEEP"
                        elif bullish_sweep:
                            confidence = 0.90  # Sweep sin WVDI
                            poi_quality = "SWEEP"
                        else:
                            confidence = 0.85  # Solo FVG
                            poi_quality = "INSTITUTIONAL_OB"
                        
                        sl_price = candle_a['high']
                        sl_distance = current_close - sl_price
                        tp_price = current_close + (sl_distance * 1.5)
                        
                        wvdi_tag = f" WVDI={wvdi_bull:.2f}" if bullish_sweep else ""
                        sweep_tag = " + SWEEP" if bullish_sweep else ""
                        justification = (
                            f"FVG Alcista VALIDADO{sweep_tag}{wvdi_tag}. "
                            f"Gap: {gap_size:.2f}. RVOL: {rvol:.2f}x. "
                            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                        )
                    else:
                        justification = f"FVG Alcista descartado: RVOL {rvol:.2f}x < {self.volume_threshold}x"
                else:
                    justification = f"FVG Alcista insignificante ({gap_size:.2f} vs {0.5*current_atr:.2f})"
                
            # FVG Bajista: High(C) < Low(A)
            elif candle_c['high'] < candle_a['low']:
                gap_size = candle_a['low'] - candle_c['high']
                
                if gap_size > (0.5 * current_atr):
                    if volume_validated:
                        has_wvdi = bearish_sweep and wvdi_bear > self.wvdi_threshold
                        sweep_detected = bearish_sweep
                        wvdi_score = wvdi_bear
                        
                        signal = "short"
                        if has_wvdi:
                            confidence = 0.95
                            poi_quality = "SWEEP"
                        elif bearish_sweep:
                            confidence = 0.90
                            poi_quality = "SWEEP"
                        else:
                            confidence = 0.85
                            poi_quality = "INSTITUTIONAL_OB"
                        
                        sl_price = candle_a['low']
                        sl_distance = sl_price - current_close
                        tp_price = current_close - (sl_distance * 1.5)
                        
                        wvdi_tag = f" WVDI={wvdi_bear:.2f}" if bearish_sweep else ""
                        sweep_tag = " + SWEEP" if bearish_sweep else ""
                        justification = (
                            f"FVG Bajista VALIDADO{sweep_tag}{wvdi_tag}. "
                            f"Gap: {gap_size:.2f}. RVOL: {rvol:.2f}x. "
                            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                        )
                    else:
                        justification = f"FVG Bajista descartado: RVOL {rvol:.2f}x < {self.volume_threshold}x"
                else:
                    justification = f"FVG Bajista insignificante ({gap_size:.2f} vs {0.5*current_atr:.2f})"
            
            # ═══════════════════════════════════════════════════════
            # SWEEP SIN FVG — requiere WVDI > threshold
            # ═══════════════════════════════════════════════════════
            elif volume_validated:
                if bullish_sweep and wvdi_bull > self.wvdi_threshold:
                    signal = "long"
                    confidence = 0.80
                    sweep_detected = True
                    wvdi_score = wvdi_bull
                    poi_quality = "SWEEP"
                    sl_price = low_24h
                    sl_distance = current_close - sl_price
                    tp_price = current_close + (sl_distance * 1.5)
                    justification = (
                        f"LIQUIDITY SWEEP ALCISTA (WVDI={wvdi_bull:.2f}). "
                        f"Barrio min 24h ({low_24h:.2f}). RVOL: {rvol:.2f}x. "
                        f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                    )
                elif bearish_sweep and wvdi_bear > self.wvdi_threshold:
                    signal = "short"
                    confidence = 0.80
                    sweep_detected = True
                    wvdi_score = wvdi_bear
                    poi_quality = "SWEEP"
                    sl_price = high_24h
                    sl_distance = sl_price - current_close
                    tp_price = current_close - (sl_distance * 1.5)
                    justification = (
                        f"LIQUIDITY SWEEP BAJISTA (WVDI={wvdi_bear:.2f}). "
                        f"Barrio max 24h ({high_24h:.2f}). RVOL: {rvol:.2f}x. "
                        f"TP: {tp_price:.2f} | SL: {sl_price:.2f} (1:1.5)"
                    )

            result = self.format_result(signal, confidence, justification)
            result.update({
                'rvol': float(rvol),
                'wvdi_score': float(wvdi_score),
                'sweep_detected': sweep_detected,
                'take_profit': float(tp_price),
                'stop_loss': float(sl_price),
                'current_atr': float(current_atr),
                'low_24h': float(low_24h),
                'high_24h': float(high_24h),
                'poi_quality': poi_quality if 'poi_quality' in locals() else 'RETAIL_SIGNAL',
            })
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en Gap Sniper: {str(e)}")
