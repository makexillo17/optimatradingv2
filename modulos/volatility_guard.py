"""
Volatility Guard Module — Filtro Macro de Volatilidad y RSI HTF

Implementa:
1. Delta-VIX proxy (Realized Volatility como sustituto del VIX para BTC)
2. RSI Diario (HTF) como interruptor de sobrecompra/sobreventa
3. Detección de régimen de volatilidad (Contango proxy vs Backwardation)

El VIX real no está disponible en datos de BTC. Se usa la volatilidad
realizada (RV) de 14 períodos como proxy: si la tasa de cambio de la RV
supera el percentil 90, el sistema bloquea entradas.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule


class VolatilityGuardModule(BaseAnalysisModule):
    def __init__(self, vix_percentile: float = 90.0, rsi_period: int = 14,
                 rsi_overbought: float = 80.0, rsi_oversold: float = 20.0,
                 vol_window: int = 14, delta_lookback: int = 5):
        super().__init__("volatility_guard")
        self.vix_percentile = vix_percentile
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.vol_window = vol_window       # Ventana para Realized Volatility
        self.delta_lookback = delta_lookback  # Periodos para Delta-VIX
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluación macro de volatilidad y momentum HTF.
        
        Returns dict con:
          - force_hold: bool → Si True, bloquear TODAS las entradas
          - block_buy: bool → RSI > overbought → no comprar
          - block_sell: bool → RSI < oversold → no vender
          - delta_vix: float → Tasa de cambio de volatilidad realizada
          - daily_rsi: float → RSI calculado sobre datos disponibles
          - vol_regime: str → 'CALM', 'ELEVATED', 'CRISIS'
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self._default_result("Datos insuficientes")
            
            if len(market_data) < 50:
                return self._default_result(f"Solo {len(market_data)} velas (min 50)")
            
            df = market_data.copy()
            
            # ═══════════════════════════════════════════════════════
            # 1. DELTA-VIX PROXY (Realized Volatility)
            # ═══════════════════════════════════════════════════════
            # Calcular Realized Volatility: stddev de log-returns * sqrt(n)
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['realized_vol'] = df['log_return'].rolling(window=self.vol_window).std() * np.sqrt(self.vol_window)
            
            current_rv = df['realized_vol'].iloc[-1]
            prev_rv = df['realized_vol'].iloc[-1 - self.delta_lookback]
            
            # Delta-VIX = tasa de cambio de la volatilidad
            if prev_rv > 0 and not np.isnan(prev_rv):
                delta_vix = (current_rv - prev_rv) / prev_rv
            else:
                delta_vix = 0.0
            
            # Percentil de la delta sobre los últimos 30 períodos
            delta_series = df['realized_vol'].pct_change(self.delta_lookback).dropna().tail(30)
            if len(delta_series) > 5:
                percentile_threshold = np.percentile(delta_series, self.vix_percentile)
                is_crisis = delta_vix > percentile_threshold
            else:
                is_crisis = False
            
            # ═══════════════════════════════════════════════════════
            # 2. CONTANGO vs BACKWARDATION PROXY
            # ═══════════════════════════════════════════════════════
            # Sin futuros VIX reales, usamos la comparación entre
            # volatilidad corto plazo (7p) vs largo plazo (21p)
            short_vol = df['log_return'].rolling(window=7).std().iloc[-1] * np.sqrt(7)
            long_vol = df['log_return'].rolling(window=21).std().iloc[-1] * np.sqrt(21)
            
            if long_vol > 0 and not np.isnan(long_vol):
                vol_term_spread = (short_vol - long_vol) / long_vol
            else:
                vol_term_spread = 0.0
            
            # Backwardation proxy: short vol > long vol (mercado estresado)
            is_backwardation = vol_term_spread > 0.15  # 15% de inversión
            
            # Determinar régimen de volatilidad
            if is_crisis or is_backwardation:
                vol_regime = "CRISIS"
            elif abs(delta_vix) > 0.3:
                vol_regime = "ELEVATED"
            else:
                vol_regime = "CALM"
            
            force_hold = (vol_regime == "CRISIS")
            
            # ═══════════════════════════════════════════════════════
            # 3. RSI HTF (Interruptor de Momentum)
            # ═══════════════════════════════════════════════════════
            # En un backtest de 1h, el "Daily RSI" se simula usando
            # periodos más largos (14 * 24 = 336 velas ~ 14 días)
            # Pero para ser práctico con datos limitados, usamos RSI(14)
            # sobre las últimas velas disponibles (funciona como proxy)
            daily_rsi = self._calculate_rsi(df['close'], period=self.rsi_period)
            
            block_buy = daily_rsi > self.rsi_overbought
            block_sell = daily_rsi < self.rsi_oversold
            
            # ═══════════════════════════════════════════════════════
            # 4. RESULTADO
            # ═══════════════════════════════════════════════════════
            justification_parts = []
            
            if force_hold:
                justification_parts.append(
                    f"CRISIS VOLATILIDAD: Delta-VIX={delta_vix:.2%}, "
                    f"Vol Regime={vol_regime}, Backwardation={is_backwardation}"
                )
            if block_buy:
                justification_parts.append(
                    f"RSI SOBRECOMPRA: {daily_rsi:.1f} > {self.rsi_overbought} (BUY bloqueado)"
                )
            if block_sell:
                justification_parts.append(
                    f"RSI SOBREVENTA: {daily_rsi:.1f} < {self.rsi_oversold} (SELL bloqueado)"
                )
            
            if not justification_parts:
                justification = f"Guard OK. RSI={daily_rsi:.1f}, Delta-VIX={delta_vix:.2%}, Regime={vol_regime}"
                recommendation = "neutral"
            else:
                justification = " | ".join(justification_parts)
                recommendation = "hold" if force_hold else "neutral"
            
            result = self.format_result(recommendation, 0.0, justification)
            result.update({
                'force_hold': force_hold,
                'block_buy': bool(block_buy),
                'block_sell': bool(block_sell),
                'delta_vix': float(delta_vix),
                'daily_rsi': float(daily_rsi),
                'vol_regime': vol_regime,
                'realized_vol': float(current_rv) if not np.isnan(current_rv) else 0.0,
                'vol_term_spread': float(vol_term_spread),
                'is_backwardation': bool(is_backwardation),
            })
            return result
            
        except Exception as e:
            return self._default_result(f"Error: {str(e)}")
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcula RSI usando método de Wilder (EMA)."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        return float(current_rsi) if not np.isnan(current_rsi) else 50.0
    
    def _default_result(self, justification: str) -> Dict[str, Any]:
        """Resultado seguro por defecto (todo permitido)."""
        result = self.format_result("neutral", 0.0, justification)
        result.update({
            'force_hold': False,
            'block_buy': False,
            'block_sell': False,
            'delta_vix': 0.0,
            'daily_rsi': 50.0,
            'vol_regime': 'UNKNOWN',
            'realized_vol': 0.0,
            'vol_term_spread': 0.0,
            'is_backwardation': False,
        })
        return result
