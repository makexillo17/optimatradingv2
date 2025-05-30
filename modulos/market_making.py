import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .base_module import BaseAnalysisModule

class MarketMakingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("market_making")
        self.required_fields = [
            'orderbook',
            'trades',
            'market_stats',
            'position_data',
            'risk_metrics'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de market making"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        market_quality = self._analyze_market_quality(
            data['orderbook'],
            data['trades']
        )
        
        position_analysis = self._analyze_position(
            data['position_data'],
            data['risk_metrics']
        )
        
        market_conditions = self._analyze_market_conditions(
            data['market_stats']
        )
        
        # Calcular señales
        signals = [
            market_quality['signal'],
            position_analysis['signal'],
            market_conditions['signal']
        ]
        
        weights = [0.4, 0.3, 0.3]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.65:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            market_quality,
            position_analysis,
            market_conditions,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_market_quality(
        self,
        orderbook: Dict[str, Any],
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analiza la calidad del mercado"""
        # Analizar spread
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price
        
        # Analizar profundidad
        depth = self._calculate_market_depth(orderbook)
        
        # Analizar flujo de órdenes
        trade_flow = self._analyze_trade_flow(trades)
        
        # Calcular señal combinada
        spread_score = np.clip(spread / 0.001, 0, 1)  # Normalizar con 10bps
        depth_score = np.clip(depth / 100000, 0, 1)  # Normalizar con 100k unidades
        flow_score = trade_flow['score']
        
        signal = np.clip(
            spread_score * 0.4 + depth_score * 0.3 + flow_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'spread': spread,
            'depth': depth,
            'trade_flow': trade_flow
        }
    
    def _analyze_position(
        self,
        position_data: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la posición actual y riesgos"""
        # Analizar exposición
        net_position = position_data['net_position']
        position_limit = risk_metrics['position_limit']
        position_utilization = abs(net_position) / position_limit
        
        # Analizar P&L
        realized_pnl = position_data['realized_pnl']
        unrealized_pnl = position_data['unrealized_pnl']
        total_pnl = realized_pnl + unrealized_pnl
        
        # Analizar riesgo
        var = risk_metrics['value_at_risk']
        max_var = risk_metrics['max_var']
        risk_utilization = var / max_var
        
        # Generar señal
        pnl_score = np.clip(total_pnl / risk_metrics['daily_target'], -1, 1)
        position_score = 1 - position_utilization
        risk_score = 1 - risk_utilization
        
        signal = np.clip(
            pnl_score * 0.4 + position_score * 0.3 + risk_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'position_utilization': position_utilization,
            'pnl_performance': pnl_score,
            'risk_utilization': risk_utilization
        }
    
    def _analyze_market_conditions(
        self,
        market_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza las condiciones generales del mercado"""
        # Analizar volatilidad
        current_vol = market_stats['current_volatility']
        avg_vol = market_stats['average_volatility']
        vol_ratio = current_vol / avg_vol
        
        # Analizar volumen
        current_volume = market_stats['current_volume']
        avg_volume = market_stats['average_volume']
        volume_ratio = current_volume / avg_volume
        
        # Analizar tendencia
        trend_strength = market_stats['trend_strength']  # -1 a 1
        
        # Generar señal
        vol_score = np.clip(2 - vol_ratio, -1, 1)
        volume_score = np.clip(volume_ratio - 1, -1, 1)
        trend_score = -abs(trend_strength)  # Penalizar tendencias fuertes
        
        signal = np.clip(
            vol_score * 0.4 + volume_score * 0.4 + trend_score * 0.2,
            -1, 1
        )
        
        return {
            'signal': signal,
            'volatility_ratio': vol_ratio,
            'volume_ratio': volume_ratio,
            'trend_strength': trend_strength
        }
    
    def _calculate_market_depth(self, orderbook: Dict[str, Any]) -> float:
        """Calcula la profundidad del mercado"""
        bid_depth = sum(level[1] for level in orderbook['bids'][:5])
        ask_depth = sum(level[1] for level in orderbook['asks'][:5])
        return min(bid_depth, ask_depth)
    
    def _analyze_trade_flow(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza el flujo de operaciones"""
        if not trades:
            return {'score': 0, 'buy_ratio': 0.5, 'trade_count': 0}
            
        df = pd.DataFrame(trades)
        
        # Calcular ratio compra/venta
        buy_volume = df[df['side'] == 'buy']['volume'].sum()
        sell_volume = df[df['side'] == 'sell']['volume'].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'score': 0, 'buy_ratio': 0.5, 'trade_count': len(trades)}
            
        buy_ratio = buy_volume / total_volume
        
        # Generar score
        score = np.clip(2 * (0.5 - abs(buy_ratio - 0.5)), -1, 1)
        
        return {
            'score': score,
            'buy_ratio': buy_ratio,
            'trade_count': len(trades)
        }
    
    def _generate_justification(
        self,
        market_quality: Dict[str, Any],
        position_analysis: Dict[str, Any],
        market_conditions: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar calidad de mercado
        if market_quality['spread'] > 0.002:
            parts.append("Spreads amplios indican buena oportunidad de market making")
        elif market_quality['spread'] < 0.0005:
            parts.append("Spreads muy ajustados limitan el margen de beneficio")
            
        # Analizar posición
        if position_analysis['position_utilization'] > 0.8:
            parts.append("Alta utilización de límites de posición sugiere cautela")
        elif position_analysis['pnl_performance'] < -0.5:
            parts.append("Rendimiento por debajo del objetivo")
            
        # Analizar condiciones
        if market_conditions['volatility_ratio'] > 1.5:
            parts.append("Alta volatilidad sugiere ampliar spreads")
        elif market_conditions['volume_ratio'] < 0.7:
            parts.append("Bajo volumen indica reducir exposición")
            
        if not parts:
            return "Condiciones normales para market making"
            
        action = "incrementar" if recommendation == "long" else "reducir" if recommendation == "short" else "mantener"
        parts.append(f"Se recomienda {action} la actividad de market making")
        
        return ". ".join(parts) + "." 