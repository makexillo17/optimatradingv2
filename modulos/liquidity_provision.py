import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .base_module import BaseAnalysisModule

class LiquidityProvisionModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("liquidity_provision")
        self.required_fields = [
            'orderbook',
            'market_impact',
            'trading_stats',
            'risk_limits'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de provisión de liquidez"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        spread_analysis = self._analyze_spreads(data['orderbook'])
        impact_analysis = self._analyze_market_impact(data['market_impact'])
        inventory_risk = self._analyze_inventory_risk(
            data['trading_stats'],
            data['risk_limits']
        )
        
        # Calcular señales
        signals = [
            spread_analysis['signal'],
            impact_analysis['signal'],
            -inventory_risk['signal']  # Invertir señal de riesgo
        ]
        
        weights = [0.4, 0.3, 0.3]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.6:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            spread_analysis,
            impact_analysis,
            inventory_risk,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_spreads(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza los spreads y profundidad del libro"""
        # Calcular spread efectivo
        best_bid = orderbook['bids'][0][0]
        best_ask = orderbook['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price
        
        # Calcular profundidad
        depth_bid = sum(level[1] for level in orderbook['bids'][:5])
        depth_ask = sum(level[1] for level in orderbook['asks'][:5])
        
        # Analizar concentración
        bid_concentration = self._calculate_concentration(
            [level[1] for level in orderbook['bids'][:5]]
        )
        ask_concentration = self._calculate_concentration(
            [level[1] for level in orderbook['asks'][:5]]
        )
        
        # Generar señal
        spread_score = np.clip(spread / 0.001, 0, 1)  # Normalizar con 10bps como referencia
        depth_score = min(depth_bid, depth_ask) / max(depth_bid, depth_ask)
        concentration_score = 1 - (bid_concentration + ask_concentration) / 2
        
        signal = np.clip(
            (spread_score * 0.5 + depth_score * 0.3 + concentration_score * 0.2),
            -1, 1
        )
        
        return {
            'signal': signal,
            'spread': spread,
            'depth_ratio': depth_score,
            'concentration': (bid_concentration + ask_concentration) / 2
        }
    
    def _analyze_market_impact(self, impact_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el impacto de mercado y costos de transacción"""
        # Analizar impacto histórico
        recent_impact = impact_data['recent_impact']
        historical_impact = impact_data['historical_impact']
        impact_ratio = recent_impact / historical_impact
        
        # Analizar costos de transacción
        transaction_costs = impact_data['transaction_costs']
        normal_costs = impact_data['normal_costs']
        cost_ratio = transaction_costs / normal_costs
        
        # Analizar volatilidad de impacto
        impact_volatility = impact_data['impact_volatility']
        normal_volatility = impact_data['normal_volatility']
        vol_ratio = impact_volatility / normal_volatility
        
        # Generar señal combinada
        signal = np.clip(
            2 - (impact_ratio + cost_ratio + vol_ratio) / 3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'impact_ratio': impact_ratio,
            'cost_ratio': cost_ratio,
            'volatility_ratio': vol_ratio
        }
    
    def _analyze_inventory_risk(
        self,
        trading_stats: Dict[str, Any],
        risk_limits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza el riesgo de inventario"""
        # Calcular exposición actual
        current_position = trading_stats['current_position']
        position_limit = risk_limits['position_limit']
        position_utilization = abs(current_position) / position_limit
        
        # Analizar rotación de inventario
        turnover_ratio = trading_stats['turnover_ratio']
        target_turnover = risk_limits['target_turnover']
        turnover_score = turnover_ratio / target_turnover
        
        # Analizar concentración de riesgo
        risk_concentration = trading_stats['risk_concentration']
        max_concentration = risk_limits['max_concentration']
        concentration_score = risk_concentration / max_concentration
        
        # Generar señal de riesgo
        signal = np.clip(
            (position_utilization + concentration_score) / 2 - turnover_score,
            -1, 1
        )
        
        return {
            'signal': signal,
            'position_utilization': position_utilization,
            'turnover_score': turnover_score,
            'concentration_score': concentration_score
        }
    
    def _calculate_concentration(self, volumes: List[float]) -> float:
        """Calcula el índice de concentración (Herfindahl)"""
        total = sum(volumes)
        if total == 0:
            return 0
            
        shares = [v/total for v in volumes]
        return sum(s*s for s in shares)
    
    def _generate_justification(
        self,
        spread_analysis: Dict[str, Any],
        impact_analysis: Dict[str, Any],
        inventory_risk: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar spreads
        if spread_analysis['spread'] > 0.002:  # 20bps
            parts.append("Spreads amplios indican oportunidad de provisión de liquidez")
        elif spread_analysis['spread'] < 0.0005:  # 5bps
            parts.append("Spreads estrechos limitan oportunidades")
            
        # Analizar impacto
        if impact_analysis['impact_ratio'] > 1.2:
            parts.append("Alto impacto de mercado sugiere precaución")
        elif impact_analysis['impact_ratio'] < 0.8:
            parts.append("Bajo impacto de mercado favorable para provisión")
            
        # Analizar inventario
        if inventory_risk['position_utilization'] > 0.8:
            parts.append("Alta utilización de límites de posición")
        elif inventory_risk['turnover_score'] < 0.5:
            parts.append("Baja rotación de inventario")
            
        if not parts:
            return "Condiciones normales para provisión de liquidez"
            
        action = "incrementar" if recommendation == "long" else "reducir" if recommendation == "short" else "mantener"
        parts.append(f"Se recomienda {action} la provisión de liquidez")
        
        return ". ".join(parts) + "." 