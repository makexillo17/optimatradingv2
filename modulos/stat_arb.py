import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from .base_module import BaseAnalysisModule

class StatisticalArbitrageModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("stat_arb")
        self.required_fields = [
            'price_series',
            'correlation_matrix',
            'risk_metrics',
            'market_conditions',
            'trading_costs'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de arbitraje estadístico"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        price_analysis = self._analyze_price_relationships(
            data['price_series'],
            data['correlation_matrix']
        )
        
        risk_analysis = self._analyze_risk_metrics(
            data['risk_metrics']
        )
        
        market_analysis = self._analyze_market_conditions(
            data['market_conditions'],
            data['trading_costs']
        )
        
        # Calcular señales
        signals = [
            price_analysis['signal'],
            -risk_analysis['signal'],  # Invertir señal de riesgo
            market_analysis['signal']
        ]
        
        weights = [0.5, 0.3, 0.2]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            price_analysis,
            risk_analysis,
            market_analysis,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_price_relationships(
        self,
        price_series: Dict[str, Any],
        correlation_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """Analiza relaciones de precios y desviaciones"""
        # Calcular desviaciones de la media
        returns = price_series['returns']
        mean_returns = price_series['mean_returns']
        std_returns = price_series['std_returns']
        
        z_scores = (returns - mean_returns) / std_returns
        
        # Analizar correlaciones
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        correlation_stability = price_series['correlation_stability']
        
        # Calcular señal de reversión
        mean_reversion = price_series['mean_reversion_score']
        
        # Generar señal combinada
        signal = np.clip(
            -np.mean(z_scores) * 0.4 +  # Negativo porque z-score alto implica sobrecompra
            correlation_stability * 0.3 +
            mean_reversion * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'z_scores': z_scores,
            'avg_correlation': avg_correlation,
            'mean_reversion': mean_reversion
        }
    
    def _analyze_risk_metrics(
        self,
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza métricas de riesgo"""
        # Analizar volatilidad
        current_vol = risk_metrics['current_volatility']
        historical_vol = risk_metrics['historical_volatility']
        vol_ratio = current_vol / historical_vol
        
        # Analizar drawdown
        max_drawdown = risk_metrics['max_drawdown']
        current_drawdown = risk_metrics['current_drawdown']
        drawdown_ratio = current_drawdown / max_drawdown if max_drawdown > 0 else 0
        
        # Analizar Sharpe ratio
        sharpe_ratio = risk_metrics['sharpe_ratio']
        min_sharpe = risk_metrics['min_acceptable_sharpe']
        sharpe_score = max(sharpe_ratio / min_sharpe, 0)
        
        # Generar señal de riesgo
        signal = np.clip(
            (1 - vol_ratio) * 0.4 +
            (1 - drawdown_ratio) * 0.3 +
            min(sharpe_score, 1) * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'volatility_ratio': vol_ratio,
            'drawdown_ratio': drawdown_ratio,
            'sharpe_score': sharpe_score
        }
    
    def _analyze_market_conditions(
        self,
        market_conditions: Dict[str, Any],
        trading_costs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza condiciones de mercado y costos"""
        # Analizar liquidez
        liquidity_score = market_conditions['liquidity_score']
        
        # Analizar spreads
        current_spread = trading_costs['current_spread']
        normal_spread = trading_costs['normal_spread']
        spread_ratio = current_spread / normal_spread
        
        # Analizar costos de impacto
        impact_cost = trading_costs['market_impact']
        max_impact = trading_costs['max_acceptable_impact']
        impact_ratio = impact_cost / max_impact
        
        # Generar señal
        signal = np.clip(
            liquidity_score * 0.4 +
            (1 - spread_ratio) * 0.3 +
            (1 - impact_ratio) * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'liquidity_score': liquidity_score,
            'spread_ratio': spread_ratio,
            'impact_ratio': impact_ratio
        }
    
    def _generate_justification(
        self,
        price_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar desviaciones de precio
        avg_zscore = np.mean(price_analysis['z_scores'])
        if abs(avg_zscore) > 2:
            direction = "sobrevaloración" if avg_zscore > 0 else "infravaloración"
            parts.append(f"Significativa {direction} estadística (z-score: {avg_zscore:.2f})")
            
        # Analizar correlaciones
        if price_analysis['avg_correlation'] > 0.7:
            parts.append("Alta correlación entre instrumentos")
        elif price_analysis['mean_reversion'] > 0.7:
            parts.append("Fuerte señal de reversión a la media")
            
        # Analizar riesgos
        if risk_analysis['volatility_ratio'] > 1.5:
            parts.append("Volatilidad por encima de niveles históricos")
        elif risk_analysis['sharpe_score'] < 0.5:
            parts.append("Ratio de Sharpe por debajo del mínimo aceptable")
            
        # Analizar condiciones de mercado
        if market_analysis['spread_ratio'] > 1.5:
            parts.append("Spreads elevados aumentan costos de ejecución")
        elif market_analysis['liquidity_score'] < 0.3:
            parts.append("Baja liquidez en el mercado")
            
        if not parts:
            return "Condiciones normales para arbitraje estadístico"
            
        action = "abrir" if recommendation != "neutral" else "mantener"
        direction = "larga" if recommendation == "long" else "corta" if recommendation == "short" else "neutral"
        parts.append(f"Se recomienda {action} posición {direction}")
        
        return ". ".join(parts) + "." 