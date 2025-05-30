import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from .base_module import BaseAnalysisModule

class PairsTradingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("pairs_trading")
        self.required_fields = [
            'pair_data',
            'correlation_stats',
            'cointegration_stats',
            'market_conditions'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de trading de pares"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        pair_analysis = self._analyze_pair_relationship(
            data['pair_data'],
            data['correlation_stats'],
            data['cointegration_stats']
        )
        
        spread_analysis = self._analyze_spread_dynamics(data['pair_data'])
        
        market_impact = self._analyze_market_conditions(
            data['market_conditions']
        )
        
        # Calcular señales
        signals = [
            pair_analysis['signal'],
            spread_analysis['signal'],
            market_impact['signal']
        ]
        
        weights = [0.4, 0.4, 0.2]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.65:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            pair_analysis,
            spread_analysis,
            market_impact,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_pair_relationship(
        self,
        pair_data: Dict[str, Any],
        correlation_stats: Dict[str, Any],
        cointegration_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la relación entre los pares"""
        # Analizar correlación
        current_corr = correlation_stats['current_correlation']
        historical_corr = correlation_stats['historical_correlation']
        corr_stability = 1 - abs(current_corr - historical_corr)
        
        # Analizar cointegración
        coint_pvalue = cointegration_stats['p_value']
        coint_score = 1 - coint_pvalue
        
        # Analizar beta
        beta = pair_data['beta']
        beta_error = pair_data['beta_error']
        beta_stability = 1 - min(beta_error / beta, 1)
        
        # Generar señal combinada
        signal = np.clip(
            corr_stability * 0.3 + coint_score * 0.4 + beta_stability * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'correlation': current_corr,
            'cointegration_score': coint_score,
            'beta_stability': beta_stability
        }
    
    def _analyze_spread_dynamics(self, pair_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza la dinámica del spread entre los pares"""
        # Calcular z-score del spread
        spread = pair_data['spread']
        mean_spread = pair_data['historical_mean']
        std_spread = pair_data['historical_std']
        z_score = (spread - mean_spread) / std_spread
        
        # Analizar velocidad de reversión
        half_life = pair_data['half_life']
        mean_reversion_score = np.clip(10 / half_life, 0, 1)  # Normalizar con 10 días
        
        # Analizar volatilidad del spread
        spread_vol = pair_data['spread_volatility']
        normal_vol = pair_data['normal_volatility']
        vol_ratio = spread_vol / normal_vol
        vol_score = np.clip(2 - vol_ratio, 0, 1)
        
        # Generar señal
        signal = -np.clip(z_score / 2, -1, 1)  # Negativo porque z-score alto implica short
        
        # Ajustar por calidad de la señal
        quality_adjustment = (mean_reversion_score + vol_score) / 2
        signal *= quality_adjustment
        
        return {
            'signal': signal,
            'z_score': z_score,
            'mean_reversion_score': mean_reversion_score,
            'volatility_score': vol_score
        }
    
    def _analyze_market_conditions(
        self,
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza las condiciones de mercado para el par"""
        # Analizar liquidez
        liquidity_score = market_conditions['liquidity_score']
        
        # Analizar costos de transacción
        transaction_costs = market_conditions['transaction_costs']
        normal_costs = market_conditions['normal_costs']
        cost_score = 1 - min(transaction_costs / normal_costs, 1)
        
        # Analizar condiciones de mercado generales
        market_stress = market_conditions['market_stress']
        stress_score = 1 - market_stress
        
        # Generar señal combinada
        signal = np.clip(
            liquidity_score * 0.4 + cost_score * 0.3 + stress_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'liquidity_score': liquidity_score,
            'cost_score': cost_score,
            'market_stress': market_stress
        }
    
    def _generate_justification(
        self,
        pair_analysis: Dict[str, Any],
        spread_analysis: Dict[str, Any],
        market_impact: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar relación de pares
        if pair_analysis['correlation'] > 0.8:
            parts.append("Alta correlación entre los pares")
        elif pair_analysis['correlation'] < 0.5:
            parts.append("Baja correlación sugiere precaución")
            
        # Analizar spread
        if abs(spread_analysis['z_score']) > 2:
            direction = "amplio" if spread_analysis['z_score'] > 0 else "estrecho"
            parts.append(f"Spread significativamente {direction} (z-score: {spread_analysis['z_score']:.2f})")
            
        # Analizar condiciones de mercado
        if market_impact['liquidity_score'] < 0.5:
            parts.append("Condiciones de liquidez subóptimas")
        elif market_impact['market_stress'] > 0.7:
            parts.append("Alto estrés de mercado sugiere cautela")
            
        if not parts:
            return "Condiciones normales para trading de pares"
            
        action = "abrir" if recommendation != "neutral" else "mantener"
        direction = "larga" if recommendation == "long" else "corta" if recommendation == "short" else "neutral"
        parts.append(f"Se recomienda {action} posición {direction}")
        
        return ". ".join(parts) + "." 