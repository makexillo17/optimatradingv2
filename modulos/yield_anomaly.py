import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy import stats
from .base_module import BaseAnalysisModule

class YieldAnomalyModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("yield_anomaly")
        self.required_fields = [
            'yield_data',
            'credit_metrics',
            'market_data',
            'risk_metrics',
            'liquidity_data'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza anomalías en rendimientos"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        yield_analysis = self._analyze_yield_relationships(
            data['yield_data'],
            data['credit_metrics']
        )
        
        market_analysis = self._analyze_market_conditions(
            data['market_data'],
            data['liquidity_data']
        )
        
        risk_analysis = self._analyze_risk_metrics(
            data['risk_metrics']
        )
        
        # Calcular señales
        signals = [
            yield_analysis['signal'],
            market_analysis['signal'],
            -risk_analysis['signal']  # Invertir señal de riesgo
        ]
        
        weights = [0.5, 0.3, 0.2]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            yield_analysis,
            market_analysis,
            risk_analysis,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_yield_relationships(
        self,
        yield_data: Dict[str, Any],
        credit_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza relaciones entre rendimientos"""
        # Analizar spreads de crédito
        current_spread = yield_data['credit_spread']
        fair_spread = credit_metrics['fair_spread']
        spread_zscore = (current_spread - fair_spread) / credit_metrics['spread_std']
        
        # Analizar curva de rendimientos
        curve_steepness = yield_data['curve_steepness']
        normal_steepness = yield_data['normal_steepness']
        curve_zscore = (curve_steepness - normal_steepness) / yield_data['steepness_std']
        
        # Analizar carry
        carry_return = yield_data['carry_return']
        funding_cost = yield_data['funding_cost']
        carry_score = (carry_return - funding_cost) / yield_data['carry_volatility']
        
        # Generar señal combinada
        signal = np.clip(
            spread_zscore * 0.4 +  # Spread alto implica oportunidad larga
            curve_zscore * 0.3 +
            carry_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'spread_zscore': spread_zscore,
            'curve_zscore': curve_zscore,
            'carry_score': carry_score
        }
    
    def _analyze_market_conditions(
        self,
        market_data: Dict[str, Any],
        liquidity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza condiciones de mercado"""
        # Analizar liquidez
        bid_ask = liquidity_data['bid_ask_spread']
        normal_spread = liquidity_data['normal_spread']
        liquidity_score = 1 - min(bid_ask / normal_spread, 1)
        
        # Analizar volumen
        current_volume = market_data['volume']
        average_volume = market_data['average_volume']
        volume_score = min(current_volume / average_volume, 1)
        
        # Analizar momentum
        price_momentum = market_data['price_momentum']
        volatility = market_data['volatility']
        momentum_score = price_momentum / volatility
        
        # Generar señal combinada
        signal = np.clip(
            liquidity_score * 0.4 +
            volume_score * 0.3 +
            momentum_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'liquidity_score': liquidity_score,
            'volume_score': volume_score,
            'momentum_score': momentum_score
        }
    
    def _analyze_risk_metrics(
        self,
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza métricas de riesgo"""
        # Analizar riesgo de crédito
        credit_risk = risk_metrics['credit_risk']
        max_credit_risk = risk_metrics['max_credit_risk']
        credit_score = 1 - min(credit_risk / max_credit_risk, 1)
        
        # Analizar riesgo de duración
        duration_risk = risk_metrics['duration_risk']
        max_duration = risk_metrics['max_duration']
        duration_score = 1 - min(duration_risk / max_duration, 1)
        
        # Analizar riesgo de base
        basis_risk = risk_metrics['basis_risk']
        max_basis = risk_metrics['max_basis_risk']
        basis_score = 1 - min(basis_risk / max_basis, 1)
        
        # Generar señal de riesgo
        signal = np.clip(
            credit_score * 0.4 +
            duration_score * 0.3 +
            basis_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'credit_score': credit_score,
            'duration_score': duration_score,
            'basis_score': basis_score
        }
    
    def _generate_justification(
        self,
        yield_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar spreads
        if abs(yield_analysis['spread_zscore']) > 2:
            direction = "amplio" if yield_analysis['spread_zscore'] > 0 else "estrecho"
            parts.append(f"Spread de crédito {direction} respecto a fundamentales")
            
        # Analizar curva
        if abs(yield_analysis['curve_zscore']) > 2:
            curve = "empinada" if yield_analysis['curve_zscore'] > 0 else "plana"
            parts.append(f"Curva de rendimientos anormalmente {curve}")
            
        # Analizar carry
        if abs(yield_analysis['carry_score']) > 1:
            carry = "atractivo" if yield_analysis['carry_score'] > 0 else "poco atractivo"
            parts.append(f"Carry {carry} después de costos")
            
        # Analizar condiciones de mercado
        if market_analysis['liquidity_score'] < 0.5:
            parts.append("Condiciones de liquidez subóptimas")
            
        # Analizar riesgos
        if risk_analysis['credit_score'] < 0.5:
            parts.append("Alto riesgo de crédito")
        elif risk_analysis['duration_score'] < 0.5:
            parts.append("Alta sensibilidad a tasas de interés")
            
        if not parts:
            return "No se detectan anomalías significativas en rendimientos"
            
        action = "tomar" if recommendation != "neutral" else "mantener"
        direction = "larga" if recommendation == "long" else "corta" if recommendation == "short" else "neutral"
        parts.append(f"Se recomienda {action} posición {direction}")
        
        return ". ".join(parts) + "." 