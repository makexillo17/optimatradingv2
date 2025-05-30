import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import norm
from .base_module import BaseAnalysisModule

class VolatilityArbitrageModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("volatility_arb")
        self.required_fields = [
            'options_data',
            'volatility_surface',
            'historical_vol',
            'market_conditions',
            'risk_limits'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de arbitraje de volatilidad"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        vol_analysis = self._analyze_volatility_surface(
            data['volatility_surface'],
            data['historical_vol']
        )
        
        options_analysis = self._analyze_options_mispricing(
            data['options_data']
        )
        
        risk_analysis = self._analyze_risk_exposure(
            data['market_conditions'],
            data['risk_limits']
        )
        
        # Calcular señales
        signals = [
            vol_analysis['signal'],
            options_analysis['signal'],
            -risk_analysis['signal']  # Invertir señal de riesgo
        ]
        
        weights = [0.4, 0.4, 0.2]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            vol_analysis,
            options_analysis,
            risk_analysis,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_volatility_surface(
        self,
        vol_surface: Dict[str, Any],
        historical_vol: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la superficie de volatilidad"""
        # Analizar skew
        current_skew = vol_surface['skew']
        historical_skew = historical_vol['average_skew']
        skew_zscore = (current_skew - historical_skew) / historical_vol['skew_std']
        
        # Analizar term structure
        term_structure = vol_surface['term_structure']
        historical_term = historical_vol['average_term_structure']
        term_zscore = (term_structure - historical_term) / historical_vol['term_std']
        
        # Analizar niveles absolutos
        current_vol = vol_surface['atm_vol']
        historical_mean = historical_vol['mean']
        vol_zscore = (current_vol - historical_mean) / historical_vol['std']
        
        # Generar señal combinada
        signal = np.clip(
            -skew_zscore * 0.4 +  # Negativo porque skew alto implica oportunidad corta
            -term_zscore * 0.3 +  # Negativo por la misma razón
            -vol_zscore * 0.3,    # Negativo por la misma razón
            -1, 1
        )
        
        return {
            'signal': signal,
            'skew_zscore': skew_zscore,
            'term_zscore': term_zscore,
            'vol_zscore': vol_zscore
        }
    
    def _analyze_options_mispricing(
        self,
        options_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza el mispricing de opciones"""
        # Analizar put-call parity
        parity_violations = options_data['parity_violations']
        max_violation = options_data['max_acceptable_violation']
        parity_score = 1 - min(parity_violations / max_violation, 1)
        
        # Analizar volatilidad implícita vs realizada
        implied_vol = options_data['implied_vol']
        realized_vol = options_data['realized_vol']
        vol_ratio = implied_vol / realized_vol
        
        # Analizar oportunidades de arbitraje
        arb_opportunities = options_data['arbitrage_opportunities']
        arb_score = min(arb_opportunities['score'], 1)
        
        # Generar señal combinada
        signal = np.clip(
            parity_score * 0.3 +
            (1 - vol_ratio) * 0.4 +  # Menor ratio implica oportunidad larga
            arb_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'parity_score': parity_score,
            'vol_ratio': vol_ratio,
            'arb_score': arb_score
        }
    
    def _analyze_risk_exposure(
        self,
        market_conditions: Dict[str, Any],
        risk_limits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la exposición al riesgo"""
        # Analizar exposición a griegas
        vega_exposure = market_conditions['vega_exposure'] / risk_limits['max_vega']
        gamma_exposure = market_conditions['gamma_exposure'] / risk_limits['max_gamma']
        theta_exposure = market_conditions['theta_exposure'] / risk_limits['max_theta']
        
        # Analizar condiciones de mercado
        liquidity_score = market_conditions['liquidity_score']
        spread_score = market_conditions['spread_score']
        
        # Analizar riesgo de cola
        tail_risk = market_conditions['tail_risk']
        max_tail_risk = risk_limits['max_tail_risk']
        tail_risk_score = 1 - min(tail_risk / max_tail_risk, 1)
        
        # Generar señal de riesgo
        risk_signal = np.clip(
            (1 - max(vega_exposure, gamma_exposure, theta_exposure)) * 0.4 +
            ((liquidity_score + spread_score) / 2) * 0.3 +
            tail_risk_score * 0.3,
            -1, 1
        )
        
        return {
            'signal': risk_signal,
            'vega_exposure': vega_exposure,
            'gamma_exposure': gamma_exposure,
            'liquidity_score': liquidity_score,
            'tail_risk_score': tail_risk_score
        }
    
    def _generate_justification(
        self,
        vol_analysis: Dict[str, Any],
        options_analysis: Dict[str, Any],
        risk_analysis: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar superficie de volatilidad
        if abs(vol_analysis['skew_zscore']) > 2:
            skew_type = "pronunciado" if vol_analysis['skew_zscore'] > 0 else "plano"
            parts.append(f"Skew {skew_type} respecto a niveles históricos")
            
        if abs(vol_analysis['vol_zscore']) > 2:
            vol_type = "alta" if vol_analysis['vol_zscore'] > 0 else "baja"
            parts.append(f"Volatilidad {vol_type} en términos históricos")
            
        # Analizar mispricing
        if options_analysis['parity_score'] < 0.7:
            parts.append("Detectadas violaciones de paridad put-call")
            
        if abs(options_analysis['vol_ratio'] - 1) > 0.2:
            vol_desc = "mayor" if options_analysis['vol_ratio'] > 1 else "menor"
            parts.append(f"Volatilidad implícita {vol_desc} que realizada")
            
        # Analizar riesgos
        if max(risk_analysis['vega_exposure'], risk_analysis['gamma_exposure']) > 0.8:
            parts.append("Alta exposición a riesgos de volatilidad y gamma")
            
        if risk_analysis['liquidity_score'] < 0.5:
            parts.append("Condiciones de liquidez subóptimas")
            
        if not parts:
            return "No hay señales claras para arbitraje de volatilidad"
            
        action = "implementar" if recommendation != "neutral" else "mantener"
        direction = "larga" if recommendation == "long" else "corta" if recommendation == "short" else "neutral"
        parts.append(f"Se recomienda {action} estrategia {direction} de volatilidad")
        
        return ". ".join(parts) + "." 