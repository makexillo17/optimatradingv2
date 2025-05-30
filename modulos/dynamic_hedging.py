import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import norm
from .base_module import BaseAnalysisModule

class DynamicHedgingModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("dynamic_hedging")
        self.required_fields = [
            'option_data',
            'underlying_data',
            'market_data',
            'risk_params'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza y recomienda ajustes de cobertura"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Calcular griegas actuales
        greeks = self._calculate_greeks(
            data['option_data'],
            data['underlying_data']
        )
        
        # Analizar exposición al riesgo
        risk_exposure = self._analyze_risk_exposure(
            greeks,
            data['risk_params']
        )
        
        # Evaluar condiciones de mercado
        market_conditions = self._evaluate_market_conditions(
            data['market_data']
        )
        
        # Calcular señales
        signals = [
            risk_exposure['signal'],
            market_conditions['signal'],
            greeks['hedge_signal']
        ]
        
        weights = [0.5, 0.3, 0.2]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            greeks,
            risk_exposure,
            market_conditions,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _calculate_greeks(
        self,
        option_data: Dict[str, Any],
        underlying_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calcula las griegas de la posición"""
        # Extraer datos
        S = underlying_data['price']
        K = option_data['strike']
        T = option_data['time_to_expiry']
        r = option_data['risk_free_rate']
        sigma = option_data['implied_volatility']
        
        # Calcular d1 y d2
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Calcular griegas principales
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
        theta = -(S*sigma*norm.pdf(d1))/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
        vega = S*np.sqrt(T)*norm.pdf(d1)
        
        # Generar señal de cobertura
        hedge_signal = self._generate_hedge_signal(delta, gamma, theta, vega)
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'hedge_signal': hedge_signal
        }
    
    def _analyze_risk_exposure(
        self,
        greeks: Dict[str, float],
        risk_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la exposición al riesgo de la posición"""
        # Calcular exposiciones normalizadas
        delta_exposure = greeks['delta'] / risk_params['delta_limit']
        gamma_exposure = greeks['gamma'] / risk_params['gamma_limit']
        theta_exposure = greeks['theta'] / risk_params['theta_limit']
        vega_exposure = greeks['vega'] / risk_params['vega_limit']
        
        # Calcular score de riesgo total
        risk_score = np.mean([
            abs(delta_exposure),
            abs(gamma_exposure),
            abs(theta_exposure),
            abs(vega_exposure)
        ])
        
        # Generar señal
        signal = -np.clip(risk_score - 0.5, -1, 1)  # Negativo porque mayor riesgo = señal negativa
        
        return {
            'signal': signal,
            'delta_exposure': delta_exposure,
            'gamma_exposure': gamma_exposure,
            'theta_exposure': theta_exposure,
            'vega_exposure': vega_exposure,
            'risk_score': risk_score
        }
    
    def _evaluate_market_conditions(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evalúa las condiciones de mercado para el hedging"""
        # Analizar liquidez
        bid_ask = market_data['bid_ask_spread']
        normal_spread = market_data['normal_spread']
        liquidity_score = 1 - min(bid_ask / normal_spread, 1)
        
        # Analizar volatilidad
        current_vol = market_data['current_volatility']
        historical_vol = market_data['historical_volatility']
        vol_score = 1 - min(current_vol / historical_vol, 1)
        
        # Generar señal combinada
        signal = np.clip((liquidity_score + vol_score) / 2, -1, 1)
        
        return {
            'signal': signal,
            'liquidity_score': liquidity_score,
            'volatility_score': vol_score
        }
    
    def _generate_hedge_signal(
        self,
        delta: float,
        gamma: float,
        theta: float,
        vega: float
    ) -> float:
        """Genera una señal de trading basada en las griegas"""
        # Pesos para cada griega
        weights = {
            'delta': 0.4,
            'gamma': 0.3,
            'theta': 0.2,
            'vega': 0.1
        }
        
        # Normalizar y combinar señales
        signal = (
            weights['delta'] * np.clip(delta, -1, 1) +
            weights['gamma'] * np.clip(gamma * 100, -1, 1) +
            weights['theta'] * np.clip(theta / 100, -1, 1) +
            weights['vega'] * np.clip(vega / 100, -1, 1)
        )
        
        return np.clip(signal, -1, 1)
    
    def _generate_justification(
        self,
        greeks: Dict[str, Any],
        risk_exposure: Dict[str, Any],
        market_conditions: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar exposiciones principales
        if abs(risk_exposure['delta_exposure']) > 0.7:
            direction = "larga" if risk_exposure['delta_exposure'] > 0 else "corta"
            parts.append(f"Exposición delta {direction} significativa")
            
        if abs(risk_exposure['gamma_exposure']) > 0.5:
            parts.append("Alta exposición a gamma")
            
        # Analizar condiciones de mercado
        if market_conditions['liquidity_score'] < 0.3:
            parts.append("Condiciones de liquidez desfavorables")
        elif market_conditions['liquidity_score'] > 0.7:
            parts.append("Buenas condiciones de liquidez")
            
        # Analizar necesidad de hedging
        if risk_exposure['risk_score'] > 0.8:
            parts.append("Se requiere ajuste inmediato de cobertura")
        elif risk_exposure['risk_score'] > 0.5:
            parts.append("Se sugiere ajuste moderado de cobertura")
            
        if not parts:
            return "Posición adecuadamente cubierta"
            
        action = "incrementar" if recommendation == "long" else "reducir" if recommendation == "short" else "mantener"
        parts.append(f"Se recomienda {action} la cobertura")
        
        return ". ".join(parts) + "." 