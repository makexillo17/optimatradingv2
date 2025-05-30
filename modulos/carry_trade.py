import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .base_module import BaseAnalysisModule

class CarryTradeModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("carry_trade")
        self.required_fields = [
            'interest_rates',
            'forward_rates',
            'volatility',
            'funding_costs'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza oportunidades de carry trade"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        carry = self._analyze_carry(
            data['interest_rates'],
            data['forward_rates']
        )
        
        risk = self._analyze_risk(
            data['volatility'],
            data['funding_costs']
        )
        
        opportunity = self._evaluate_opportunity(carry, risk)
        
        # Calcular señales
        signals = [
            carry['signal'],
            -risk['signal'],  # Invertir señal de riesgo
            opportunity['signal']
        ]
        
        weights = [0.4, 0.3, 0.3]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.6:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(carry, risk, opportunity, recommendation)
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_carry(
        self,
        interest_rates: Dict[str, float],
        forward_rates: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analiza el diferencial de tasas y carry potencial"""
        # Calcular diferencial de tasas
        rate_differential = interest_rates['target'] - interest_rates['base']
        
        # Calcular forward discount/premium
        forward_points = forward_rates['outright'] - forward_rates['spot']
        
        # Calcular carry anualizado
        days_to_maturity = forward_rates.get('days_to_maturity', 360)
        annualized_carry = (rate_differential - (forward_points / forward_rates['spot'])) * (360 / days_to_maturity)
        
        # Generar señal normalizada
        signal = np.clip(annualized_carry / 0.05, -1, 1)  # Normalizar con 5% como máximo
        
        return {
            'signal': signal,
            'rate_differential': rate_differential,
            'forward_points': forward_points,
            'annualized_carry': annualized_carry
        }
    
    def _analyze_risk(
        self,
        volatility: Dict[str, float],
        funding_costs: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analiza los riesgos asociados"""
        # Analizar volatilidad
        vol_signal = (volatility['current'] - volatility['historical_mean']) / volatility['historical_std']
        
        # Analizar costos de financiamiento
        funding_spread = funding_costs['target'] - funding_costs['base']
        funding_signal = funding_spread / funding_costs.get('threshold', 0.02)
        
        # Combinar señales de riesgo
        risk_signal = np.clip((vol_signal + funding_signal) / 2, -1, 1)
        
        return {
            'signal': risk_signal,
            'volatility_z_score': vol_signal,
            'funding_spread': funding_spread
        }
    
    def _evaluate_opportunity(
        self,
        carry: Dict[str, Any],
        risk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evalúa la calidad de la oportunidad"""
        # Calcular ratio de Sharpe modificado
        carry_return = carry['annualized_carry']
        risk_factor = max(abs(risk['volatility_z_score']), 1)
        sharpe = carry_return / risk_factor
        
        # Evaluar costos de implementación
        implementation_score = 1 - abs(risk['funding_spread'])
        
        # Generar señal combinada
        signal = np.clip(sharpe * implementation_score, -1, 1)
        
        return {
            'signal': signal,
            'sharpe_ratio': sharpe,
            'implementation_score': implementation_score
        }
    
    def _generate_justification(
        self,
        carry: Dict[str, Any],
        risk: Dict[str, Any],
        opportunity: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar carry
        carry_desc = "positivo" if carry['annualized_carry'] > 0 else "negativo"
        parts.append(f"Carry anualizado {carry_desc} de {carry['annualized_carry']:.2%}")
        
        # Analizar riesgos
        if abs(risk['volatility_z_score']) > 1:
            vol_desc = "alta" if risk['volatility_z_score'] > 0 else "baja"
            parts.append(f"Volatilidad {vol_desc} respecto a la media histórica")
            
        # Analizar oportunidad
        if abs(opportunity['sharpe_ratio']) > 0.5:
            quality = "favorable" if opportunity['sharpe_ratio'] > 0 else "desfavorable"
            parts.append(f"Ratio de Sharpe {quality} de {opportunity['sharpe_ratio']:.2f}")
            
        if not parts:
            return "No hay señales claras para carry trade"
            
        sentiment = "favorable" if recommendation == "long" else "desfavorable" if recommendation == "short" else "neutral"
        parts.append(f"Condiciones generales {sentiment} para carry trade")
        
        return ". ".join(parts) + "." 