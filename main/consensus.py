import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from utils.logger import setup_logger

class ConsensusAnalyzer:
    def __init__(self):
        self.logger = setup_logger("ConsensusAnalyzer")
        self.module_weights = self._initialize_weights()
        self.correlation_matrix = self._initialize_correlations()
        
    def analyze(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza los resultados con Jerarquía de Decisión:
        1. Gap Sniper (Prioridad MÁXIMA)
        2. Consenso de Motores (Confirmación de Tendencia)
        3. Espera (Neutral)
        """
        try:
            # Validar resultados
            if not module_results:
                return self._generate_neutral_response("No hay resultados de módulos")

            # --- PRIORIDAD 1: EL FRANCOTIRADOR (Gap Sniper) ---
            if 'gap_sniper' in module_results:
                sniper_result = module_results['gap_sniper']
                sniper_rec = sniper_result.get('recommendation', 'neutral')
                
                if sniper_rec == 'long':
                    return {
                        'recommendation': "STRONG_BUY_GAP",
                        'confidence': sniper_result.get('confidence', 0.9),
                        'justification': "🚨 OPORTUNIDAD DE GAP ALCISTA: Detectado y validado por volumen.",
                        'details': {'source': 'gap_sniper_priority'}
                    }
                elif sniper_rec == 'short':
                    return {
                        'recommendation': "STRONG_SELL_GAP",
                        'confidence': sniper_result.get('confidence', 0.9),
                        'justification': "🚨 ALERTA DE GAP BAJISTA: Hueco de liquidez detectado. Salida recomendada.",
                        'details': {'source': 'gap_sniper_priority'}
                    }

            # --- PRIORIDAD 2: EL CONSENSO (Análisis de Motores) ---
            # Calcular señales ajustadas para el resto de motores
            adjusted_signals = self._calculate_adjusted_signals(module_results)
            dynamic_weights = self._calculate_dynamic_weights(module_results)
            
            # Calcular promedio ponderado (excluyendo o incluyendo gap_sniper? 
            # Si gap_sniper es neutral, no afecta mucho, pero mejor usamos todos para el 'Trend')
            
            total_signal = 0.0
            total_weight = 0.0
            
            for module, signal in adjusted_signals.items():
                # Podemos excluir gap_sniper del promedio de "motores" si queremos pureza,
                # pero dejarlo tampoco daña si es neutral.
                weight = dynamic_weights.get(module, 0.0)
                total_signal += signal * weight
                total_weight += weight
            
            avg_signal = total_signal / total_weight if total_weight > 0 else 0.0
            
            # Verificar confirmación de SMC (Contexto Macro)
            smc_trend = "neutral"
            if 'smc_ict' in module_results:
                smc_rec = module_results['smc_ict'].get('recommendation', 'neutral')
                smc_trend = smc_rec # long/short/neutral
            
            # -- Lógica de Tendencia --
            # Si promedio muy alto (> 0.6) y SMC confirma
            if avg_signal > 0.6 and smc_trend == 'long':
                return {
                    'recommendation': "BUY_TREND",
                    'confidence': 0.8, # Confianza alta por confirmación múltiple
                    'justification': "✅ ENTRADA TÉCNICA: Análisis de motores positivo (Tendencia + Indicadores).",
                    'details': {'avg_signal': avg_signal, 'smc_trend': smc_trend}
                }
            
            # Si promedio muy bajo (< -0.6) - (Nota: SMC bajista sería ideal confirmar, pero prompt solo dijo < -0.6)
            # Agregamos sentido común: idealmente SMC no debería ser 'long'
            elif avg_signal < -0.6: 
                 return {
                    'recommendation': "SELL_TREND",
                    'confidence': 0.8,
                    'justification': "🔻 SALIDA TÉCNICA: Debilidad estructural en múltiples motores.",
                    'details': {'avg_signal': avg_signal}
                }

            # --- PRIORIDAD 3: ESPERA ---
            return {
                'recommendation': "NEUTRAL",
                'confidence': 0.0,
                'justification': "⏳ Esperando configuración clara. Mercado sin dirección definida.",
                'details': {'avg_signal': avg_signal, 'smc_trend': smc_trend}
            }
            
        except Exception as e:
            self.logger.error(f"Error en análisis de consenso: {str(e)}")
            return self._generate_neutral_response(f"Error: {str(e)}")
            
    def _initialize_weights(self) -> Dict[str, float]:
        """Inicializa los pesos base de cada módulo"""
        return {
            'broker_behavior': 1.0,
            'carry_trade': 1.0,
            'dynamic_hedging': 1.0,
            'liquidity_provision': 1.0,
            'market_making': 1.0,
            'pairs_trading': 1.0,
            'smc_ict': 2.5,
            'stat_arb': 1.0,
            'volatility_arb': 1.0,
            'yield_anomaly': 1.0
        }
        
    def _initialize_correlations(self) -> np.ndarray:
        """Inicializa la matriz de correlaciones entre módulos"""
        modules = list(self.module_weights.keys())
        n_modules = len(modules)
        
        # Matriz base de correlaciones (se puede ajustar según análisis histórico)
        correlations = np.eye(n_modules)
        
        # Definir correlaciones conocidas
        correlations[modules.index('broker_behavior')][modules.index('market_making')] = 0.6
        correlations[modules.index('carry_trade')][modules.index('yield_anomaly')] = 0.5
        correlations[modules.index('pairs_trading')][modules.index('stat_arb')] = 0.7
        correlations[modules.index('smc_ict')][modules.index('market_making')] = 0.4
        
        # Hacer la matriz simétrica
        correlations = (correlations + correlations.T) / 2
        
        return correlations
        
    def _calculate_adjusted_signals(
        self,
        module_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcula señales ajustadas por correlación"""
        modules = list(self.module_weights.keys())
        n_modules = len(modules)
        
        # Extraer señales originales
        raw_signals = np.zeros(n_modules)
        for i, module in enumerate(modules):
            if module in module_results:
                result = module_results[module]
                recommendation = result.get('recommendation', 'neutral')
                signal = 1.0 if recommendation == 'long' else -1.0 if recommendation == 'short' else 0.0
                confidence = result.get('confidence', 0.0)
                raw_signals[i] = signal * confidence
                
        # Ajustar por correlaciones
        adjusted_signals = {}
        for i, module in enumerate(modules):
            if module in module_results:
                # Calcular ajuste por correlación
                correlations = self.correlation_matrix[i]
                other_signals = raw_signals * correlations
                
                # La señal ajustada es un promedio entre la señal original y el efecto de correlación
                adjusted_signal = (raw_signals[i] + np.mean(other_signals)) / 2
                adjusted_signals[module] = adjusted_signal
                
        return adjusted_signals
        
    def _calculate_dynamic_weights(
        self,
        module_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calcula pesos dinámicos basados en confianza y rendimiento"""
        dynamic_weights = {}
        
        for module, result in module_results.items():
            base_weight = self.module_weights.get(module, 1.0)
            confidence = result.get('confidence', 0.0)
            
            # Ajustar peso por confianza
            dynamic_weight = base_weight * confidence
            
            # Aquí se pueden agregar más factores de ajuste:
            # - Rendimiento histórico
            # - Volatilidad de señales
            # - Condiciones de mercado específicas
            
            dynamic_weights[module] = dynamic_weight
            
        # Normalizar pesos
        total_weight = sum(dynamic_weights.values())
        if total_weight > 0:
            dynamic_weights = {k: v/total_weight for k, v in dynamic_weights.items()}
            
        return dynamic_weights
        
    def _generate_weighted_consensus(
        self,
        adjusted_signals: Dict[str, float],
        dynamic_weights: Dict[str, float],
        module_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera el consenso final ponderado"""
        # Calcular señal agregada
        weighted_signal = 0.0
        total_weight = 0.0
        
        for module, signal in adjusted_signals.items():
            weight = dynamic_weights.get(module, 0.0)
            weighted_signal += signal * weight
            total_weight += weight
            
        if total_weight > 0:
            final_signal = weighted_signal / total_weight
        else:
            return self._generate_neutral_response("No se pudo calcular consenso")
            
        # Determinar recomendación
        if abs(final_signal) < 0.3:
            recommendation = "neutral"
        else:
            recommendation = "long" if final_signal > 0 else "short"
            
        # Calcular confianza
        # Basada en:
        # - Magnitud de la señal
        # - Consistencia entre módulos
        # - Número de módulos activos
        signal_confidence = min(abs(final_signal), 1.0)
        
        module_consistency = self._calculate_module_consistency(
            module_results,
            recommendation
        )
        
        coverage = len(module_results) / len(self.module_weights)
        
        final_confidence = (
            signal_confidence * 0.4 +
            module_consistency * 0.4 +
            coverage * 0.2
        )
        
        return {
            'recommendation': recommendation,
            'confidence': final_confidence,
            'signal': final_signal,
            'consistency': module_consistency,
            'coverage': coverage
        }
        
    def _calculate_module_consistency(
        self,
        module_results: Dict[str, Any],
        consensus_recommendation: str
    ) -> float:
        """Calcula la consistencia entre las recomendaciones de los módulos"""
        if not module_results:
            return 0.0
            
        consistent_count = 0
        total_count = len(module_results)
        
        for result in module_results.values():
            recommendation = result.get('recommendation', 'neutral')
            if recommendation == consensus_recommendation:
                consistent_count += 1
                
        return consistent_count / total_count
        
    def _generate_justification(
        self,
        consensus: Dict[str, Any],
        module_results: Dict[str, Any],
        adjusted_signals: Dict[str, float],
        dynamic_weights: Dict[str, float]
    ) -> str:
        """Genera una justificación detallada del consenso"""
        parts = []
        
        # Analizar consenso general
        signal = consensus.get('signal', 0.0)
        confidence = consensus.get('confidence', 0.0)
        signal_desc = "alcista" if signal > 0 else "bajista" if signal < 0 else "neutral"
        parts.append(f"Señal de consenso {signal_desc} con {confidence:.1%} de confianza")
        
        # Identificar módulos más influyentes
        weighted_signals = {
            module: abs(signal * dynamic_weights[module])
            for module, signal in adjusted_signals.items()
        }
        
        top_modules = sorted(
            weighted_signals.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        # Agregar justificaciones de módulos principales
        for module, _ in top_modules:
            result = module_results.get(module, {})
            justification = result.get('justification', '')
            if justification:
                parts.append(f"{module}: {justification}")
                
        # Analizar consistencia
        consistency = consensus.get('consistency', 0.0)
        if consistency > 0.7:
            parts.append("Alta consistencia entre módulos")
        elif consistency < 0.3:
            parts.append("Baja consistencia entre módulos sugiere cautela")
            
        # Analizar cobertura
        coverage = consensus.get('coverage', 0.0)
        if coverage < 0.5:
            parts.append("Análisis basado en conjunto limitado de módulos")
            
        return ". ".join(parts) + "."
        
    def _generate_neutral_response(self, reason: str) -> Dict[str, Any]:
        """Genera una respuesta neutral con explicación"""
        return {
            'recommendation': 'neutral',
            'confidence': 0.0,
            'justification': reason,
            'details': {
                'reason': reason
            }
        } 