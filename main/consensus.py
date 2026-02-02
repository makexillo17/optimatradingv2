import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from utils.logger import setup_logger

class ConsensusAnalyzer:
    def __init__(self):
        self.logger = setup_logger("ConsensusAnalyzer")
        self.module_weights = self._initialize_weights()
        self.correlation_matrix = self._initialize_correlations()
        
    def analyze(self, module_results: Dict[str, Any], market_regime: str = None) -> Dict[str, Any]:
        """
        Analiza los resultados con Jerarquía de Decisión:
        1. Gap Sniper (Prioridad MÁXIMA)
        2. Consenso de Motores (Confirmación de Tendencia)
        3. Espera (Neutral)
        4. Filtro de Régimen de Mercado (Noise/Trending/Ranging)
        """
        try:
            # Validar resultados
            if not module_results:
                return self._generate_neutral_response("No hay resultados de módulos")
                
            # --- FILTRO DE NOISE (Pánico) ---
            if market_regime == 'NOISE':
                return self._generate_neutral_response("Régimen de Mercado: NOISE (Alto Riesgo/Chop). Operativa detenida.")

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

            # --- PRIORIDAD 2: EL CONSENSO (Convicción Neta) ---
            
            long_power = 0.0
            short_power = 0.0
            active_count = 0
            
            # Usar dynamic weights para referencia si es necesario, pero la lógica solicitada es Net Power
            # Calculamos sobre todos los resultados disponibles
            
            for module, result in module_results.items():
                if module == 'gap_sniper': continue # Gap Sniper es prioridad 1, ya evaluado o se excluye del promedio general
                
                confidence = result.get('confidence', 0.0)
                rec = result.get('recommendation', 'neutral')
                
                if rec == 'long':
                    long_power += confidence
                    active_count += 1
                elif rec == 'short':
                    short_power += confidence
                    active_count += 1
                # Neutral no suma a power ni a count (para no diluir la convicción direccional pura)
                
            if active_count > 0:
                net_power = (long_power - short_power) / active_count
            else:
                net_power = 0.0
                
            final_confidence = abs(net_power)
            signal_numeric = net_power
            
            # Umbral 0.15
            bias_text = "Neutral"
            if final_confidence < 0.15:
                recommendation = "neutral"
                justification_text = "⚖️ Mercado en equilibrio. Convicción neta insuficiente (< 15%)."
            else:
                if net_power > 0:
                    recommendation = "long"
                    bias_text = "Alcista"
                    justification_text = f"🐂 Convicción Neta Alcista ({final_confidence:.1%}). Sesgo comprador dominante."
                else:
                    recommendation = "short"
                    bias_text = "Bajista"
                    justification_text = f"🐻 Convicción Neta Bajista ({final_confidence:.1%}). Sesgo vendedor dominante."
            
            # Verificar confirmación de SMC para reforzar o debilitar (opcional, pero buena práctica mantener contexto)
            smc_trend = "neutral"
            if 'smc_ict' in module_results:
                smc_rec = module_results['smc_ict'].get('recommendation', 'neutral')
                if smc_rec == recommendation and final_confidence > 0.15:
                    justification_text += " Confirmado por Estructura (SMC)."
            
            return {
                'recommendation': recommendation,
                'confidence': final_confidence,
                'signal': signal_numeric,
                'justification': justification_text,
                'details': {
                    'net_power': net_power,
                    'active_modules': active_count,
                    'long_power': long_power,
                    'short_power': short_power
                }
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
        """Calcula señales ajustadas por correlación, considerando solo módulos activos"""
        modules = list(self.module_weights.keys())
        n_modules = len(modules)
        
        # Identificar qué módulos están activos
        active_indices = [i for i, m in enumerate(modules) if m in module_results]
        
        if not active_indices:
            return {}

        # Extraer señales originales
        raw_signals = np.zeros(n_modules)
        for i in active_indices:
            module = modules[i]
            result = module_results[module]
            recommendation = result.get('recommendation', 'neutral')
            signal = 1.0 if recommendation == 'long' else -1.0 if recommendation == 'short' else 0.0
            confidence = result.get('confidence', 0.0)
            raw_signals[i] = signal * confidence
                
        # Ajustar por correlaciones (Sólo entre activos)
        adjusted_signals = {}
        for i in active_indices:
            module = modules[i]
            
            # Calcular influencia de otros módulos ACTIVOS
            correlations = self.correlation_matrix[i]
            
            # Filtramos 'other_signals' para usar solo índices activos
            # (raw_signals ya es 0 en inactivos, pero para el promedio/mean
            # es importante dividir por el número correcto de contribuyentes, no N total)
            
            influenced_signal_sum = 0.0
            influence_count = 0
            
            for j in active_indices:
                if i == j: continue # No auto-correlación en el promedio externo
                influenced_signal_sum += raw_signals[j] * correlations[j]
                influence_count += 1
            
            if influence_count > 0:
                mean_influence = influenced_signal_sum / influence_count
                # La señal ajustada es promedio entre propia y la influencia externa
                # Si la influencia externa es baja (porque los otros son neutrales), baja la señal.
                # Si los otros confirman, se mantiene o sube.
                adjusted_signal = (raw_signals[i] + mean_influence) / 2
            else:
                # Si es el único módulo activo, no hay ajuste por correlación
                adjusted_signal = raw_signals[i]
                
            adjusted_signals[module] = adjusted_signal
                
        return adjusted_signals
        
    def _calculate_dynamic_weights(
        self,
        module_results: Dict[str, Any],
        market_regime: str = None
    ) -> Dict[str, float]:
        """Calcula pesos dinámicos basados en confianza y régimen de mercado"""
        dynamic_weights = {}
        
        # Multiplicadores por Régimen
        regime_multipliers = {}
        if market_regime == 'TRENDING':
            regime_multipliers = {
                'smc_ict': 2.0,
                'carry_trade': 2.0,
                'market_making': 0.0
            }
        elif market_regime == 'RANGING':
            regime_multipliers = {
                'market_making': 2.5,
                'liquidity_provision': 2.0,
                'carry_trade': 0.0,
                'smc_ict': 0.5
            }
        
        for module, result in module_results.items():
            base_weight = self.module_weights.get(module, 1.0)
            
            # Aplicar multiplicador de régimen
            multiplier = regime_multipliers.get(module, 1.0)
            base_weight *= multiplier
            
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
        """Genera el consenso usando Net Power (Convicción Neta Promedio)"""
        
        long_power = 0.0
        short_power = 0.0
        active_count = 0
        
        # Usamos la confianza directa de los módulos para "Power"
        # Ignoramos adjusted_signals y dynamic_weights para este cálculo específico
        # para cumplir con la lógica de "Suma de Confianza / Conteo".
        
        for module, result in module_results.items():
            confidence = result.get('confidence', 0.0)
            rec = result.get('recommendation', 'neutral')
            
            if rec == 'long':
                long_power += confidence
                active_count += 1
            elif rec == 'short':
                short_power += confidence
                active_count += 1
            
        if active_count > 0:
            net_power = long_power - short_power
            final_confidence = abs(net_power) / active_count
            signal_numeric = net_power / active_count
        else:
            net_power = 0.0
            final_confidence = 0.0
            signal_numeric = 0.0
            
        # Definir Umbral de Activación (0.10)
        # Si la confianza neta promedio es muy baja, nos mantenemos neutrales
        if final_confidence < 0.10:
            recommendation = "neutral"
        else:
            recommendation = "long" if net_power > 0 else "short"
            
        # Métricas secundarias
        module_consistency = self._calculate_module_consistency(
            module_results,
            recommendation
        )
        
        coverage = len(module_results) / len(self.module_weights)
        
        return {
            'recommendation': recommendation,
            'confidence': final_confidence,
            'signal': signal_numeric, # Mapeado a signal_numeric como net_power promedio
            'consistency': module_consistency,
            'coverage': coverage,
            'details': {
                'long_power': long_power,
                'short_power': short_power,
                'active_count': active_count,
                'net_power': net_power
            }
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