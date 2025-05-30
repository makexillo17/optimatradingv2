import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .base_module import BaseAnalysisModule

class SmcIctModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("smc_ict")
        self.required_fields = [
            'price_data',
            'volume_data',
            'order_blocks',
            'liquidity_levels',
            'market_structure'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el mercado usando conceptos SMC/ICT"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        order_flow = self._analyze_order_flow(
            data['price_data'],
            data['volume_data']
        )
        
        structure = self._analyze_market_structure(
            data['market_structure']
        )
        
        liquidity = self._analyze_liquidity_levels(
            data['liquidity_levels'],
            data['order_blocks']
        )
        
        # Calcular señales
        signals = [
            order_flow['signal'],
            structure['signal'],
            liquidity['signal']
        ]
        
        weights = [0.4, 0.3, 0.3]
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        justification = self._generate_justification(
            order_flow,
            structure,
            liquidity,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_order_flow(
        self,
        price_data: Dict[str, Any],
        volume_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza el flujo de órdenes y volumen"""
        # Analizar imbalance de volumen
        buy_volume = volume_data['buy_volume']
        sell_volume = volume_data['sell_volume']
        volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        # Analizar delta de precio
        open_price = price_data['open']
        close_price = price_data['close']
        high_price = price_data['high']
        low_price = price_data['low']
        
        price_range = high_price - low_price
        close_position = (close_price - low_price) / price_range
        
        # Analizar momentum
        momentum = price_data['momentum_score']
        
        # Generar señal combinada
        signal = np.clip(
            volume_imbalance * 0.4 + (close_position - 0.5) * 0.3 + momentum * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'volume_imbalance': volume_imbalance,
            'close_position': close_position,
            'momentum': momentum
        }
    
    def _analyze_market_structure(
        self,
        market_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza la estructura del mercado"""
        # Analizar niveles de estructura
        trend_direction = market_structure['trend_direction']  # 1: up, -1: down, 0: sideways
        structure_strength = market_structure['structure_strength']
        
        # Analizar swing points
        recent_highs = market_structure['recent_highs']
        recent_lows = market_structure['recent_lows']
        
        # Calcular tendencia de swings
        swing_trend = self._calculate_swing_trend(recent_highs, recent_lows)
        
        # Analizar breaks de estructura
        break_direction = market_structure['break_direction']
        break_strength = market_structure['break_strength']
        
        # Generar señal
        base_signal = trend_direction * structure_strength
        break_signal = break_direction * break_strength
        swing_signal = swing_trend
        
        signal = np.clip(
            base_signal * 0.4 + break_signal * 0.3 + swing_signal * 0.3,
            -1, 1
        )
        
        return {
            'signal': signal,
            'trend_direction': trend_direction,
            'structure_strength': structure_strength,
            'break_direction': break_direction
        }
    
    def _analyze_liquidity_levels(
        self,
        liquidity_levels: Dict[str, Any],
        order_blocks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analiza niveles de liquidez y bloques de órdenes"""
        # Analizar niveles de liquidez
        nearest_level = liquidity_levels['nearest_level']
        distance_to_level = liquidity_levels['distance_to_level']
        level_strength = liquidity_levels['level_strength']
        
        # Analizar bloques de órdenes
        nearest_block = order_blocks['nearest_block']
        block_type = order_blocks['block_type']  # 1: demand, -1: supply
        block_strength = order_blocks['block_strength']
        
        # Calcular señales de niveles
        level_signal = level_strength * (1 - min(distance_to_level, 1))
        block_signal = block_type * block_strength
        
        # Generar señal combinada
        signal = np.clip(
            level_signal * 0.5 + block_signal * 0.5,
            -1, 1
        )
        
        return {
            'signal': signal,
            'nearest_level': nearest_level,
            'level_strength': level_strength,
            'block_type': block_type,
            'block_strength': block_strength
        }
    
    def _calculate_swing_trend(
        self,
        recent_highs: List[float],
        recent_lows: List[float]
    ) -> float:
        """Calcula la tendencia basada en swings"""
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return 0.0
            
        # Calcular pendientes
        high_slope = (recent_highs[-1] - recent_highs[-2]) / 1
        low_slope = (recent_lows[-1] - recent_lows[-2]) / 1
        
        # Combinar pendientes
        trend = (high_slope + low_slope) / 2
        
        # Normalizar
        return np.clip(trend / 0.01, -1, 1)  # Normalizar con 1% como referencia
    
    def _generate_justification(
        self,
        order_flow: Dict[str, Any],
        structure: Dict[str, Any],
        liquidity: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar flujo de órdenes
        if abs(order_flow['volume_imbalance']) > 0.3:
            direction = "comprador" if order_flow['volume_imbalance'] > 0 else "vendedor"
            parts.append(f"Desequilibrio {direction} en el flujo de órdenes")
            
        # Analizar estructura
        if abs(structure['trend_direction']) > 0.7:
            trend = "alcista" if structure['trend_direction'] > 0 else "bajista"
            parts.append(f"Estructura de mercado {trend} con fuerza {structure['structure_strength']:.2f}")
            
        # Analizar liquidez
        if liquidity['block_strength'] > 0.7:
            block = "demanda" if liquidity['block_type'] > 0 else "oferta"
            parts.append(f"Bloque de {block} fuerte cercano")
            
        if not parts:
            return "No hay señales claras en la estructura del mercado"
            
        sentiment = "alcista" if recommendation == "long" else "bajista" if recommendation == "short" else "neutral"
        parts.append(f"Sesgo {sentiment} basado en análisis SMC/ICT")
        
        return ". ".join(parts) + "." 