import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from .base_module import BaseAnalysisModule

class BrokerBehaviorModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("broker_behavior")
        self.required_fields = [
            'orderbook',
            'trades',
            'volume_profile'
        ]
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el comportamiento de los brokers"""
        if not self.validate_data(data, self.required_fields):
            return self.format_result("neutral", 0.0, "Datos insuficientes")
            
        # Analizar componentes
        order_flow = self._analyze_order_flow(data['orderbook'])
        trade_impact = self._analyze_trade_impact(data['trades'])
        volume_clusters = self._analyze_volume_clusters(data['volume_profile'])
        
        # Calcular señales individuales
        signals = [
            order_flow['signal'],
            trade_impact['signal'],
            volume_clusters['signal']
        ]
        
        weights = [0.4, 0.4, 0.2]  # Pesos para cada componente
        confidence = self.calculate_confidence(signals, weights)
        
        # Determinar recomendación
        if confidence > 0.7:
            recommendation = "long" if np.mean(signals) > 0 else "short"
        else:
            recommendation = "neutral"
            
        # Generar justificación
        justification = self._generate_justification(
            order_flow,
            trade_impact,
            volume_clusters,
            recommendation
        )
        
        return self.format_result(recommendation, confidence, justification)
    
    def _analyze_order_flow(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza el flujo de órdenes"""
        bids = np.array(orderbook['bids'])
        asks = np.array(orderbook['asks'])
        
        # Calcular desequilibrio de órdenes
        bid_volume = np.sum(bids[:, 1])
        ask_volume = np.sum(asks[:, 1])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        
        # Analizar concentración de órdenes
        bid_concentration = self._calculate_concentration(bids[:, 1])
        ask_concentration = self._calculate_concentration(asks[:, 1])
        
        # Generar señal (-1 a 1)
        signal = imbalance * (bid_concentration - ask_concentration)
        
        return {
            'signal': signal,
            'imbalance': imbalance,
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration
        }
    
    def _analyze_trade_impact(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza el impacto de las operaciones"""
        df = pd.DataFrame(trades)
        
        # Calcular volumen acumulado por dirección
        buy_volume = df[df['side'] == 'buy']['volume'].sum()
        sell_volume = df[df['side'] == 'sell']['volume'].sum()
        
        # Calcular impacto en precio
        price_impact = df.groupby('side')['price_impact'].mean()
        
        # Generar señal
        volume_ratio = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        impact_ratio = price_impact.get('buy', 0) - price_impact.get('sell', 0)
        
        signal = (volume_ratio + impact_ratio) / 2
        
        return {
            'signal': signal,
            'volume_ratio': volume_ratio,
            'impact_ratio': impact_ratio
        }
    
    def _analyze_volume_clusters(self, volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza clusters de volumen"""
        prices = np.array(volume_profile['prices'])
        volumes = np.array(volume_profile['volumes'])
        
        # Encontrar niveles de alta actividad
        mean_vol = np.mean(volumes)
        high_activity = volumes > (mean_vol * 1.5)
        
        # Calcular centro de masa del volumen
        volume_center = np.average(prices, weights=volumes)
        current_price = volume_profile['current_price']
        
        # Generar señal basada en la posición del precio actual
        signal = (current_price - volume_center) / (np.std(prices) * 2)
        signal = np.clip(signal, -1, 1)
        
        return {
            'signal': signal,
            'volume_center': volume_center,
            'high_activity_levels': prices[high_activity].tolist()
        }
    
    def _calculate_concentration(self, volumes: np.ndarray) -> float:
        """Calcula la concentración del volumen (índice Herfindahl)"""
        total_volume = np.sum(volumes)
        if total_volume == 0:
            return 0
            
        shares = volumes / total_volume
        return np.sum(shares * shares)
    
    def _generate_justification(
        self,
        order_flow: Dict[str, Any],
        trade_impact: Dict[str, Any],
        volume_clusters: Dict[str, Any],
        recommendation: str
    ) -> str:
        """Genera una justificación detallada"""
        parts = []
        
        # Analizar flujo de órdenes
        if abs(order_flow['imbalance']) > 0.2:
            direction = "compra" if order_flow['imbalance'] > 0 else "venta"
            parts.append(f"Desequilibrio significativo en el flujo de órdenes hacia {direction}")
            
        # Analizar impacto de trades
        if abs(trade_impact['volume_ratio']) > 0.3:
            direction = "compradores" if trade_impact['volume_ratio'] > 0 else "vendedores"
            parts.append(f"Dominancia de {direction} en el volumen de operaciones")
            
        # Analizar clusters de volumen
        if len(volume_clusters['high_activity_levels']) > 0:
            parts.append("Niveles significativos de actividad detectados")
            
        if not parts:
            return "No hay señales claras en el mercado"
            
        sentiment = "alcista" if recommendation == "long" else "bajista" if recommendation == "short" else "neutral"
        parts.append(f"Sentimiento general {sentiment}")
        
        return ". ".join(parts) + "." 