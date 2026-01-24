import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class StatArbModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("stat_arb")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis de Arbitraje Estadístico (Mean Reversion) usando Z-Score.
        Detecta desviaciones extremas del precio respecto a su media (Distribución Normal).
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 20:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas (Min 20)")
            
            df = market_data.copy()
            
            # --- 1. CÁLCULO DE Z-SCORE ---
            
            # SMA (20)
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Desviación Estándar (20)
            df['std_20'] = df['close'].rolling(window=20).std()
            
            # Z-Score Formula: (Precio - Media) / Desviación
            df['z_score'] = (df['close'] - df['sma_20']) / df['std_20']
            
            current_z = df['z_score'].iloc[-1]
            
            # --- 2. LÓGICA DE REVERSIÓN A LA MEDIA ---
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Z-Score normal ({current_z:.2f}σ). Ruido de mercado habitual."
            
            # Extremo Bajista (Oportunidad de Compra)
            if current_z < -2.0:
                if current_z < -3.0:
                    signal = "long"
                    confidence = 0.95 # Confianza máxima (Anomalía extrema)
                    justification = f"📉 Z-SCORE EXTREMO ({current_z:.2f}σ): ¡ANOMALÍA! Precio extremadamente infravalorado."
                else:
                    signal = "long"
                    confidence = 0.8
                    justification = f"📉 Z-SCORE BAJO ({current_z:.2f}σ): Precio infravalorado estadísticamente. Esperando reversión a la media."
            
            # Extremo Alcista (Oportunidad de Venta)
            elif current_z > 2.0:
                if current_z > 3.0:
                    signal = "short"
                    confidence = 0.95
                    justification = f"📈 Z-SCORE EXTREMO ({current_z:.2f}σ): ¡BURBUJA LOCAL! Precio extremadamente sobreextendido."
                else:
                    signal = "short"
                    confidence = 0.8
                    justification = f"📈 Z-SCORE ALTO ({current_z:.2f}σ): Precio sobreextendido."
            
            # Resultado
            result = self.format_result(signal, confidence, justification)
            # Asegurar que z_score sea float nativo para JSON serialization
            result['z_score'] = float(current_z) if not np.isnan(current_z) else 0.0
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Stat Arb: {str(e)}")
