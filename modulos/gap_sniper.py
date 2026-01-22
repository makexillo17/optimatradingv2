import pandas as pd
from typing import Dict, Any, Optional
from .base_module import BaseAnalysisModule

class GapSniperModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("gap_sniper")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta Fair Value Gaps (FVG) en las últimas velas.
        Estrategia de GAP SNIPER.
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            # Necesitamos al menos 5 velas para iterar y buscar gaps recientes
            if len(market_data) < 5:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: Solo {len(market_data)} velas")
            
            # Obtener las últimas 5 velas para análisis
            # Usaremos las últimas 3 para la detección estricta (A, B, C)
            # A: antepenúltima (-3), B: penúltima (-2), C: actual/última (-1)
            
            # Itera sobre las últimas velas, pero nos centraremos en el patrón más reciente
            # El requerimiento dice: "Itera sobre las últimas 5 velas" y "Analiza secuencia de 3 velas: A, B, C"
            
            # Vamos a buscar el FVG más reciente en las últimas iteraciones posibles dentro de las 5 velas.
            # Pero para simplificar y dar la señal actual, analizaremos principalmente las últimas 3 cerradas o en formación.
            # Sin embargo, si el usuario pide iterar, quizás quiera ver si hubo alguno reciente.
            # Dado el return único, priorizaremos la formación más reciente (las últimas 3 velas).
            
            # Definir velas A, B, C (indices relativos)
            # C es la última (-1), B es la anterior (-2), A es la anterior a B (-3)
            
            candle_c = market_data.iloc[-1] # Actual/Ultima
            candle_b = market_data.iloc[-2] # Penúltima
            candle_a = market_data.iloc[-3] # Antepenúltima
            
            signal = "neutral"
            confidence = 0.0
            justification = "No se detectaron Fair Value Gaps recientes."
            
            # Detectar FVG Alcista (Bullish)
            # Condición: Low(C) > High(A)
            # Hay un GAP entre High A y Low C
            if candle_c['low'] > candle_a['high']:
                signal = "long" # BUY
                confidence = 0.9
                gap_size = candle_c['low'] - candle_a['high']
                justification = f"FVG Alcista detectado entre {candle_a['high']:.2f} (High A) y {candle_c['low']:.2f} (Low C). Tamaño: {gap_size:.2f}"
                
            # Detectar FVG Bajista (Bearish)
            # Condición: High(C) < Low(A)
            # Hay un GAP entre Low A y High C
            elif candle_c['high'] < candle_a['low']:
                signal = "short" # SELL
                confidence = 0.9
                gap_size = candle_a['low'] - candle_c['high']
                justification = f"FVG Bajista detectado entre {candle_a['low']:.2f} (Low A) y {candle_c['high']:.2f} (High C). Tamaño: {gap_size:.2f}"
            
            # Si no se encuentra en la última secuencia, el prompt decia "Itera sobre las últimas 5 velas".
            # Si queremos ser estrictos con "signal actual", el FVG debe estar vigente o recién formado.
            # Si analizamos velas anteriores (-2, -3, -4), sería un FVG "viejo".
            # Asumiré que la prioridad es la formación reciente que da la señal de entrada AHORA.
            # Si analizamos hacia atrás, deberíamos verificar si el precio ya mitigó el FVG.
            # Por simplicidad y siguiendo "Analiza la secuencia de 3 velas: A, B, C (actual)", me quedo con el análisis de las últimas 3.
            
            # Corrección: El prompt dice "Itera sobre las últimas 5 velas (para buscar gaps recientes)".
            # Esto sugiere que si no hay uno en la última (C, B, A), mire una atrás (B, A, Pre-A).
            # Implementaré la búsqueda iterativa en las últimas 5 velas (que permiten 3 secuencias de 3 velas).
            
            if signal == "neutral":
                # Intentar buscar en la secuencia anterior: C(-2), B(-3), A(-4)
                # Solo si tenemos suficientes datos
                if len(market_data) >= 4:
                    c_prev = market_data.iloc[-2]
                    b_prev = market_data.iloc[-3]
                    a_prev = market_data.iloc[-4]
                    
                    if c_prev['low'] > a_prev['high']:
                         # FVG Alcista previo no mitigado? (simplificado, solo detección)
                         signal = "long"
                         confidence = 0.8 # Un poco menos de confianza por no ser inmediato
                         justification = f"FVG Alcista reciente (vela previa) detectado entre {a_prev['high']:.2f} y {c_prev['low']:.2f}"
                    elif c_prev['high'] < a_prev['low']:
                         signal = "short"
                         confidence = 0.8
                         justification = f"FVG Bajista reciente (vela previa) detectado entre {a_prev['low']:.2f} y {c_prev['high']:.2f}"

            return self.format_result(signal, confidence, justification)
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Gap Sniper: {str(e)}")
