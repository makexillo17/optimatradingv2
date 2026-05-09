import json
import asyncio
import numpy as np

# Registro de Autoridad
AUTHORITY_SCORES = {
    'OFFICIAL': 1.0, # SEC / FED / Central Banks
    'REUTERS': 0.95, # Bloomberg, WSJ, Reuters
    'CRYPTO_T1': 0.80, # CoinDesk, Cointelegraph
    'SOCIAL': 0.40 # Twitter, Reddit
}

class SentimentService:
    def __init__(self):
        # En produccion, aqui se cargaria el modelo ONNX:
        # import onnxruntime as ort
        # self.session = ort.InferenceSession("phi3.5-mini.onnx")
        self.is_ready = True
        
    async def analyze_news_feed(self, text: str, source: str = 'REUTERS'):
        """
        Simula el pipeline Phi-3.5 SLM Local.
        Latencia garantizada < 100ms.
        """
        # Simulamos latencia de red neuronal
        await asyncio.sleep(0.05) 
        
        # Simulamos la inferencia del modelo
        # Extraemos variables ficticias basadas en el texto mockeado
        sentiment = np.random.uniform(-1.0, 1.0)
        urgency = np.random.uniform(0.1, 1.0)
        
        # Si el texto dice "crash" o "hacked", forzamos el panico
        if 'crash' in text.lower() or 'hack' in text.lower() or 'sec sues' in text.lower():
            sentiment = -0.95
            urgency = 0.99
            
        S_pol = abs(sentiment) * 10  # Escalamiento para el algoritmo
        W_urg = urgency
        A_cred = AUTHORITY_SCORES.get(source, 0.5)
        
        # Algoritmo de impacto
        impact_raw = int(S_pol * W_urg * A_cred)
        impact = min(5, max(1, impact_raw))
        
        result = {
            "ticker": "BTC",
            "impact": impact,
            "sentiment": sentiment,
            "urgency": urgency
        }
        
        return result

# Singleton para uso en Anemona
sentiment_engine = SentimentService()
