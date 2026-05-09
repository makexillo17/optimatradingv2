import asyncio
import grpc
import psycopg2
import time
import math
import numpy as np
from datetime import datetime, timedelta
import protos.anemona_pb2 as anemona_pb2
import protos.anemona_pb2_grpc as anemona_pb2_grpc
from sentiment_service import sentiment_engine

# Configuración QuestDB (Postgres Wire)
QDB_HOST = "localhost"
QDB_PORT = 8812
QDB_USER = "admin"
QDB_PASS = "quest"

def setup_questdb():
    try:
        conn = psycopg2.connect(host=QDB_HOST, port=QDB_PORT, user=QDB_USER, password=QDB_PASS, database="qdb")
        conn.autocommit = True
        cur = conn.cursor()
        
        # DDL
        cur.execute("""
        CREATE TABLE IF NOT EXISTS institutional_flows (
            timestamp TIMESTAMP,
            asset_symbol SYMBOL,
            flow_direction SYMBOL,
            volume_usd DOUBLE,
            entity_class SYMBOL,
            confidence_score DOUBLE
        ) TIMESTAMP(timestamp) PARTITION BY DAY 
        DEDUP UPSERT KEYS(timestamp, asset_symbol, entity_class);
        """)
        print("[QuestDB] Tabla institutional_flows verificada.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[QuestDB] No se pudo conectar a QuestDB en el puerto 8812: {e}")
        print("Asegúrate de ejecutar QuestDB en Docker: docker run -p 9000:9000 -p 8812:8812 questdb/questdb")

class AlphaSignalEngineServicer(anemona_pb2_grpc.AlphaSignalEngineServicer):
    
    async def get_etf_signal(self):
        # Pilar 1 (TradFi/ETFs): Mock de Scraping Asincrono
        # Umbral $200M USD
        await asyncio.sleep(0.01)
        # Simulamos flujos
        inflows = np.random.normal(100, 150) # En Millones USD
        gbtc_outflows = abs(np.random.normal(50, 80)) # Salidas de GBTC
        
        # Filtro de estabilización T+3 para GBTC (simplificado)
        net_flow = inflows - (gbtc_outflows * 0.5) 
        
        signal = 0.0
        if net_flow > 200:
            signal = 1.0 # Bias alcista fuerte
        elif net_flow < -200:
            signal = -1.0
        else:
            signal = net_flow / 200.0
            
        return signal

    async def get_sec_proxy_signal(self):
        # Pilar 2 (SEC/Proxy): Mineria 13F-HR / MSTR NAV Delta
        await asyncio.sleep(0.01)
        # Delta_MSTR = (MSTR Market Cap / BTC Holdings Value) - 1
        # Si la delta baja (descuento) o sube aceleradamente (prima institucional)
        delta_mstr = np.random.normal(1.5, 0.2)
        if delta_mstr > 1.8:
            return 0.8 # Acumulacion indirecta
        elif delta_mstr < 1.0:
            return -0.5
        return 0.0

    async def get_onchain_signal(self):
        # Pilar 3 (On-Chain): OTC Outflows > 10k BTC
        await asyncio.sleep(0.01)
        outflow_btc = np.random.exponential(2000)
        if outflow_btc > 10000:
            return 1.0 # Acumulación Silenciosa Fuerte
        return 0.0

    async def check_triple_witching(self):
        # Escudo de Gestión de Riesgo (Triple Witching)
        # Ocurre los 3ros viernes de Marzo, Junio, Septiembre, Diciembre
        now = datetime.utcnow()
        # Simulamos si faltan < 48h (Mock)
        if now.month in (3, 6, 9, 12) and 15 <= now.day <= 21 and now.weekday() in (3, 4):
            return True
        return False

    async def SubscribePositionMultipliers(self, request, context):
        print(f"[Anemona] Cliente {request.client_id} subscrito a flujo Alpha.")
        
        w1, w2, w3 = 0.4, 0.3, 0.3 # Pesos de los factores
        
        while True:
            # 1. Recoleccion Concurrente de Pilares y Sentiment
            # Simulamos texto de noticias entrantes
            mock_news = "Bitcoin shows resilience amidst macro uncertainty" if np.random.rand() > 0.1 else "SEC sues major crypto exchange"
            mock_source = "REUTERS" if np.random.rand() > 0.5 else "OFFICIAL"
            
            sig_etf, sig_sec, sig_onchain, is_witching, nlp_result = await asyncio.gather(
                self.get_etf_signal(),
                self.get_sec_proxy_signal(),
                self.get_onchain_signal(),
                self.check_triple_witching(),
                sentiment_engine.analyze_news_feed(mock_news, source=mock_source)
            )
            
            # 2. Inferencia: Anemona Alpha Factor
            # Normalizacion Winsorized simulada
            raw_af = (w1 * sig_etf) + (w2 * sig_sec) + (w3 * sig_onchain)
            
            # Curva Sigmoidea para evitar outliers
            af_t = (2 / (1 + math.exp(-2 * raw_af))) - 1
            
            # 3. Triple Witching Escudo
            pos_mult = 1.0
            warning = False
            if is_witching:
                pos_mult = 0.5
                warning = True
                
            # 4. Envio por gRPC Stream
            update = anemona_pb2.MultiplierUpdate(
                exact_timestamp_ms=int(time.time() * 1000),
                raw_current_alpha_factor=af_t,
                terminal_position_multiplier=pos_mult,
                severe_macro_rebalance_warning=warning,
                event_impact_level=nlp_result['impact'],
                event_sentiment=nlp_result['sentiment']
            )
            
            yield update
            await asyncio.sleep(1) # Actualización 1Hz

async def serve():
    setup_questdb()
    server = grpc.aio.server()
    anemona_pb2_grpc.add_AlphaSignalEngineServicer_to_server(AlphaSignalEngineServicer(), server)
    server.add_insecure_port('[::]:50051')
    print("[Anemona] Microservicio gRPC corriendo en puerto 50051...")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())
