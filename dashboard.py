import streamlit as st
import ccxt
import pandas as pd
import plotly.graph_objects as go
import requests
from ta.trend import EMAIndicator
import time

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Optima Trading V2",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CONSTANTES ---
API_URL = "https://optimatradingv2.onrender.com"
SYMBOL_CCXT = "BTC/USD"
SYMBOL_API = "BTCUSD"

# --- FUNCIONES ---

from typing import cast, Any

def get_market_data():
    """Descarga datos de Kraken y calcula EMAs."""
    try:
        exchange = ccxt.kraken()
        ohlcv = exchange.fetch_ohlcv(SYMBOL_CCXT, timeframe='1h', limit=100)
        columns = cast(Any, ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = pd.DataFrame(ohlcv, columns=columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calcular EMAs
        ema50 = EMAIndicator(close=df['close'], window=50).ema_indicator()
        ema200 = EMAIndicator(close=df['close'], window=200).ema_indicator()
        
        df['EMA50'] = ema50
        df['EMA200'] = ema200
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos de mercado: {e}")
        return None

def get_ceo_verdict():
    """Obtiene el análisis en tiempo real de la API."""
    try:
        response = requests.get(f"{API_URL}/analyze/{SYMBOL_API}", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error API Verdict: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error conectando a API Verdict: {e}")
        return None

def get_sniper_history():
    """Obtiene el historial de señales."""
    try:
        response = requests.get(f"{API_URL}/history", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error API History: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error conectando a API History: {e}")
        return None

# --- UI LAYOUT ---

# SECCIÓN 1: HEADER
st.title("🚀 Optima Trading V2 - Command Center")
if st.button('🔄 Actualizar Dashboard'):
    st.rerun()

st.divider()

# SECCIÓN 2: LIVE MARKET CHART
st.header("1. Live Market Structure (BTC/USD 1h)")

market_df = get_market_data()

if market_df is not None:
    # Crear gráfico con Plotly
    fig = go.Figure()

    # Velas
    fig.add_trace(go.Candlestick(
        x=market_df['timestamp'],
        open=market_df['open'],
        high=market_df['high'],
        low=market_df['low'],
        close=market_df['close'],
        name='Price'
    ))

    # EMA 50
    fig.add_trace(go.Scatter(
        x=market_df['timestamp'],
        y=market_df['EMA50'],
        line=dict(color='orange', width=2),
        name='EMA 50'
    ))

    # EMA 200
    fig.add_trace(go.Scatter(
        x=market_df['timestamp'],
        y=market_df['EMA200'],
        line=dict(color='blue', width=2),
        name='EMA 200'
    ))

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=f"BTC/USD - Últimas 100 Velas"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# SECCIÓN 3: THE CEO VERDICT
st.header("2. The CEO Verdict 🤖")

verdict_data = get_ceo_verdict()

if verdict_data:
    col1, col2, col3 = st.columns([1, 1, 2])
    
    rec = verdict_data.get('recommendation', 'NEUTRAL').upper()
    conf = verdict_data.get('confidence', 0.0)
    just = verdict_data.get('justification', 'No justification')
    
    # Definir color
    color = "gray"
    if "BUY" in rec or "LONG" in rec:
        color = "green"
    elif "SELL" in rec or "SHORT" in rec:
        color = "red"
        
    with col1:
        st.markdown(f"### Recommendation")
        st.markdown(f":{color}[**{rec}**]")
        
    with col2:
        st.metric("Confidence", f"{conf:.1%}")
        
    with col3:
        st.info(f"**Justification:** {just}")
        
    # Mostrar detalles técnicos RAW (opcional, expandible)
    with st.expander("Ver Detalles Técnicos (Raw JSON)"):
        st.json(verdict_data)

st.divider()

# SECCIÓN 4: SNIPER HISTORY
st.header("3. Sniper Signal History 📜")

history_data = get_sniper_history()

if history_data and 'history' in history_data:
    signals = history_data['history']
    if len(signals) > 0:
        df_hist = pd.DataFrame(signals)
        # Reordenar columnas para legibilidad
        cols = ['timestamp', 'asset', 'signal', 'confidence', 'justification', 'id']
        # Filtrar columnas que existan
        cols = [c for c in cols if c in df_hist.columns]
        st.dataframe(df_hist[cols], use_container_width=True)
    else:
        st.info("No hay historial de señales guardado aún.")
else:
    st.warning("No se pudo cargar el historial.")

