import streamlit as st
import requests
import pandas as pd
# Asegurarse de que el directorio raíz está en el path si es necesario, 
# pero al correr con 'streamlit run dashboard_tecnico.py' desde la raíz debería funcionar.
import sys
import os
sys.path.append(os.getcwd())

from modulos.database import get_recent_signals

st.set_page_config(page_title="Optima Monitor", layout="wide")

st.title("📊 OptimaTradingV2 - Monitor Técnico")

# Pestañas
tab1, tab2 = st.tabs(["Análisis en Vivo", "Historial de Señales"])

with tab1:
    st.header("Análisis de Mercado Actual")
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        symbol = st.text_input("Símbolo (ej: BTC/USDT)", "BTC/USDT")
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        run_btn = st.button("Ejecutar Análisis", type="primary")
    
    if run_btn:
        try:
            # Normalizar símbolo para la URL (usualmente las APIs REST no usan / en path params si no está codificado)
            # Asumimos que el backend espera BTCUSDT o similar
            api_symbol = symbol.replace("/", "").upper()
            
            with st.spinner(f"Analizando {symbol}..."):
                # URL del backend - puerto 8000
                response = requests.get(f"http://localhost:8000/analyze/{api_symbol}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Métricas Principales
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Recomendación", data.get("recommendation", "N/A"))
                col2.metric("Confianza", f"{data.get('confidence', 0.0):.2%}")
                col3.metric("Precio", f"${data.get('current_price', 0.0):,.2f}")
                col4.metric("Señal Numérica", f"{data.get('signal', 0.0):.2f}")
                
                # Alerta o Info
                st.subheader("Justificación")
                if "STRONG" in data.get("recommendation", ""):
                    st.warning(data.get("justification", "Sin justificación"))
                else:
                    st.info(data.get("justification", "Sin justificación"))
                
                # Detalles Técnicos (JSON Expandible)
                with st.expander("Ver Detalles de Motores (JSON)"):
                    st.json(data.get("module_results", {}))
                    
            else:
                st.error(f"Error en API: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("⛔ No se pudo conectar con el Backend (localhost:8000). Asegúrate de que la API esté corriendo (`python -m main.main`).")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado: {str(e)}")

with tab2:
    st.header("Últimas Señales en Base de Datos")
    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 Refrescar Historial"):
            st.rerun()

    try:
        signals = get_recent_signals(limit=20)
        if signals:
            df = pd.DataFrame(signals)
            # Reordenar columnas si existen
            cols = ['timestamp', 'asset', 'signal', 'confidence', 'justification']
            available_cols = [c for c in cols if c in df.columns]
            if available_cols:
                st.dataframe(df[available_cols], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No hay señales registradas en la base de datos todavía.")
    except Exception as e:
        st.error(f"Error al leer la base de datos: {e}")
