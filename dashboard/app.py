import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any
import os

from main.main import OptimatradingMain
from explainer.explainer import AnalysisExplainer

# Configuración de la página
st.set_page_config(
    page_title="Optimatrading Dashboard",
    page_icon="📈",
    layout="wide"
)

class OptimatradingDashboard:
    def __init__(self):
        self.main = OptimatradingMain()
        self.explainer = AnalysisExplainer()
        
    def run(self):
        """Ejecuta el dashboard"""
        self._render_header()
        
        # Sidebar para configuración
        with st.sidebar:
            self._render_sidebar()
            
        # Contenido principal
        if 'selected_asset' in st.session_state:
            self._render_analysis()
            
    def _render_header(self):
        """Renderiza el encabezado"""
        st.title("📈 Optimatrading Dashboard")
        st.markdown("""
        Sistema de análisis de trading que integra múltiples estrategias y genera
        recomendaciones basadas en consenso.
        """)
        
    def _render_sidebar(self):
        """Renderiza la barra lateral"""
        st.sidebar.header("Configuración")
        
        # Selector de activo
        asset = st.sidebar.text_input(
            "Símbolo del activo",
            value=st.session_state.get('selected_asset', ''),
            help="Ingrese el símbolo del activo a analizar"
        )
        
        if st.sidebar.button("Analizar"):
            st.session_state['selected_asset'] = asset
            st.session_state['analysis_result'] = self.main.run_analysis(asset)
            st.session_state['explanations'] = self.explainer.explain_analysis(
                st.session_state['analysis_result']
            )
            
    def _render_analysis(self):
        """Renderiza el análisis principal"""
        if 'analysis_result' not in st.session_state:
            return
            
        result = st.session_state['analysis_result']
        explanations = st.session_state.get('explanations', {})
        
        # Dividir en columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_recommendation(result)
            self._render_explanation_tabs(explanations)
            
        with col2:
            self._render_confidence_metrics(result)
            self._render_module_results(result)
            
    def _render_main_recommendation(self, result: Dict[str, Any]):
        """Renderiza la recomendación principal"""
        st.header("Recomendación Principal")
        
        # Crear tarjeta de recomendación
        recommendation = result['recommendation'].upper()
        confidence = result['confidence']
        
        color = {
            'LONG': 'success',
            'SHORT': 'danger',
            'NEUTRAL': 'warning'
        }.get(recommendation, 'warning')
        
        st.markdown(f"""
        <div style='padding: 1em; border-radius: 5px; background-color: {self._get_color(color)}'>
            <h2 style='margin: 0; color: white'>{recommendation}</h2>
            <p style='margin: 0; color: white'>Confianza: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Justificación:** {result['justification']}")
        
    def _render_explanation_tabs(self, explanations: Dict[str, str]):
        """Renderiza las pestañas de explicación"""
        st.header("Análisis Detallado")
        
        tabs = st.tabs(["Resumen", "Detallado", "Técnico"])
        
        with tabs[0]:
            st.markdown(explanations.get('summary', 'No hay resumen disponible'))
            
        with tabs[1]:
            st.markdown(explanations.get('detailed', 'No hay análisis detallado disponible'))
            
        with tabs[2]:
            st.markdown(explanations.get('technical', 'No hay análisis técnico disponible'))
            
    def _render_confidence_metrics(self, result: Dict[str, Any]):
        """Renderiza métricas de confianza"""
        st.header("Métricas de Confianza")
        
        consensus = result.get('consensus_details', {}).get('raw_consensus', {})
        
        metrics = {
            'Confianza General': result['confidence'],
            'Consistencia': consensus.get('consistency', 0),
            'Cobertura': consensus.get('coverage', 0)
        }
        
        for label, value in metrics.items():
            self._render_metric_gauge(label, value)
            
    def _render_module_results(self, result: Dict[str, Any]):
        """Renderiza resultados de módulos individuales"""
        st.header("Resultados por Módulo")
        
        module_results = result.get('module_results', {})
        weights = result.get('consensus_details', {}).get('dynamic_weights', {})
        
        for module, details in module_results.items():
            with st.expander(f"{module} ({weights.get(module, 0):.2%})"):
                recommendation = details['recommendation'].upper()
                confidence = details['confidence']
                
                st.markdown(f"""
                **Recomendación:** {recommendation}  
                **Confianza:** {confidence:.1%}  
                **Justificación:** {details['justification']}
                """)
                
    def _render_metric_gauge(self, label: str, value: float):
        """Renderiza un medidor para una métrica"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value * 100,
            title = {'text': label},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': self._get_gauge_color(value)},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ]
            }
        ))
        
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        
    @staticmethod
    def _get_color(color: str) -> str:
        """Obtiene el color CSS"""
        colors = {
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107'
        }
        return colors.get(color, '#6c757d')
        
    @staticmethod
    def _get_gauge_color(value: float) -> str:
        """Obtiene el color para el medidor basado en el valor"""
        if value < 0.33:
            return "#dc3545"  # Rojo
        elif value < 0.66:
            return "#ffc107"  # Amarillo
        else:
            return "#28a745"  # Verde
            
if __name__ == "__main__":
    dashboard = OptimatradingDashboard()
    dashboard.run() 