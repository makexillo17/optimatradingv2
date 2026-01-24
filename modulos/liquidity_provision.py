import numpy as np
import pandas as pd
from typing import Dict, Any
from .base_module import BaseAnalysisModule

class LiquidityProvisionModule(BaseAnalysisModule):
    def __init__(self):
        super().__init__("liquidity_provision")
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proveedor de Liquidez: Reversión a la Media con Canales de Regresión Lineal.
        Opera extremos estadísticos (2 Sigma) filtrando tendencias fuertes (ADX).
        """
        try:
            market_data = data.get('market_data')
            
            if market_data is None or not isinstance(market_data, pd.DataFrame):
                return self.format_result("neutral", 0.0, "Datos insuficientes: No hay DataFrame")
            
            if len(market_data) < 30:
                return self.format_result("neutral", 0.0, f"Datos insuficientes: {len(market_data)} velas (Min 30)")
            
            df = market_data.copy()
            
            # --- 1. INDICADORES (ADX) ---
            from ta.trend import ADXIndicator
            
            adx_ind = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['adx'] = adx_ind.adx()
            current_adx = df['adx'].iloc[-1]
            
            # --- 2. CANALES DE REGRESIÓN LINEAL (20) ---
            # Usamos los últimos 20 cierres
            window_size = 20
            subset = df['close'].tail(window_size).values
            x = np.arange(window_size)
            
            # Ajuste Lineal (Polyfit grado 1)
            # y = mx + b
            slope, intercept = np.polyfit(x, subset, 1)
            
            # Línea de Regresión (Fair Value)
            regression_line = slope * x + intercept
            
            # Calcular Residuales y Desviación Estándar
            residuals = subset - regression_line
            std_dev = np.std(residuals)
            
            # Valores actuales (último punto de la línea)
            current_fair_value = regression_line[-1]
            
            # Canales (2 Sigma)
            upper_channel = current_fair_value + (2.0 * std_dev)
            lower_channel = current_fair_value - (2.0 * std_dev)
            
            current_close = df['close'].iloc[-1]
            current_open = df['open'].iloc[-1]
            distance_to_mean = current_close - current_fair_value
            
            # --- 3. LÓGICA DE REVERSIÓN ---
            
            signal = "neutral"
            confidence = 0.0
            justification = f"Precio en rango estadístico normal. Distancia a media: {distance_to_mean:.2f}"
            
            # FILTRO DE SEGURIDAD: NO operar reversión si la tendencia es muy fuerte
            if current_adx > 40:
                signal = "neutral"
                confidence = 0.5
                justification = f"🚫 FILTRO ADX ACTIVO: Tendencia fuerte detectada (ADX={current_adx:.1f}). Reversión cancelada."
            else:
                # Oportunidad de COMPRA (Soporte Estadístico)
                # Toca linea inferior y Vela Alcista (Rechazo)
                # Usamos low/high para "tocar" o close para romper
                # Prompt dice: "Toca o rompe... y Cierre > Apertura"
                if df['low'].iloc[-1] <= lower_channel:
                    if current_close > current_open:
                        signal = "long"
                        confidence = 0.75
                        justification = "🛡️ SOPORTE ESTADÍSTICO: Precio en -2 Sigma de Regresión. Probable reversión a la media."
                
                # Oportunidad de VENTA (Resistencia Estadística)
                elif df['high'].iloc[-1] >= upper_channel:
                    if current_close < current_open:
                        signal = "short"
                        confidence = 0.75
                        justification = "🛡️ RESISTENCIA ESTADÍSTICA: Precio en +2 Sigma de Regresión. Sobre-extensión."

            result = self.format_result(signal, confidence, justification)
            result.update({
                'distance_to_mean': float(distance_to_mean),
                'linear_regression_price': float(current_fair_value),
                'std_dev_channel': float(std_dev),
                'adx_value': float(current_adx)
            })
            
            return result
            
        except Exception as e:
            return self.format_result("neutral", 0.0, f"Error en análisis Liquidity Provision: {str(e)}")
