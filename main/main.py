import logging
import traceback
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI                     #  ←  Import correcto, fuera de la clase
from loader.loader import MarketDataLoader as DataLoader
from dispatcher.dispatcher import ModuleDispatcher
from main.consensus import ConsensusAnalyzer
from utils.logger import setup_logger
from modulos.gap_sniper import GapSniperModule
from modulos.database import init_db, save_signal, get_recent_signals     #  <-- Importar persistencia y query
import ccxt

import os

class OptimatradingMain:
    def __init__(self):
        self.logger = setup_logger("OptimatradingMain")
        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
        
        # Determinar la ruta a pasar (String o None)
        final_config_path = str(config_path) if config_path.exists() else None
        
        if final_config_path:
            self.logger.info(f"Cargando configuración desde: {final_config_path}")
        else:
            self.logger.warning(f"No se encontró config.yaml. Usando configuración por defecto en memoria.")

        # Inicializar DataLoader
        self.data_loader = DataLoader(config_path=final_config_path)
        
        # --- AQUÍ ESTABA EL ERROR ---
        # Antes: self.dispatcher = ModuleDispatcher()
        # Ahora: Le pasamos el path (o None)
        self.dispatcher = ModuleDispatcher(config_path=final_config_path)
        
        self.consensus = ConsensusAnalyzer()
        
        # Inicializar Base de Datos
        init_db()

    # ----------------------------------------------------
    # 1) Método público
    # ----------------------------------------------------
    def run_analysis(self, asset_symbol: str, current_price: Optional[float] = None, market_df: Optional[Any] = None) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de análisis para un activo."""
        try:
            self.logger.info(f"Iniciando análisis para {asset_symbol}")

            # 1. Cargar datos de mercado
            market_data = self._load_market_data(asset_symbol)
            if not market_data:
                return self._generate_error_response("Error cargando datos de mercado")

            # 2. Ejecutar módulos analíticos - pasar el DataFrame si está disponible
            module_results = self._run_analysis_modules(market_df if market_df is not None else market_data)
            if not module_results:
                # Si module_results está vacío, devolver respuesta con precio y status
                return {
                    "timestamp": datetime.now().isoformat(),
                    "asset_symbol": asset_symbol,
                    "recommendation": "HOLD",
                    "current_price": float(current_price) if current_price is not None else None,
                    "status": "waiting_strategies",
                    "module_results": {},
                    "consensus_details": {}
                }

            # 3. Generar consenso
            consensus_result = self._generate_consensus(module_results)

            # 4. Empaquetar respuesta
            final_result = self._format_final_result(
                consensus_result,
                module_results,
                asset_symbol,
            )

            # 5. Guardar Señal en Base de Datos (Persistencia)
            # Guardamos SIEMPRE por ahora, para verificar flujo.
            save_signal(
                asset=asset_symbol,
                signal=consensus_result["recommendation"],
                confidence=consensus_result["confidence"],
                justification=consensus_result["justification"],
                raw_data=module_results  # Guardamos los resultados crudos de los módulos
            )

            self.logger.info(f"Análisis completado para {asset_symbol}")
            return final_result

        except Exception as e:
            self.logger.error(f"Error en análisis: {str(e)}")
            return self._generate_error_response(f"Error inesperado: {str(e)}")

    # ----------------------------------------------------
    # 2) Cargar datos
    # ----------------------------------------------------
    def _load_market_data(self, asset_symbol: str) -> Optional[Dict[str, Any]]:
        """Carga todos los datos necesarios para el análisis."""
        try:
            # Esta es la ÚNICA que funciona actualmente
            market_data = self.data_loader.load_market_data(asset_symbol)
            
            # --- COMENTAR ESTO PARA EVITAR CRASH ---
            # market_data.update(self.data_loader.load_broker_data(asset_symbol))
            # market_data.update(self.data_loader.load_carry_data(asset_symbol))
            # market_data.update(self.data_loader.load_options_data(asset_symbol))
            # market_data.update(self.data_loader.load_liquidity_data(asset_symbol))
            # market_data.update(self.data_loader.load_market_making_data(asset_symbol))
            # market_data.update(self.data_loader.load_pairs_data(asset_symbol))
            # market_data.update(self.data_loader.load_smc_data(asset_symbol))
            # market_data.update(self.data_loader.load_stat_arb_data(asset_symbol))
            # market_data.update(self.data_loader.load_volatility_data(asset_symbol))
            # market_data.update(self.data_loader.load_yield_data(asset_symbol))
            
            return market_data

        except Exception as e:
            self.logger.error(f"Error cargando datos: {str(e)}")
            return None

    # ----------------------------------------------------
    # 3) Ejecutar módulos
    # ----------------------------------------------------
    def _run_analysis_modules(self, market_data) -> Optional[Dict[str, Any]]:
        """
        Ejecuta todos los módulos de análisis.
        
        Args:
            market_data: Puede ser un DataFrame de pandas o un Dict con datos de mercado
        """
        try:
            module_results: Dict[str, Any] = {}
            modules = [
                "broker_behavior",
                "carry_trade",
                "dynamic_hedging",
                "liquidity_provision",
                "market_making",
                "pairs_trading",
                "smc_ict",
                "stat_arb",
                "volatility_arb",
                "yield_anomaly",
                "gap_sniper",
            ]

            for module_name in modules:
                try:
                    result = self.dispatcher.run_module(module_name, market_data)
                    if result:
                        module_results[module_name] = result
                    else:
                        self.logger.warning(f"Módulo {module_name} no generó resultados")
                except Exception as e:
                    # Error individual del módulo - no rompe el programa
                    self.logger.error(f"Error ejecutando módulo {module_name}: {str(e)}")
                    continue

            return module_results if module_results else None

        except Exception as e:
            self.logger.error(f"Error ejecutando módulos: {str(e)}")
            return None

    # ----------------------------------------------------
    # 4) Generar consenso
    # ----------------------------------------------------
    def _generate_consensus(self, module_results: Dict[str, Any]) -> Dict[str, Any]:
        """Genera el consenso final basado en resultados de módulos."""
        try:
            return self.consensus.analyze(module_results)
        except Exception as e:
            self.logger.error(f"Error generando consenso: {str(e)}")
            return {
                "recommendation": "neutral",
                "confidence": 0.0,
                "justification": f"Error generando consenso: {str(e)}",
            }

    # ----------------------------------------------------
    # 5) Formatear salida
    # ----------------------------------------------------
    def _format_final_result(
        self,
        consensus_result: Dict[str, Any],
        module_results: Dict[str, Any],
        asset_symbol: str,
    ) -> Dict[str, Any]:
        """Formatea el resultado final del análisis."""
        return {
            "timestamp": datetime.now().isoformat(),
            "asset_symbol": asset_symbol,
            "recommendation": consensus_result["recommendation"],
            "confidence": consensus_result["confidence"],
            "justification": consensus_result["justification"],
            "module_results": module_results,
            "consensus_details": consensus_result.get("details", {}),
        }

    # ----------------------------------------------------
    # 6) Respuesta de error estándar
    # ----------------------------------------------------
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Genera respuesta de error estándar."""
        return {
            "timestamp": datetime.now().isoformat(),
            "recommendation": "neutral",
            "confidence": 0.0,
            "justification": error_message,
            "module_results": {},
            "consensus_details": {},
            "error": error_message,
        }


# ---------------------------------------------------------------------
# FastAPI app & endpoints
# ---------------------------------------------------------------------
app = FastAPI()

optimatrading = OptimatradingMain()


@app.get("/")
def root():
    return {"message": "Servidor de Optimatrading activo"}


@app.get("/analyze/{asset_symbol}")
def analyze(asset_symbol: str):
    # Normalizar el símbolo para ccxt (asegurar formato correcto)
    # Si viene como 'BTCUSDT', convertirlo a 'BTC/USDT'
    if '/' not in asset_symbol and len(asset_symbol) > 3:
        # Intentar detectar el par (asumiendo que los últimos 4 caracteres son la moneda base)
        # Por ejemplo: BTCUSDT -> BTC/USDT
        if asset_symbol.endswith('USDT'):
            base = asset_symbol[:-4]
            asset_symbol = f"{base}/USDT"
        elif asset_symbol.endswith('USD'):
            base = asset_symbol[:-3]
            asset_symbol = f"{base}/USD"
    
    # Capturar el precio actual del DataFrame y obtener el DataFrame completo
    current_price = None
    market_df = None
    try:
        loader = optimatrading.data_loader
        df = loader.load_broker_data(asset_symbol, timeframe='1h', limit=500)
        
        # Guardar el DataFrame para pasarlo a los módulos
        market_df = df
        
        # Capturar el precio actual (precio de cierre más reciente)
        if len(df) > 0:
            current_price = float(df['close'].iloc[-1])  # Convertir a float nativo de Python
            print(f"Precio de cierre más reciente para {asset_symbol}: {current_price}")
        else:
            print(f"No se obtuvieron datos para {asset_symbol}")
    except Exception as e:
        print(f"Error cargando datos de broker para {asset_symbol}: {str(e)}")
        # Continuar con el análisis aunque falle la carga de broker data
    
    try:
        return optimatrading.run_analysis(asset_symbol, current_price=current_price, market_df=market_df)
    except Exception as e:
        print(traceback.format_exc())
        return {
            "timestamp": datetime.now().isoformat(),
            "recommendation": "neutral",
            "confidence": 0.0,
            "justification": "Error ejecutando módulos de análisis",
            "module_results": {},
            "consensus_details": {},
            "error": f"ERROR REAL: {str(e)}"
        }



@app.get("/history")
def history():
    """Devuelve las últimas 5 señales guardadas en la DB."""
    try:
        signals = get_recent_signals(limit=5)
        return {"count": len(signals), "history": signals}
    except Exception as e:
        return {"error": str(e)}

@app.get("/test-sniper")
def test_sniper():
    """
    Endpoint temporal para validar la estrategia Gap Sniper con datos históricos de Kraken (via CCXT).
    """
    try:
        # 1. Conectar a Kraken (público)
        exchange = ccxt.kraken()
        symbol = 'BTC/USD'
        timeframe = '1h'
        limit = 500
        
        # 2. Descargar datos OHLCV
        # Estructura: [timestamp, open, high, low, close, volume]
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not ohlcv:
            return {"error": "No se pudieron descargar datos de Kraken"}

        # 3. Convertir a DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convertir timestamp a datetime para que sea legible (opcional, pero ayuda en debug)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Seteamos el índice o lo dejamos, GapSniper usa .iloc así que el índice no es crítico 
        # pero es bueno tenerlo
        df.set_index('datetime', inplace=True)

        results = []
        
        # 4. Instanciar GapSniperModule
        sniper = GapSniperModule()
        
        # 5. Iterar sobre las velas históricas
        # Necesitamos una ventana mínima mayor para ATR(14), VolMA(20) y ahora EMA(200) para SMC
        window_size = 205 
        
        for i in range(window_size, len(df)):
            # Slice simulation
            slice_data = df.iloc[i-window_size : i+1].copy()
            
            data_packet = {'market_data': slice_data}
            
            # Ejecutar análisis
            analysis = sniper.analyze(data_packet)
            
            if analysis['recommendation'] != 'neutral':
                # Guardar hallazgo
                # La fecha es el índice
                timestamp_str = str(slice_data.index[-1])
                
                results.append({
                    "timestamp": timestamp_str,
                    # El precio de cierre de la 'vela actual' del análisis (la última del slice)
                    "price": float(slice_data['close'].iloc[-1]),
                    "type": analysis['recommendation'], # long/short
                    "justification": analysis['justification']
                })
        
        return {
            "source": "Kraken",
            "symbol": symbol,
            "total_candles_analyzed": len(df),
            "gaps_detected_count": len(results),
            "gaps": results
        }
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main.main:app", host="0.0.0.0", port=port, reload=True)

