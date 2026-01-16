import logging
import traceback
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI                     #  ←  Import correcto, fuera de la clase
from loader.loader import MarketDataLoader as DataLoader
from dispatcher.dispatcher import ModuleDispatcher
from main.consensus import ConsensusAnalyzer
from utils.logger import setup_logger

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

    # ----------------------------------------------------
    # 1) Método público
    # ----------------------------------------------------
    def run_analysis(self, asset_symbol: str, current_price: Optional[float] = None) -> Dict[str, Any]:
        """Ejecuta el pipeline completo de análisis para un activo."""
        try:
            self.logger.info(f"Iniciando análisis para {asset_symbol}")

            # 1. Cargar datos de mercado
            market_data = self._load_market_data(asset_symbol)
            if not market_data:
                return self._generate_error_response("Error cargando datos de mercado")

            # 2. Ejecutar módulos analíticos
            module_results = self._run_analysis_modules(market_data)
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
    def _run_analysis_modules(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Ejecuta todos los módulos de análisis."""
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
            ]

            for module_name in modules:
                result = self.dispatcher.run_module(module_name, market_data)
                if result:
                    module_results[module_name] = result
                else:
                    self.logger.warning(f"Módulo {module_name} no generó resultados")

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
    
    # Capturar el precio actual del DataFrame
    current_price = None
    try:
        loader = optimatrading.data_loader
        df = loader.load_broker_data(asset_symbol, timeframe='1h', limit=100)
        
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
        return optimatrading.run_analysis(asset_symbol, current_price=current_price)
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


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main.main:app", host="0.0.0.0", port=port, reload=True)

