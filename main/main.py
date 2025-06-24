import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from fastapi import FastAPI                     #  ←  Import correcto, fuera de la clase
from loader.loader import MarketDataLoader as DataLoader
from dispatcher.dispatcher import ModuleDispatcher
from main.consensus import ConsensusAnalyzer
from utils.logger import setup_logger


class OptimatradingMain:
    def __init__(self):
        self.logger = setup_logger("OptimatradingMain")
        self.data_loader = DataLoader()
        self.dispatcher = ModuleDispatcher()
        self.consensus = ConsensusAnalyzer()

    # ----------------------------------------------------
    # 1) Método público
    # ----------------------------------------------------
    def run_analysis(self, asset_symbol: str) -> Dict[str, Any]:
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
                return self._generate_error_response("Error ejecutando módulos de análisis")

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
            market_data = self.data_loader.load_market_data(asset_symbol)

            # Extiende con datos específicos de cada módulo
            market_data.update(self.data_loader.load_broker_data(asset_symbol))
            market_data.update(self.data_loader.load_carry_data(asset_symbol))
            market_data.update(self.data_loader.load_options_data(asset_symbol))
            market_data.update(self.data_loader.load_liquidity_data(asset_symbol))
            market_data.update(self.data_loader.load_market_making_data(asset_symbol))
            market_data.update(self.data_loader.load_pairs_data(asset_symbol))
            market_data.update(self.data_loader.load_smc_data(asset_symbol))
            market_data.update(self.data_loader.load_stat_arb_data(asset_symbol))
            market_data.update(self.data_loader.load_volatility_data(asset_symbol))
            market_data.update(self.data_loader.load_yield_data(asset_symbol))

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
    return optimatrading.run_analysis(asset_symbol)

