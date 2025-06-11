import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from loader.loader import MarketDataLoader as DataLoader
from dispatcher.module_dispatcher import ModuleDispatcher
from main.consensus import ConsensusAnalyzer
from utils.logger import setup_logger

class OptimatradingMain:
    def __init__(self):
        self.logger = setup_logger("OptimatradingMain")
        self.data_loader = DataLoader()
        self.dispatcher = ModuleDispatcher()
        self.consensus = ConsensusAnalyzer()
        
    def run_analysis(self, asset_symbol: str) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de análisis para un activo
        
        Args:
            asset_symbol: Símbolo del activo a analizar
            
        Returns:
            Dict con recomendación final, confianza y justificación
        """
        try:
            self.logger.info(f"Iniciando análisis para {asset_symbol}")
            
            # Cargar datos
            market_data = self._load_market_data(asset_symbol)
            if not market_data:
                return self._generate_error_response("Error cargando datos de mercado")
                
            # Ejecutar módulos de análisis
            module_results = self._run_analysis_modules(market_data)
            if not module_results:
                return self._generate_error_response("Error ejecutando módulos de análisis")
                
            # Generar consenso
            consensus_result = self._generate_consensus(module_results)
            
            # Formatear resultado final
            final_result = self._format_final_result(
                consensus_result,
                module_results,
                asset_symbol
            )
            
            self.logger.info(f"Análisis completado para {asset_symbol}")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error en análisis: {str(e)}")
            return self._generate_error_response(f"Error inesperado: {str(e)}")
            
    def _load_market_data(self, asset_symbol: str) -> Optional[Dict[str, Any]]:
        """Carga todos los datos necesarios para el análisis"""
        try:
            # Cargar datos base
            market_data = self.data_loader.load_market_data(asset_symbol)
            
            # Cargar datos específicos para cada módulo
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
            
    def _run_analysis_modules(
        self,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Ejecuta todos los módulos de análisis"""
        try:
            # Ejecutar cada módulo
            module_results = {}
            
            modules = [
                'broker_behavior',
                'carry_trade',
                'dynamic_hedging',
                'liquidity_provision',
                'market_making',
                'pairs_trading',
                'smc_ict',
                'stat_arb',
                'volatility_arb',
                'yield_anomaly'
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
            
    def _generate_consensus(
        self,
        module_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Genera el consenso final basado en resultados de módulos"""
        try:
            return self.consensus.analyze(module_results)
        except Exception as e:
            self.logger.error(f"Error generando consenso: {str(e)}")
            return {
                'recommendation': 'neutral',
                'confidence': 0.0,
                'justification': f"Error generando consenso: {str(e)}"
            }
            
    def _format_final_result(
        self,
        consensus_result: Dict[str, Any],
        module_results: Dict[str, Any],
        asset_symbol: str
    ) -> Dict[str, Any]:
        """Formatea el resultado final del análisis"""
        return {
            'timestamp': datetime.now().isoformat(),
            'asset_symbol': asset_symbol,
            'recommendation': consensus_result['recommendation'],
            'confidence': consensus_result['confidence'],
            'justification': consensus_result['justification'],
            'module_results': module_results,
            'consensus_details': consensus_result.get('details', {})
        }
        
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Genera respuesta de error estándar"""
        return {
            'timestamp': datetime.now().isoformat(),
            'recommendation': 'neutral',
            'confidence': 0.0,
            'justification': error_message,
            'module_results': {},
            'consensus_details': {},
            'error': error_message
        } 
    from fastapi import FastAPI

app = FastAPI()

# Instanciar tu clase
optimatrading = OptimatradingMain()

@app.get("/")
def root():
    return {"message": "Servidor de Optimatrading activo"}

# Ejemplo de endpoint usando tu clase
@app.get("/analyze/{asset_symbol}")
def analyze(asset_symbol: str):
    result = optimatrading.run_analysis(asset_symbol)
    return result
