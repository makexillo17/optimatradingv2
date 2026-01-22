import logging
import importlib
import traceback
from typing import Dict, Any, Optional
import pandas as pd

class ModuleDispatcher:
    def __init__(self, config_path: Optional[str] = None):
        """Inicializa el ModuleDispatcher con imports dinámicos."""
        self.logger = logging.getLogger('ModuleDispatcher')
        # Configuración básica de logging si no está configurado
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)
        
        # Cache de módulos cargados para evitar reimportar
        self._module_cache: Dict[str, Any] = {}
        
        # Mapeo de nombres de módulos a nombres de clases
        self._module_class_mapping = {
            'broker_behavior': 'BrokerBehaviorModule',
            'carry_trade': 'CarryTradeModule',
            'dynamic_hedging': 'DynamicHedgingModule',
            'liquidity_provision': 'LiquidityProvisionModule',
            'market_making': 'MarketMakingModule',
            'pairs_trading': 'PairsTradingModule',
            'smc_ict': 'SmcIctModule',
            'stat_arb': 'StatArbModule',
            'volatility_arb': 'VolatilityArbModule',
            'yield_anomaly': 'YieldAnomalyModule',
            'gap_sniper': 'GapSniperModule',
        }
    
    def _load_module(self, module_name: str):
        """
        Carga dinámicamente un módulo de análisis.
        
        Args:
            module_name: Nombre del módulo (ej: 'smc_ict')
            
        Returns:
            Instancia de la clase del módulo o None si falla
        """
        # Si ya está en cache, retornarlo
        if module_name in self._module_cache:
            return self._module_cache[module_name]
        
        try:
            # Importar el módulo dinámicamente
            module_path = f"modulos.{module_name}"
            module = importlib.import_module(module_path)
            
            # Obtener el nombre de la clase
            class_name = self._module_class_mapping.get(module_name)
            if not class_name:
                self.logger.error(f"No se encontró mapeo de clase para módulo: {module_name}")
                return None
            
            # Obtener la clase del módulo
            module_class = getattr(module, class_name, None)
            if not module_class:
                self.logger.error(f"No se encontró la clase {class_name} en el módulo {module_name}")
                return None
            
            # Instanciar la clase
            module_instance = module_class()
            
            # Guardar en cache
            self._module_cache[module_name] = module_instance
            
            self.logger.info(f"Módulo {module_name} cargado exitosamente")
            return module_instance
            
        except ImportError as e:
            self.logger.error(f"Error importando módulo {module_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
        except AttributeError as e:
            self.logger.error(f"Error accediendo a clase en módulo {module_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
        except Exception as e:
            self.logger.error(f"Error inesperado cargando módulo {module_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
    
    def run_module(self, module_name: str, market_data) -> Optional[Dict[str, Any]]:
        """
        Ejecuta un módulo de análisis específico con el DataFrame de market_data.
        
        Args:
            module_name: Nombre del módulo a ejecutar (ej: 'smc_ict')
            market_data: DataFrame de pandas con datos de mercado (OHLCV) o Dict con datos
            
        Returns:
            Diccionario con resultados del módulo o None si falla
        """
        try:
            self.logger.info(f"Ejecutando módulo: {module_name}")
            
            # Cargar el módulo dinámicamente
            module_instance = self._load_module(module_name)
            if not module_instance:
                self.logger.warning(f"No se pudo cargar el módulo {module_name}")
                return None
            
            # Preparar datos para el módulo
            # Convertir DataFrame a diccionario con estructura esperada
            if isinstance(market_data, pd.DataFrame):
                data_dict = {
                    'market_data': market_data,
                    'price_data': market_data[['close', 'open', 'high', 'low']].to_dict('records'),
                    'volume_data': market_data['volume'].tolist(),
                    'timestamp': market_data['timestamp'].tolist() if 'timestamp' in market_data.columns else [],
                }
            else:
                # Si es un dict, usarlo directamente pero asegurar estructura básica
                data_dict = market_data if isinstance(market_data, dict) else {'market_data': market_data}
            
            # Intentar llamar al método analyze() o run()
            result = None
            if hasattr(module_instance, 'analyze'):
                result = module_instance.analyze(data_dict)
            elif hasattr(module_instance, 'run'):
                result = module_instance.run(data_dict)
            else:
                self.logger.error(f"El módulo {module_name} no tiene método 'analyze' ni 'run'")
                return None
            
            if result:
                self.logger.info(f"Módulo {module_name} ejecutado exitosamente")
                return result
            else:
                self.logger.warning(f"Módulo {module_name} no devolvió resultados")
                return None
                
        except Exception as e:
            # Error individual del módulo - NO rompe el programa
            self.logger.error(f"Error ejecutando módulo {module_name}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            # Retornar None para que los otros módulos puedan seguir funcionando
            return None
