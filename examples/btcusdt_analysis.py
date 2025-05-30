"""
Script de ejemplo que muestra el uso del sistema Optimatrading
analizando el par BTCUSDT.
"""

# Librerías estándar
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

# Librerías externas
import pandas as pd
import ccxt

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Módulos locales
from optimatrading.main import OptimatradingMain
from optimatrading.logging import LoggerManager
from optimatrading.cache import CacheManager
from optimatrading.config import ConfigManager
from optimatrading.utils import DataValidator

def fetch_market_data(
    symbol: str,
    timeframe: str = '1h',
    limit: int = 100,
    config: ConfigManager = None
) -> pd.DataFrame:
    """
    Obtiene datos de mercado de Binance
    
    Args:
        symbol: Par de trading (e.g. 'BTCUSDT')
        timeframe: Intervalo temporal
        limit: Cantidad de velas
        config: Instancia de ConfigManager
        
    Returns:
        DataFrame con datos OHLCV
    """
    try:
        # Obtener configuración de Binance
        binance_config = config.get('apis.binance', required=True)
        
        # Inicializar exchange
        exchange = ccxt.binance({
            'apiKey': binance_config['api_key'],
            'secret': binance_config['api_secret'],
            'enableRateLimit': binance_config.get('rate_limit', True),
            'timeout': binance_config.get('timeout', 30000),
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'testnet': binance_config.get('testnet', False)
            }
        })
        
        # Obtener datos OHLCV
        ohlcv = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            limit=limit
        )
        
        # Convertir a DataFrame
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Error obteniendo datos de mercado: {str(e)}")

def main():
    """Función principal que ejecuta el análisis completo"""
    try:
        # Inicializar gestor de configuración
        config_manager = ConfigManager(
            config_dir="config",
            config_file="config.yaml",
            auto_reload=True,
            validate_schema=True
        )
        
        # Obtener configuración de logging
        log_config = config_manager.get('logging', required=True)
        
        # Inicializar componentes
        logger_manager = LoggerManager(
            app_name="optimatrading",
            log_dir=log_config['file']['directory'],
            console_level=log_config['console']['level'],
            file_level=log_config['file']['level']
        )
        logger = logger_manager.get_logger("example")
        
        # Obtener configuración de Redis
        redis_config = config_manager.get('redis', required=True)
        
        cache_manager = CacheManager(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            logger_manager=logger_manager,
            default_ttl=redis_config.get('default_ttl', 3600),
            socket_timeout=redis_config.get('socket_timeout', 5),
            socket_connect_timeout=redis_config.get('socket_connect_timeout', 5),
            retry_on_timeout=redis_config.get('retry_on_timeout', True)
        )
        
        data_validator = DataValidator(logger_manager=logger_manager)
        
        # Inicializar sistema principal
        system = OptimatradingMain(
            logger_manager=logger_manager,
            cache_manager=cache_manager,
            config_manager=config_manager,
            data_validator=data_validator
        )
        
        logger.info("Sistema inicializado correctamente")
        
        # Obtener datos de mercado
        symbol = "BTCUSDT"
        market_data = fetch_market_data(
            symbol,
            config=config_manager
        )
        logger.info(f"Datos obtenidos para {symbol}: {len(market_data)} registros")
        
        # Validar datos
        validation_config = config_manager.get('validation', required=True)
        errors = data_validator.validate_market_data(
            market_data.to_dict('records')[0],
            required_fields=validation_config['required_fields'],
            max_missing_values=validation_config.get('max_missing_values', 0.1)
        )
        
        if errors:
            logger.error(f"Errores en los datos: {errors}")
            return
            
        # Ejecutar análisis
        result = system.run_analysis(
            asset_symbol=symbol,
            market_data=market_data
        )
        
        # Mostrar resultados
        print("\n=== Resultados del Análisis ===")
        print(f"Recomendación: {result['recommendation']}")
        print(f"Confianza: {result['confidence']:.2%}")
        print(f"Justificación: {result['justification']}")
        print("\nResultados por Módulo:")
        
        for module, module_result in result['module_results'].items():
            print(f"\n{module}:")
            print(f"  Recomendación: {module_result['recommendation']}")
            print(f"  Confianza: {module_result['confidence']:.2%}")
            print(f"  Justificación: {module_result['justification']}")
            
        # Calcular métricas históricas si están habilitadas
        metrics_config = config_manager.get('metrics')
        if metrics_config and metrics_config.get('enabled', False):
            historical_results = cache_manager.get(
                f"historical_results:{symbol}",
                namespace="analysis"
            ) or []
            
            if historical_results:
                timestamps = [r['timestamp'] for r in historical_results]
                returns = [r['actual_return'] for r in historical_results]
                
                metrics = system.calculate_performance_metrics(
                    historical_results,
                    returns,
                    timestamps
                )
                
                print("\n=== Métricas de Rendimiento ===")
                for metric, enabled in metrics_config['calculate'].items():
                    if enabled and metric in metrics:
                        print(f"{metric.replace('_', ' ').title()}: {metrics[metric]:.2%}")
            
    except Exception as e:
        logger.error(f"Error en la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main() 