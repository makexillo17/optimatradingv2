import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any

from main.main import OptimatradingMain
from logging.logger_manager import LoggerManager
from cache.cache_manager import CacheManager
from configuracion_dinamica.config_manager import ConfigManager
from utils.data_validator import DataValidator

class TestSystemIntegration:
    @pytest.fixture
    def setup_system(self):
        """Configura el sistema completo para pruebas"""
        logger_manager = LoggerManager(app_name="test_optimatrading")
        cache_manager = CacheManager(
            host="localhost",
            port=6379,
            db=1,  # Base de datos separada para pruebas
            logger_manager=logger_manager
        )
        config_manager = ConfigManager(
            config_dir="test_config",
            logger_manager=logger_manager,
            cache_manager=cache_manager
        )
        data_validator = DataValidator(logger_manager=logger_manager)
        
        system = OptimatradingMain(
            logger_manager=logger_manager,
            cache_manager=cache_manager,
            config_manager=config_manager,
            data_validator=data_validator
        )
        
        return {
            'system': system,
            'logger': logger_manager,
            'cache': cache_manager,
            'config': config_manager,
            'validator': data_validator
        }
        
    def test_full_analysis_flow(self, setup_system):
        """Prueba el flujo completo de análisis"""
        system = setup_system['system']
        
        # Generar datos de prueba
        test_data = self._generate_test_data()
        
        # Ejecutar análisis
        result = system.run_analysis(
            asset_symbol="BTCUSDT",
            market_data=test_data
        )
        
        # Validar estructura del resultado
        assert isinstance(result, dict)
        assert 'recommendation' in result
        assert 'confidence' in result
        assert 'justification' in result
        assert 'module_results' in result
        
        # Validar recomendación
        assert result['recommendation'] in ['LONG', 'SHORT', 'NEUTRAL']
        assert 0 <= result['confidence'] <= 1
        assert isinstance(result['justification'], str)
        
        # Validar resultados de módulos
        assert isinstance(result['module_results'], dict)
        for module_result in result['module_results'].values():
            assert 'recommendation' in module_result
            assert 'confidence' in module_result
            assert 'justification' in module_result
            
    def test_error_handling(self, setup_system):
        """Prueba el manejo de errores"""
        system = setup_system['system']
        
        # Datos inválidos
        invalid_data = pd.DataFrame({
            'timestamp': [],
            'close': []
        })
        
        with pytest.raises(ValueError) as exc_info:
            system.run_analysis(
                asset_symbol="BTCUSDT",
                market_data=invalid_data
            )
        assert "insufficient_data" in str(exc_info.value)
        
    def test_caching_integration(self, setup_system):
        """Prueba la integración con el sistema de caché"""
        system = setup_system['system']
        cache = setup_system['cache']
        
        test_data = self._generate_test_data()
        
        # Primera ejecución
        result1 = system.run_analysis(
            asset_symbol="BTCUSDT",
            market_data=test_data
        )
        
        # Verificar que se almacenó en caché
        cached_result = cache.get(
            "analysis:BTCUSDT",
            namespace="results"
        )
        assert cached_result is not None
        assert cached_result['recommendation'] == result1['recommendation']
        
        # Segunda ejecución (debería usar caché)
        result2 = system.run_analysis(
            asset_symbol="BTCUSDT",
            market_data=test_data
        )
        assert result2 == result1
        
    def test_config_integration(self, setup_system):
        """Prueba la integración con la configuración dinámica"""
        system = setup_system['system']
        config = setup_system['config']
        
        # Actualizar configuración
        config.update_config(
            "analysis",
            {
                "confidence_threshold": 0.8,
                "cache_ttl": 3600
            }
        )
        
        test_data = self._generate_test_data()
        result = system.run_analysis(
            asset_symbol="BTCUSDT",
            market_data=test_data
        )
        
        # Verificar que se aplicó la configuración
        assert result['confidence'] >= 0.8
        
    def test_validation_integration(self, setup_system):
        """Prueba la integración con el validador de datos"""
        system = setup_system['system']
        validator = setup_system['validator']
        
        # Datos con valores faltantes
        invalid_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=10),
            'close': [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        })
        
        # Validar datos
        errors = validator.validate_time_series(
            invalid_data,
            required_fields=['timestamp', 'close']
        )
        
        assert len(errors) > 0
        assert 'missing_values' in errors
        
    def test_metrics_integration(self, setup_system):
        """Prueba la integración con el sistema de métricas"""
        system = setup_system['system']
        
        test_data = self._generate_test_data()
        
        # Ejecutar múltiples análisis
        results = []
        timestamps = []
        returns = []
        
        for i in range(5):
            shifted_data = test_data.shift(i)
            result = system.run_analysis(
                asset_symbol="BTCUSDT",
                market_data=shifted_data
            )
            results.append(result)
            timestamps.append(datetime.now() - timedelta(days=i))
            returns.append(0.01 * (i - 2))  # Retornos simulados
            
        # Calcular métricas
        metrics = system.calculate_performance_metrics(
            results,
            returns,
            timestamps
        )
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'win_rate' in metrics
        
    def _generate_test_data(self) -> pd.DataFrame:
        """Genera datos de prueba"""
        dates = pd.date_range(start='2024-01-01', periods=100)
        
        # Simular un movimiento de precios realista
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, size=100)
        price = 100 * np.exp(np.cumsum(returns))
        
        # Generar OHLCV
        data = pd.DataFrame({
            'timestamp': dates,
            'open': price * (1 + np.random.normal(0, 0.002, size=100)),
            'high': price * (1 + abs(np.random.normal(0, 0.004, size=100))),
            'low': price * (1 - abs(np.random.normal(0, 0.004, size=100))),
            'close': price,
            'volume': np.random.lognormal(10, 1, size=100)
        })
        
        # Asegurar que high >= open >= low
        data['high'] = data[['high', 'open', 'close']].max(axis=1)
        data['low'] = data[['low', 'open', 'close']].min(axis=1)
        
        return data 