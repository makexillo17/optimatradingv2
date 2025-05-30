import pytest
from unittest.mock import Mock, patch
import os
from logging.logger_manager import LoggerManager

@pytest.fixture
def logger_manager():
    return LoggerManager(app_name="test_app")

def test_logger_initialization(logger_manager):
    """Prueba la inicialización del logger"""
    assert logger_manager.app_name == "test_app"
    assert os.path.exists(logger_manager.log_dir)
    assert logger_manager.metrics is not None

def test_get_logger(logger_manager):
    """Prueba obtener un logger para un módulo"""
    logger = logger_manager.get_logger("test_module")
    assert logger is not None

@patch('structlog.get_logger')
def test_log_error(mock_get_logger, logger_manager):
    """Prueba el registro de errores"""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    error = ValueError("Test error")
    context = {"additional": "info"}
    
    logger_manager.log_error("test_module", error, context)
    
    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[1]
    assert call_args["error_type"] == "ValueError"
    assert call_args["module"] == "test_module"
    assert call_args["additional"] == "info"

@patch('structlog.get_logger')
def test_log_metric(mock_get_logger, logger_manager):
    """Prueba el registro de métricas"""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    logger_manager.log_metric(
        "test_module",
        "test_operation",
        1.5,
        {"test": "metadata"}
    )
    
    # Verificar que se registró la métrica
    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[1]
    assert call_args["operation"] == "test_operation"
    assert call_args["duration"] == 1.5
    assert call_args["test"] == "metadata"

@patch('structlog.get_logger')
def test_log_validation_error(mock_get_logger, logger_manager):
    """Prueba el registro de errores de validación"""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    validation_errors = {
        "field1": ["Error 1", "Error 2"],
        "field2": ["Error 3"]
    }
    
    logger_manager.log_validation_error("test_module", validation_errors)
    
    mock_logger.warning.assert_called_once()
    call_args = mock_logger.warning.call_args[1]
    assert call_args["module"] == "test_module"
    assert call_args["errors"] == validation_errors

@patch('structlog.get_logger')
def test_log_api_request(mock_get_logger, logger_manager):
    """Prueba el registro de llamadas a API"""
    mock_logger = Mock()
    mock_get_logger.return_value = mock_logger
    
    logger_manager.log_api_request(
        module="test_module",
        api_name="test_api",
        endpoint="/test",
        response_time=0.5,
        status_code=200,
        success=True
    )
    
    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[1]
    assert call_args["module"] == "test_module"
    assert call_args["api_name"] == "test_api"
    assert call_args["endpoint"] == "/test"
    assert call_args["response_time"] == 0.5
    assert call_args["status_code"] == 200
    assert call_args["success"] is True

def test_metrics_initialization(logger_manager):
    """Prueba la inicialización de métricas"""
    metrics = logger_manager.metrics
    
    assert 'log_entries' in metrics
    assert 'processing_time' in metrics
    
    # Verificar contadores
    assert metrics['log_entries']._type == "counter"
    assert metrics['processing_time']._type == "histogram" 