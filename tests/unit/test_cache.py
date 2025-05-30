import pytest
from unittest.mock import Mock, patch
import redis
import pickle
from cache.cache_manager import CacheManager

@pytest.fixture
def mock_redis():
    with patch('redis.Redis') as mock:
        yield mock

@pytest.fixture
def cache_manager(mock_redis):
    return CacheManager(host="localhost", port=6379, db=0)

def test_get_success(cache_manager, mock_redis):
    """Prueba obtener un valor exitosamente"""
    test_value = {"key": "value"}
    mock_redis.return_value.get.return_value = pickle.dumps(test_value)
    
    result = cache_manager.get("test_key")
    assert result == test_value
    mock_redis.return_value.get.assert_called_once_with("test_key")

def test_get_with_namespace(cache_manager, mock_redis):
    """Prueba obtener un valor con namespace"""
    test_value = {"key": "value"}
    mock_redis.return_value.get.return_value = pickle.dumps(test_value)
    
    result = cache_manager.get("test_key", namespace="test_ns")
    assert result == test_value
    mock_redis.return_value.get.assert_called_once_with("test_ns:test_key")

def test_get_missing_key(cache_manager, mock_redis):
    """Prueba obtener un valor que no existe"""
    mock_redis.return_value.get.return_value = None
    
    result = cache_manager.get("missing_key")
    assert result is None

def test_set_success(cache_manager, mock_redis):
    """Prueba almacenar un valor exitosamente"""
    test_value = {"key": "value"}
    mock_redis.return_value.set.return_value = True
    
    result = cache_manager.set("test_key", test_value)
    assert result is True
    
    mock_redis.return_value.set.assert_called_once()
    call_args = mock_redis.return_value.set.call_args
    assert call_args[0][0] == "test_key"
    assert pickle.loads(call_args[0][1]) == test_value

def test_set_with_ttl(cache_manager, mock_redis):
    """Prueba almacenar un valor con TTL"""
    test_value = {"key": "value"}
    mock_redis.return_value.setex.return_value = True
    
    result = cache_manager.set("test_key", test_value, ttl=60)
    assert result is True
    
    mock_redis.return_value.setex.assert_called_once()
    call_args = mock_redis.return_value.setex.call_args
    assert call_args[0][0] == "test_key"
    assert call_args[0][1] == 60
    assert pickle.loads(call_args[0][2]) == test_value

def test_delete_success(cache_manager, mock_redis):
    """Prueba eliminar un valor exitosamente"""
    mock_redis.return_value.delete.return_value = 1
    
    result = cache_manager.delete("test_key")
    assert result is True
    mock_redis.return_value.delete.assert_called_once_with("test_key")

def test_clear_namespace(cache_manager, mock_redis):
    """Prueba limpiar un namespace"""
    mock_redis.return_value.keys.return_value = ["ns:key1", "ns:key2"]
    mock_redis.return_value.delete.return_value = 2
    
    result = cache_manager.clear_namespace("ns")
    assert result == 2
    
    mock_redis.return_value.keys.assert_called_once_with("ns:*")
    mock_redis.return_value.delete.assert_called_once_with("ns:key1", "ns:key2")

def test_get_or_set_existing(cache_manager, mock_redis):
    """Prueba get_or_set con valor existente"""
    test_value = {"key": "value"}
    mock_redis.return_value.get.return_value = pickle.dumps(test_value)
    
    def value_func():
        return {"new": "value"}
    
    result = cache_manager.get_or_set("test_key", value_func)
    assert result == test_value
    mock_redis.return_value.set.assert_not_called()

def test_get_or_set_missing(cache_manager, mock_redis):
    """Prueba get_or_set con valor faltante"""
    mock_redis.return_value.get.return_value = None
    mock_redis.return_value.set.return_value = True
    
    test_value = {"new": "value"}
    def value_func():
        return test_value
    
    result = cache_manager.get_or_set("test_key", value_func)
    assert result == test_value
    
    mock_redis.return_value.set.assert_called_once()
    call_args = mock_redis.return_value.set.call_args
    assert pickle.loads(call_args[0][1]) == test_value

def test_mget_success(cache_manager, mock_redis):
    """Prueba obtener múltiples valores"""
    values = [
        pickle.dumps({"key1": "value1"}),
        pickle.dumps({"key2": "value2"})
    ]
    mock_redis.return_value.mget.return_value = values
    
    result = cache_manager.mget(["key1", "key2"])
    assert len(result) == 2
    assert result["key1"] == {"key1": "value1"}
    assert result["key2"] == {"key2": "value2"}

def test_mset_success(cache_manager, mock_redis):
    """Prueba almacenar múltiples valores"""
    test_values = {
        "key1": {"value": 1},
        "key2": {"value": 2}
    }
    mock_redis.return_value.mset.return_value = True
    
    result = cache_manager.mset(test_values)
    assert result is True
    
    mock_redis.return_value.mset.assert_called_once()
    call_args = mock_redis.return_value.mset.call_args[0][0]
    assert len(call_args) == 2
    assert pickle.loads(call_args["key1"]) == {"value": 1}
    assert pickle.loads(call_args["key2"]) == {"value": 2}

def test_get_stats(cache_manager, mock_redis):
    """Prueba obtener estadísticas"""
    mock_redis.return_value.info.return_value = {
        'used_memory': 1000,
        'keyspace_hits': 100,
        'keyspace_misses': 10,
        'db0': {
            'keys': 50,
            'expires': 20
        }
    }
    
    stats = cache_manager.get_stats()
    assert stats['used_memory'] == 1000
    assert stats['hits'] == 100
    assert stats['misses'] == 10
    assert stats['keys'] == 50
    assert stats['expires'] == 20 