# Optimatrading - Sistema de Análisis de Trading

## Descripción General

Optimatrading es un sistema avanzado de análisis de trading que combina múltiples estrategias y módulos para generar recomendaciones de trading precisas y fundamentadas. El sistema utiliza una arquitectura modular y escalable, con componentes independientes que trabajan en conjunto para proporcionar un análisis integral del mercado.

## Arquitectura del Sistema

El sistema está compuesto por los siguientes componentes principales:

### 1. Módulos de Análisis

- **Broker Behavior Module**: Analiza el comportamiento de los brokers y market makers
- **Carry Trade Module**: Evalúa oportunidades de carry trade entre diferentes mercados
- **Dynamic Hedging Module**: Proporciona estrategias de cobertura dinámica
- **Liquidity Provision Module**: Analiza y aprovecha la provisión de liquidez
- **Market Making Module**: Implementa estrategias de market making
- **Pairs Trading Module**: Identifica y explota oportunidades de trading de pares
- **SMC/ICT Module**: Aplica conceptos de Smart Money Concepts e ICT
- **Statistical Arbitrage Module**: Busca oportunidades de arbitraje estadístico
- **Volatility Arbitrage Module**: Explota ineficiencias en la volatilidad
- **Yield Anomaly Module**: Identifica y aprovecha anomalías en rendimientos

### 2. Componentes de Soporte

#### Sistema de Logging
- Logging centralizado con structlog
- Métricas Prometheus integradas
- Múltiples handlers (archivo y consola)
- Logging estructurado con contexto

#### Sistema de Caché
- Caché basado en Redis
- Soporte para TTL
- Gestión de namespaces
- Operaciones atómicas
- Estadísticas de uso

#### Configuración Dinámica
- Configuración basada en YAML
- Actualizaciones en tiempo real
- Caché local y distribuido
- Validación de esquemas
- Backups automáticos

#### Métricas de Rendimiento
- Métricas de trading (Sharpe, drawdown)
- Rendimiento por módulo
- Métricas de confianza
- Seguimiento histórico
- Análisis basado en señales

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/optimatrading.git
cd optimatrading
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar Redis:
```bash
# Instalar Redis según tu sistema operativo
# Windows: https://github.com/microsoftarchive/redis/releases
# Linux: sudo apt-get install redis-server
# macOS: brew install redis
```

4. Configurar el sistema:
```bash
# Copiar configuración de ejemplo
cp config/config.example.yml config/config.yml
# Editar configuración según necesidades
```

## Uso

### Ejemplo Básico

```python
from main.main import OptimatradingMain
from logging.logger_manager import LoggerManager
from cache.cache_manager import CacheManager
from configuracion_dinamica.config_manager import ConfigManager
from utils.data_validator import DataValidator

# Inicializar componentes
logger_manager = LoggerManager(app_name="optimatrading")
cache_manager = CacheManager(host="localhost", port=6379)
config_manager = ConfigManager(config_dir="config")
data_validator = DataValidator(logger_manager=logger_manager)

# Inicializar sistema
system = OptimatradingMain(
    logger_manager=logger_manager,
    cache_manager=cache_manager,
    config_manager=config_manager,
    data_validator=data_validator
)

# Ejecutar análisis
result = system.run_analysis(
    asset_symbol="BTCUSDT",
    market_data=your_market_data
)

print(f"Recomendación: {result['recommendation']}")
print(f"Confianza: {result['confidence']}")
print(f"Justificación: {result['justification']}")
```

### Script de Ejemplo

Ver `examples/btcusdt_analysis.py` para un ejemplo completo de uso del sistema.

## Pruebas

El sistema incluye pruebas unitarias y de integración:

```bash
# Ejecutar todas las pruebas
pytest

# Ejecutar pruebas unitarias
pytest tests/unit/

# Ejecutar pruebas de integración
pytest tests/integration/
```

## Estructura de Directorios

```
optimatrading/
├── cache/                 # Sistema de caché
├── configuracion_dinamica/# Configuración dinámica
├── docs/                  # Documentación
├── examples/             # Scripts de ejemplo
├── logging/              # Sistema de logging
├── main/                 # Módulo principal
├── metricas_rendimiento/ # Métricas de rendimiento
├── modulos/              # Módulos de análisis
├── tests/                # Pruebas
│   ├── unit/            # Pruebas unitarias
│   └── integration/     # Pruebas de integración
└── utils/                # Utilidades generales
```

## Contribuir

1. Fork el repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo `LICENSE` para detalles.

## Contacto

Tu Nombre - [@tu_twitter](https://twitter.com/tu_twitter) - email@example.com

Project Link: [https://github.com/tu-usuario/optimatrading](https://github.com/tu-usuario/optimatrading) 