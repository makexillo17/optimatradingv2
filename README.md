# Optimatrading

Sistema modular de análisis de trading que integra múltiples estrategias y fuentes de datos para proporcionar análisis y recomendaciones de trading.

## Estructura del Proyecto

```
optimatrading/
├── configuracion dinamica/    # Configuración del sistema
├── loader/                   # Módulo de recolección de datos
├── dispatcher/              # Distribuidor de datos
├── modulos/                # Módulos de análisis
├── main/                   # Integrador de resultados
├── logging/               # Sistema de logging
└── tests/                # Tests unitarios y de integración
```

## Módulos de Análisis

1. Broker Behavior
2. Carry Trade
3. Dynamic Hedging
4. Liquidity Provision
5. Market Making
6. Pairs Trading
7. SMC/ICT
8. Statistical Arbitrage
9. Momentum
10. Volatility

## Requisitos

- Python 3.8+
- Redis
- TA-Lib

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/optimatrading.git
cd optimatrading
```

2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar las APIs:
- Copiar `config.yaml.example` a `config.yaml`
- Agregar las API keys necesarias

## Uso

1. Iniciar Redis:
```bash
redis-server
```

2. Ejecutar el sistema:
```bash
python main.py
```

## Configuración

El sistema utiliza un archivo YAML para la configuración. Los principales parámetros son:

- API keys para diferentes proveedores de datos
- Configuración de logging
- Parámetros de caché
- Pesos de los módulos de análisis

## Desarrollo

Para contribuir al proyecto:

1. Crear un branch para tu feature
2. Implementar los cambios
3. Ejecutar los tests:
```bash
pytest
```
4. Crear un pull request

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.

## Despliegue en Render

El proyecto está configurado para desplegarse en Render usando:

- `render.yaml`: Configuración del servicio
- `Procfile`: Comando de arranque

### Comando de arranque
```bash
gunicorn -k uvicorn.workers.UvicornWorker -w 1 --threads 8 --timeout 60 optimatrading.api.routes:app --bind 0.0.0.0:$PORT
```

### Versiones principales
- Python 3.9.13
- FastAPI (última versión estable)
- Gunicorn con worker Uvicorn 