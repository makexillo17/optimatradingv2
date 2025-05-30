"""
Endpoints de la API REST
"""

from typing import Dict, List, Optional, Union
from fastapi import FastAPI, Depends, HTTPException, Security, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import redis
from datetime import datetime, timedelta

from .models import (
    Usuario,
    TokenAPI,
    WebhookConfig,
    ConectorTrading,
    ExportConfig,
    RateLimitConfig,
    ErrorAPI,
    EstadoSistema
)
from .auth import Auth
from .webhooks import WebhookManager
from .trading import GestorConectores
from .export import ExportManager
from ..logging import LoggerManager

# Crear aplicación FastAPI
app = FastAPI(
    title="Optimatrading API",
    description="API REST para la plataforma Optimatrading",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Inicializar componentes
logger_manager = LoggerManager()
auth = Auth("secret_key", logger_manager)  # TODO: Usar variable de entorno
webhook_manager = WebhookManager(logger_manager)
gestor_conectores = GestorConectores(logger_manager)
export_manager = ExportManager(logger_manager)

# Configurar rate limiting
redis_client = redis.Redis(host='localhost', port=6379, db=0)
rate_limit_config = RateLimitConfig()

def verificar_rate_limit(ip: str, usuario_id: Optional[str] = None) -> None:
    """Verificar límite de tasa"""
    ahora = datetime.now()
    
    # Verificar por IP
    if rate_limit_config.por_ip and ip not in rate_limit_config.excluir_ips:
        key = f"rate_limit:ip:{ip}"
        requests = redis_client.get(key)
        
        if requests and int(requests) >= rate_limit_config.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Límite de tasa excedido"
            )
            
        redis_client.incr(key)
        redis_client.expire(key, rate_limit_config.ventana_tiempo)
        
    # Verificar por usuario
    if (rate_limit_config.por_usuario and
        usuario_id and
        usuario_id not in rate_limit_config.excluir_usuarios):
        key = f"rate_limit:usuario:{usuario_id}"
        requests = redis_client.get(key)
        
        if requests and int(requests) >= rate_limit_config.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Límite de tasa excedido"
            )
            
        redis_client.incr(key)
        redis_client.expire(key, rate_limit_config.ventana_tiempo)

# Endpoints de autenticación
@app.post("/token", response_model=TokenAPI)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
) -> TokenAPI:
    """Obtener token de acceso"""
    # TODO: Implementar verificación de credenciales
    usuario = Usuario(
        email=form_data.username,
        nombre="Usuario",
        rol="trader"
    )
    
    return auth.crear_token(usuario)

# Endpoints de webhooks
@app.post("/webhooks", response_model=WebhookConfig)
async def crear_webhook(
    config: WebhookConfig,
    token: Dict = Depends(auth.verificar_token)
) -> WebhookConfig:
    """Crear nuevo webhook"""
    await webhook_manager.registrar_webhook(config)
    return config

@app.delete("/webhooks/{webhook_id}")
async def eliminar_webhook(
    webhook_id: str,
    token: Dict = Depends(auth.verificar_token)
) -> None:
    """Eliminar webhook"""
    await webhook_manager.eliminar_webhook(webhook_id)

@app.get("/webhooks/{webhook_id}/historial")
async def obtener_historial_webhook(
    webhook_id: str,
    limite: int = 100,
    token: Dict = Depends(auth.verificar_token)
) -> List[Dict]:
    """Obtener historial de envíos"""
    return webhook_manager.obtener_historial(webhook_id, limite)

# Endpoints de trading
@app.post("/conectores", response_model=ConectorTrading)
async def crear_conector(
    config: ConectorTrading,
    token: Dict = Depends(auth.verificar_token)
) -> ConectorTrading:
    """Crear nuevo conector de trading"""
    await gestor_conectores.crear_conector(config)
    return config

@app.delete("/conectores/{conector_id}")
async def eliminar_conector(
    conector_id: str,
    token: Dict = Depends(auth.verificar_token)
) -> None:
    """Eliminar conector"""
    await gestor_conectores.eliminar_conector(conector_id)

@app.post("/conectores/{conector_id}/ordenes")
async def crear_orden(
    conector_id: str,
    simbolo: str,
    tipo: str,
    lado: str,
    cantidad: float,
    precio: Optional[float] = None,
    token: Dict = Depends(auth.verificar_token)
) -> Dict:
    """Crear orden de trading"""
    conector = gestor_conectores.obtener_conector(conector_id)
    if not conector:
        raise HTTPException(
            status_code=404,
            detail="Conector no encontrado"
        )
        
    return await conector.crear_orden(
        simbolo,
        tipo,
        lado,
        cantidad,
        precio
    )

@app.delete("/conectores/{conector_id}/ordenes/{orden_id}")
async def cancelar_orden(
    conector_id: str,
    orden_id: str,
    token: Dict = Depends(auth.verificar_token)
) -> Dict:
    """Cancelar orden"""
    conector = gestor_conectores.obtener_conector(conector_id)
    if not conector:
        raise HTTPException(
            status_code=404,
            detail="Conector no encontrado"
        )
        
    return await conector.cancelar_orden(orden_id)

@app.get("/conectores/{conector_id}/posiciones")
async def obtener_posiciones(
    conector_id: str,
    token: Dict = Depends(auth.verificar_token)
) -> List[Dict]:
    """Obtener posiciones abiertas"""
    conector = gestor_conectores.obtener_conector(conector_id)
    if not conector:
        raise HTTPException(
            status_code=404,
            detail="Conector no encontrado"
        )
        
    return await conector.obtener_posiciones()

@app.get("/conectores/{conector_id}/balance")
async def obtener_balance(
    conector_id: str,
    token: Dict = Depends(auth.verificar_token)
) -> Dict[str, float]:
    """Obtener balance de la cuenta"""
    conector = gestor_conectores.obtener_conector(conector_id)
    if not conector:
        raise HTTPException(
            status_code=404,
            detail="Conector no encontrado"
        )
        
    return await conector.obtener_balance()

# Endpoints de exportación
@app.post("/exportar")
async def exportar_datos(
    datos: Union[Dict, List],
    config: ExportConfig,
    background_tasks: BackgroundTasks,
    token: Dict = Depends(auth.verificar_token)
) -> Dict[str, str]:
    """Exportar datos"""
    # Ejecutar exportación en segundo plano
    background_tasks.add_task(
        export_manager.exportar_datos,
        datos,
        config
    )
    
    return {
        "mensaje": "Exportación iniciada",
        "id": config.id
    }

# Endpoints de sistema
@app.get("/estado", response_model=EstadoSistema)
async def obtener_estado() -> EstadoSistema:
    """Obtener estado del sistema"""
    return EstadoSistema(
        version="1.0.0",
        estado="operativo",
        servicios={
            "api": True,
            "webhooks": True,
            "trading": True,
            "export": True
        },
        metricas={
            "usuarios_activos": 0,
            "webhooks_activos": len(webhook_manager.webhooks),
            "conectores_activos": len(gestor_conectores.conectores)
        }
    )

# Personalizar documentación OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="Optimatrading API",
        version="1.0.0",
        description="API REST para la plataforma Optimatrading",
        routes=app.routes
    )
    
    # Agregar componentes de seguridad
    openapi_schema["components"]["securitySchemes"] = {
        "OAuth2": {
            "type": "oauth2",
            "flows": {
                "password": {
                    "tokenUrl": "token",
                    "scopes": {}
                }
            }
        },
        "APIKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi 