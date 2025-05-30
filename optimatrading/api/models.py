"""
Modelos de datos para la API REST
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr, SecretStr, HttpUrl, validator
import uuid

class Usuario(BaseModel):
    """Modelo de usuario"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    nombre: str
    api_key: Optional[str] = None
    activo: bool = True
    rol: str = Field(regex="^(admin|trader|viewer)$")
    fecha_creacion: datetime = Field(default_factory=datetime.now)
    configuracion: Dict[str, Union[str, int, float, bool]] = {}

class TokenAPI(BaseModel):
    """Token de acceso a la API"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class WebhookConfig(BaseModel):
    """Configuración de webhook"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nombre: str
    url: HttpUrl
    eventos: List[str]
    headers: Dict[str, str] = {}
    formato_payload: Dict[str, Union[str, List[str]]] = {}
    activo: bool = True
    secreto: SecretStr
    max_reintentos: int = Field(ge=0, le=10, default=3)
    timeout: int = Field(ge=1, le=30, default=5)
    
    @validator("eventos")
    def validar_eventos(cls, v):
        eventos_validos = {
            "trade_executed", "order_created", "order_updated",
            "position_opened", "position_closed", "alert_triggered",
            "system_error", "performance_update"
        }
        if not all(e in eventos_validos for e in v):
            raise ValueError(f"Eventos inválidos. Permitidos: {eventos_validos}")
        return v

class ConectorTrading(BaseModel):
    """Configuración de conector de trading"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nombre: str
    plataforma: str = Field(regex="^(binance|ftx|kraken|interactive_brokers)$")
    credenciales: Dict[str, SecretStr]
    configuracion: Dict[str, Union[str, int, float, bool]] = {}
    modo_simulacion: bool = False
    limites: Dict[str, float] = {
        "operacion_maxima": 0.0,
        "posicion_maxima": 0.0,
        "perdida_maxima": 0.0
    }
    instrumentos_permitidos: List[str] = []

class ExportConfig(BaseModel):
    """Configuración de exportación"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nombre: str
    tipo: str = Field(regex="^(csv|json|excel)$")
    programacion: Optional[str] = Field(regex="^[0-9*/,-]+\\s[0-9*/,-]+\\s[0-9*/,-]+\\s[0-9*/,-]+\\s[0-9*/,-]+$")
    filtros: Dict[str, Union[str, List[str], Dict[str, Union[str, float]]]] = {}
    formato: Dict[str, Union[str, List[str]]] = {}
    destino: Dict[str, str] = {
        "tipo": "local",  # local, s3, gcs, azure
        "ruta": ""
    }
    compresion: bool = False
    encriptacion: bool = False

class RateLimitConfig(BaseModel):
    """Configuración de rate limiting"""
    ventana_tiempo: int = Field(ge=1, le=3600, default=60)  # segundos
    max_requests: int = Field(ge=1, le=1000, default=100)
    por_ip: bool = True
    por_usuario: bool = True
    excluir_ips: List[str] = []
    excluir_usuarios: List[str] = []

class ErrorAPI(BaseModel):
    """Modelo de error de API"""
    codigo: str
    mensaje: str
    detalles: Optional[Dict[str, Union[str, int, float, bool]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class EstadoSistema(BaseModel):
    """Estado del sistema"""
    version: str
    estado: str = Field(regex="^(operativo|mantenimiento|error)$")
    servicios: Dict[str, bool]
    metricas: Dict[str, Union[int, float]]
    ultimo_check: datetime = Field(default_factory=datetime.now) 