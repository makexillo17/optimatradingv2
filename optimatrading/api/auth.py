"""
Sistema de autenticación y autorización
"""

from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from passlib.context import CryptContext
from pydantic import BaseModel

from .models import Usuario, TokenAPI
from ..logging import LoggerManager

class Auth:
    """Gestor de autenticación y autorización"""
    
    def __init__(
        self,
        secret_key: str,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Inicializar gestor de autenticación.
        
        Args:
            secret_key: Clave secreta para JWT
            logger_manager: Gestor de logs
        """
        self.secret_key = secret_key
        self.logger = logger_manager.get_logger("Auth") if logger_manager else None
        
        # Configuración de seguridad
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
        self.api_key_header = APIKeyHeader(name="X-API-Key")
        
        # Cache de tokens revocados
        self.revoked_tokens: Dict[str, datetime] = {}
        
    def verificar_password(self, password: str, hashed_password: str) -> bool:
        """Verificar contraseña"""
        return self.pwd_context.verify(password, hashed_password)
        
    def get_password_hash(self, password: str) -> str:
        """Generar hash de contraseña"""
        return self.pwd_context.hash(password)
        
    def crear_token(
        self,
        usuario: Usuario,
        expires_delta: Optional[timedelta] = None
    ) -> TokenAPI:
        """
        Crear token de acceso.
        
        Args:
            usuario: Usuario
            expires_delta: Tiempo de expiración
            
        Returns:
            Token de acceso
        """
        try:
            if expires_delta is None:
                expires_delta = timedelta(minutes=30)
                
            expire = datetime.utcnow() + expires_delta
            
            # Crear payload
            payload = {
                "sub": usuario.email,
                "id": usuario.id,
                "rol": usuario.rol,
                "exp": expire
            }
            
            # Generar tokens
            access_token = jwt.encode(
                payload,
                self.secret_key,
                algorithm="HS256"
            )
            
            refresh_token = jwt.encode(
                {**payload, "tipo": "refresh"},
                self.secret_key,
                algorithm="HS256"
            )
            
            return TokenAPI(
                access_token=access_token,
                expires_in=expires_delta.seconds,
                refresh_token=refresh_token
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_crear_token",
                    error=str(e)
                )
            raise HTTPException(
                status_code=500,
                detail="Error al crear token"
            )
            
    def verificar_token(
        self,
        token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))
    ) -> Dict:
        """
        Verificar token JWT.
        
        Args:
            token: Token JWT
            
        Returns:
            Payload del token
        """
        try:
            # Verificar si el token está revocado
            if token in self.revoked_tokens:
                raise HTTPException(
                    status_code=401,
                    detail="Token revocado"
                )
                
            # Decodificar token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token expirado"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=401,
                detail="Token inválido"
            )
            
    def verificar_api_key(
        self,
        api_key: str = Security(APIKeyHeader(name="X-API-Key"))
    ) -> Usuario:
        """
        Verificar API key.
        
        Args:
            api_key: API key
            
        Returns:
            Usuario asociado a la API key
        """
        # TODO: Implementar búsqueda de usuario por API key
        raise HTTPException(
            status_code=401,
            detail="API key inválida"
        )
        
    def verificar_permisos(
        self,
        token: Dict = Depends(verificar_token),
        roles_requeridos: List[str] = []
    ) -> bool:
        """
        Verificar permisos de usuario.
        
        Args:
            token: Token decodificado
            roles_requeridos: Roles requeridos
            
        Returns:
            True si tiene permisos
        """
        if not roles_requeridos:
            return True
            
        return token.get("rol") in roles_requeridos
        
    def revocar_token(self, token: str) -> None:
        """
        Revocar token.
        
        Args:
            token: Token a revocar
        """
        try:
            # Decodificar token para obtener expiración
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=["HS256"]
            )
            
            # Almacenar en cache de revocados
            self.revoked_tokens[token] = datetime.fromtimestamp(
                payload["exp"]
            )
            
            # Limpiar tokens expirados
            self._limpiar_tokens_revocados()
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_revocar_token",
                    error=str(e)
                )
            
    def _limpiar_tokens_revocados(self) -> None:
        """Limpiar tokens revocados expirados"""
        ahora = datetime.utcnow()
        self.revoked_tokens = {
            token: exp
            for token, exp in self.revoked_tokens.items()
            if exp > ahora
        } 