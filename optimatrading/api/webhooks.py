"""
Sistema de webhooks para notificaciones
"""

from typing import Dict, List, Optional, Union
import asyncio
import aiohttp
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel

from .models import WebhookConfig
from ..logging import LoggerManager

class WebhookManager:
    """Gestor de webhooks"""
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Inicializar gestor de webhooks.
        
        Args:
            logger_manager: Gestor de logs
        """
        self.logger = logger_manager.get_logger("WebhookManager") if logger_manager else None
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.historial_envios: Dict[str, List[Dict]] = {}
        
    async def registrar_webhook(
        self,
        config: WebhookConfig
    ) -> None:
        """
        Registrar nuevo webhook.
        
        Args:
            config: Configuración del webhook
        """
        self.webhooks[config.id] = config
        self.historial_envios[config.id] = []
        
    async def eliminar_webhook(
        self,
        webhook_id: str
    ) -> None:
        """
        Eliminar webhook.
        
        Args:
            webhook_id: ID del webhook
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            del self.historial_envios[webhook_id]
            
    async def notificar_evento(
        self,
        evento: str,
        datos: Dict
    ) -> None:
        """
        Notificar evento a webhooks suscritos.
        
        Args:
            evento: Tipo de evento
            datos: Datos del evento
        """
        # Encontrar webhooks suscritos al evento
        webhooks_suscritos = [
            webhook for webhook in self.webhooks.values()
            if evento in webhook.eventos and webhook.activo
        ]
        
        # Enviar notificaciones en paralelo
        await asyncio.gather(*[
            self._enviar_notificacion(webhook, evento, datos)
            for webhook in webhooks_suscritos
        ])
        
    async def _enviar_notificacion(
        self,
        webhook: WebhookConfig,
        evento: str,
        datos: Dict
    ) -> None:
        """
        Enviar notificación a un webhook.
        
        Args:
            webhook: Configuración del webhook
            evento: Tipo de evento
            datos: Datos del evento
        """
        # Preparar payload
        payload = self._formatear_payload(webhook, evento, datos)
        
        # Calcular firma
        firma = self._calcular_firma(
            payload,
            webhook.secreto.get_secret_value()
        )
        
        # Preparar headers
        headers = {
            "Content-Type": "application/json",
            "X-Optimatrading-Signature": firma,
            "X-Optimatrading-Event": evento,
            **webhook.headers
        }
        
        # Intentar envío con reintentos
        for intento in range(webhook.max_reintentos + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        str(webhook.url),
                        json=payload,
                        headers=headers,
                        timeout=webhook.timeout
                    ) as response:
                        resultado = {
                            "timestamp": datetime.now(),
                            "evento": evento,
                            "status_code": response.status,
                            "intento": intento + 1
                        }
                        
                        if response.status in (200, 201, 202):
                            resultado["exito"] = True
                            self.historial_envios[webhook.id].append(resultado)
                            return
                            
                        resultado["exito"] = False
                        resultado["error"] = await response.text()
                        
                # Esperar antes de reintentar
                if intento < webhook.max_reintentos:
                    await asyncio.sleep(2 ** intento)  # Backoff exponencial
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        "error_envio_webhook",
                        webhook_id=webhook.id,
                        evento=evento,
                        error=str(e),
                        intento=intento + 1
                    )
                    
                if intento == webhook.max_reintentos:
                    self.historial_envios[webhook.id].append({
                        "timestamp": datetime.now(),
                        "evento": evento,
                        "exito": False,
                        "error": str(e),
                        "intento": intento + 1
                    })
                    
    def _formatear_payload(
        self,
        webhook: WebhookConfig,
        evento: str,
        datos: Dict
    ) -> Dict:
        """
        Formatear payload según configuración.
        
        Args:
            webhook: Configuración del webhook
            evento: Tipo de evento
            datos: Datos del evento
            
        Returns:
            Payload formateado
        """
        payload = {
            "id": str(hash(f"{evento}_{datetime.now()}")),
            "evento": evento,
            "timestamp": datetime.now().isoformat(),
            "datos": datos
        }
        
        # Aplicar formato personalizado
        if webhook.formato_payload:
            payload_formateado = {}
            for key, formato in webhook.formato_payload.items():
                if isinstance(formato, str):
                    if formato in datos:
                        payload_formateado[key] = datos[formato]
                elif isinstance(formato, list):
                    payload_formateado[key] = {
                        k: datos.get(k) for k in formato
                        if k in datos
                    }
            payload["datos"] = payload_formateado
            
        return payload
        
    def _calcular_firma(
        self,
        payload: Dict,
        secreto: str
    ) -> str:
        """
        Calcular firma HMAC del payload.
        
        Args:
            payload: Payload a firmar
            secreto: Secreto del webhook
            
        Returns:
            Firma HMAC
        """
        mensaje = json.dumps(payload, sort_keys=True)
        return hmac.new(
            secreto.encode(),
            mensaje.encode(),
            hashlib.sha256
        ).hexdigest()
        
    def obtener_historial(
        self,
        webhook_id: str,
        limite: int = 100
    ) -> List[Dict]:
        """
        Obtener historial de envíos.
        
        Args:
            webhook_id: ID del webhook
            limite: Límite de registros
            
        Returns:
            Historial de envíos
        """
        if webhook_id not in self.historial_envios:
            return []
            
        return sorted(
            self.historial_envios[webhook_id],
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limite]
        
    def limpiar_historial(
        self,
        dias: int = 30
    ) -> None:
        """
        Limpiar historial antiguo.
        
        Args:
            dias: Días a mantener
        """
        limite = datetime.now() - timedelta(days=dias)
        
        for webhook_id in self.historial_envios:
            self.historial_envios[webhook_id] = [
                envio for envio in self.historial_envios[webhook_id]
                if envio["timestamp"] > limite
            ] 