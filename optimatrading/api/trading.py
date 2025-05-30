"""
Conectores para plataformas de trading
"""

from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import ccxt
import asyncio
from datetime import datetime
from decimal import Decimal

from .models import ConectorTrading
from ..logging import LoggerManager

class ConectorBase(ABC):
    """Clase base para conectores de trading"""
    
    def __init__(
        self,
        config: ConectorTrading,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Inicializar conector.
        
        Args:
            config: Configuración del conector
            logger_manager: Gestor de logs
        """
        self.config = config
        self.logger = logger_manager.get_logger(f"Conector{config.plataforma}") if logger_manager else None
        
    @abstractmethod
    async def conectar(self) -> None:
        """Establecer conexión con la plataforma"""
        pass
        
    @abstractmethod
    async def desconectar(self) -> None:
        """Cerrar conexión con la plataforma"""
        pass
        
    @abstractmethod
    async def crear_orden(
        self,
        simbolo: str,
        tipo: str,
        lado: str,
        cantidad: float,
        precio: Optional[float] = None
    ) -> Dict:
        """Crear orden de trading"""
        pass
        
    @abstractmethod
    async def cancelar_orden(
        self,
        orden_id: str
    ) -> Dict:
        """Cancelar orden"""
        pass
        
    @abstractmethod
    async def obtener_posiciones(self) -> List[Dict]:
        """Obtener posiciones abiertas"""
        pass
        
    @abstractmethod
    async def obtener_balance(self) -> Dict[str, float]:
        """Obtener balance de la cuenta"""
        pass

class ConectorCCXT(ConectorBase):
    """Conector genérico usando CCXT"""
    
    async def conectar(self) -> None:
        """Establecer conexión con la plataforma"""
        try:
            # Crear instancia de exchange
            exchange_class = getattr(ccxt, self.config.plataforma)
            self.exchange = exchange_class({
                'apiKey': self.config.credenciales['api_key'].get_secret_value(),
                'secret': self.config.credenciales['api_secret'].get_secret_value(),
                'enableRateLimit': True,
                'options': self.config.configuracion
            })
            
            # Cargar mercados
            await self.exchange.load_markets()
            
            if self.logger:
                self.logger.info(
                    "conexion_establecida",
                    plataforma=self.config.plataforma
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_conexion",
                    plataforma=self.config.plataforma,
                    error=str(e)
                )
            raise
            
    async def desconectar(self) -> None:
        """Cerrar conexión con la plataforma"""
        self.exchange = None
        
    async def crear_orden(
        self,
        simbolo: str,
        tipo: str,
        lado: str,
        cantidad: float,
        precio: Optional[float] = None
    ) -> Dict:
        """
        Crear orden de trading.
        
        Args:
            simbolo: Par de trading
            tipo: Tipo de orden (market, limit)
            lado: Dirección (buy, sell)
            cantidad: Cantidad a operar
            precio: Precio límite (opcional)
            
        Returns:
            Información de la orden
        """
        try:
            # Validar límites
            if not self._validar_limites(simbolo, cantidad, precio):
                raise ValueError("Orden excede límites configurados")
                
            # Crear orden
            params = {}
            if self.config.modo_simulacion:
                params['test'] = True
                
            orden = await self.exchange.create_order(
                simbolo,
                tipo,
                lado,
                cantidad,
                precio,
                params
            )
            
            if self.logger:
                self.logger.info(
                    "orden_creada",
                    orden_id=orden['id'],
                    simbolo=simbolo,
                    tipo=tipo,
                    lado=lado
                )
                
            return orden
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_crear_orden",
                    simbolo=simbolo,
                    tipo=tipo,
                    lado=lado,
                    error=str(e)
                )
            raise
            
    async def cancelar_orden(
        self,
        orden_id: str
    ) -> Dict:
        """
        Cancelar orden.
        
        Args:
            orden_id: ID de la orden
            
        Returns:
            Información de la cancelación
        """
        try:
            resultado = await self.exchange.cancel_order(orden_id)
            
            if self.logger:
                self.logger.info(
                    "orden_cancelada",
                    orden_id=orden_id
                )
                
            return resultado
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_cancelar_orden",
                    orden_id=orden_id,
                    error=str(e)
                )
            raise
            
    async def obtener_posiciones(self) -> List[Dict]:
        """
        Obtener posiciones abiertas.
        
        Returns:
            Lista de posiciones
        """
        try:
            posiciones = await self.exchange.fetch_positions()
            
            # Filtrar por instrumentos permitidos
            if self.config.instrumentos_permitidos:
                posiciones = [
                    p for p in posiciones
                    if p['symbol'] in self.config.instrumentos_permitidos
                ]
                
            return posiciones
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_obtener_posiciones",
                    error=str(e)
                )
            raise
            
    async def obtener_balance(self) -> Dict[str, float]:
        """
        Obtener balance de la cuenta.
        
        Returns:
            Balance por moneda
        """
        try:
            balance = await self.exchange.fetch_balance()
            return {
                moneda: float(datos['free'])
                for moneda, datos in balance['total'].items()
                if float(datos['free']) > 0
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_obtener_balance",
                    error=str(e)
                )
            raise
            
    def _validar_limites(
        self,
        simbolo: str,
        cantidad: float,
        precio: Optional[float]
    ) -> bool:
        """
        Validar límites de operación.
        
        Args:
            simbolo: Par de trading
            cantidad: Cantidad a operar
            precio: Precio límite
            
        Returns:
            True si cumple los límites
        """
        try:
            # Validar instrumento permitido
            if (self.config.instrumentos_permitidos and
                simbolo not in self.config.instrumentos_permitidos):
                return False
                
            # Validar tamaño máximo de operación
            valor_operacion = cantidad
            if precio:
                valor_operacion *= precio
                
            if (self.config.limites['operacion_maxima'] > 0 and
                valor_operacion > self.config.limites['operacion_maxima']):
                return False
                
            # Validar posición máxima
            posiciones = asyncio.run(self.obtener_posiciones())
            posicion_total = sum(
                float(p['contracts'])
                for p in posiciones
                if p['symbol'] == simbolo
            )
            
            if (self.config.limites['posicion_maxima'] > 0 and
                posicion_total + cantidad > self.config.limites['posicion_maxima']):
                return False
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_validar_limites",
                    simbolo=simbolo,
                    cantidad=cantidad,
                    error=str(e)
                )
            return False

class GestorConectores:
    """Gestor de conectores de trading"""
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Inicializar gestor.
        
        Args:
            logger_manager: Gestor de logs
        """
        self.logger = logger_manager.get_logger("GestorConectores") if logger_manager else None
        self.conectores: Dict[str, ConectorBase] = {}
        
    async def crear_conector(
        self,
        config: ConectorTrading
    ) -> None:
        """
        Crear nuevo conector.
        
        Args:
            config: Configuración del conector
        """
        try:
            # Crear instancia según plataforma
            if config.plataforma in ccxt.exchanges:
                conector = ConectorCCXT(config, self.logger)
            else:
                raise ValueError(f"Plataforma no soportada: {config.plataforma}")
                
            # Conectar
            await conector.conectar()
            
            # Almacenar
            self.conectores[config.id] = conector
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_crear_conector",
                    plataforma=config.plataforma,
                    error=str(e)
                )
            raise
            
    async def eliminar_conector(
        self,
        conector_id: str
    ) -> None:
        """
        Eliminar conector.
        
        Args:
            conector_id: ID del conector
        """
        if conector_id in self.conectores:
            await self.conectores[conector_id].desconectar()
            del self.conectores[conector_id]
            
    def obtener_conector(
        self,
        conector_id: str
    ) -> Optional[ConectorBase]:
        """
        Obtener conector por ID.
        
        Args:
            conector_id: ID del conector
            
        Returns:
            Instancia del conector
        """
        return self.conectores.get(conector_id) 