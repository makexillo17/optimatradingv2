"""
Sistema de exportación de datos
"""

from typing import Dict, List, Optional, Union, BinaryIO
import pandas as pd
import json
import csv
from datetime import datetime
import boto3
from google.cloud import storage
from azure.storage.blob import BlobServiceClient
import tempfile
import os
import gzip
from cryptography.fernet import Fernet

from .models import ExportConfig
from ..logging import LoggerManager

class ExportManager:
    """Gestor de exportación de datos"""
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None,
        encryption_key: Optional[str] = None
    ):
        """
        Inicializar gestor de exportación.
        
        Args:
            logger_manager: Gestor de logs
            encryption_key: Clave de encriptación
        """
        self.logger = logger_manager.get_logger("ExportManager") if logger_manager else None
        self.encryption_key = encryption_key
        if encryption_key:
            self.fernet = Fernet(encryption_key.encode())
            
        # Clientes de almacenamiento
        self.s3_client = None
        self.gcs_client = None
        self.azure_client = None
        
    def exportar_datos(
        self,
        datos: Union[pd.DataFrame, Dict, List],
        config: ExportConfig
    ) -> str:
        """
        Exportar datos según configuración.
        
        Args:
            datos: Datos a exportar
            config: Configuración de exportación
            
        Returns:
            Ruta del archivo exportado
        """
        try:
            # Aplicar filtros
            datos_filtrados = self._aplicar_filtros(datos, config.filtros)
            
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Exportar según formato
                if config.tipo == 'csv':
                    self._exportar_csv(datos_filtrados, temp_file, config.formato)
                elif config.tipo == 'json':
                    self._exportar_json(datos_filtrados, temp_file, config.formato)
                else:  # excel
                    self._exportar_excel(datos_filtrados, temp_file, config.formato)
                    
            # Comprimir si es necesario
            if config.compresion:
                temp_file = self._comprimir_archivo(temp_file.name)
                
            # Encriptar si es necesario
            if config.encriptacion and self.encryption_key:
                temp_file = self._encriptar_archivo(temp_file)
                
            # Subir a destino
            ruta_final = self._subir_archivo(temp_file, config.destino)
            
            # Limpiar archivo temporal
            os.unlink(temp_file)
            
            if self.logger:
                self.logger.info(
                    "datos_exportados",
                    config_id=config.id,
                    tipo=config.tipo,
                    ruta=ruta_final
                )
                
            return ruta_final
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "error_exportacion",
                    config_id=config.id,
                    error=str(e)
                )
            raise
            
    def _aplicar_filtros(
        self,
        datos: Union[pd.DataFrame, Dict, List],
        filtros: Dict
    ) -> Union[pd.DataFrame, Dict, List]:
        """
        Aplicar filtros a los datos.
        
        Args:
            datos: Datos a filtrar
            filtros: Configuración de filtros
            
        Returns:
            Datos filtrados
        """
        if isinstance(datos, pd.DataFrame):
            for columna, condicion in filtros.items():
                if isinstance(condicion, dict):
                    # Filtros de rango
                    if 'min' in condicion:
                        datos = datos[datos[columna] >= condicion['min']]
                    if 'max' in condicion:
                        datos = datos[datos[columna] <= condicion['max']]
                elif isinstance(condicion, list):
                    # Filtros de lista
                    datos = datos[datos[columna].isin(condicion)]
                else:
                    # Filtro exacto
                    datos = datos[datos[columna] == condicion]
                    
        elif isinstance(datos, list):
            for filtro, valor in filtros.items():
                if isinstance(valor, dict):
                    datos = [
                        d for d in datos
                        if (('min' not in valor or d.get(filtro, 0) >= valor['min']) and
                            ('max' not in valor or d.get(filtro, 0) <= valor['max']))
                    ]
                elif isinstance(valor, list):
                    datos = [
                        d for d in datos
                        if d.get(filtro) in valor
                    ]
                else:
                    datos = [
                        d for d in datos
                        if d.get(filtro) == valor
                    ]
                    
        return datos
        
    def _exportar_csv(
        self,
        datos: Union[pd.DataFrame, List[Dict]],
        archivo: BinaryIO,
        formato: Dict
    ) -> None:
        """
        Exportar datos a CSV.
        
        Args:
            datos: Datos a exportar
            archivo: Archivo destino
            formato: Configuración de formato
        """
        if isinstance(datos, pd.DataFrame):
            datos.to_csv(
                archivo.name,
                index=formato.get('incluir_indices', False),
                sep=formato.get('separador', ','),
                encoding=formato.get('encoding', 'utf-8')
            )
        else:
            campos = formato.get('campos', list(datos[0].keys()))
            writer = csv.DictWriter(
                archivo,
                fieldnames=campos,
                delimiter=formato.get('separador', ',')
            )
            writer.writeheader()
            writer.writerows(datos)
            
    def _exportar_json(
        self,
        datos: Union[pd.DataFrame, Dict, List],
        archivo: BinaryIO,
        formato: Dict
    ) -> None:
        """
        Exportar datos a JSON.
        
        Args:
            datos: Datos a exportar
            archivo: Archivo destino
            formato: Configuración de formato
        """
        if isinstance(datos, pd.DataFrame):
            datos = datos.to_dict(
                orient=formato.get('orientacion', 'records')
            )
            
        json.dump(
            datos,
            archivo,
            indent=formato.get('indent', 2),
            default=str
        )
        
    def _exportar_excel(
        self,
        datos: Union[pd.DataFrame, List[Dict]],
        archivo: BinaryIO,
        formato: Dict
    ) -> None:
        """
        Exportar datos a Excel.
        
        Args:
            datos: Datos a exportar
            archivo: Archivo destino
            formato: Configuración de formato
        """
        if not isinstance(datos, pd.DataFrame):
            datos = pd.DataFrame(datos)
            
        datos.to_excel(
            archivo.name,
            sheet_name=formato.get('hoja', 'Datos'),
            index=formato.get('incluir_indices', False)
        )
        
    def _comprimir_archivo(
        self,
        ruta: str
    ) -> str:
        """
        Comprimir archivo.
        
        Args:
            ruta: Ruta del archivo
            
        Returns:
            Ruta del archivo comprimido
        """
        ruta_gz = f"{ruta}.gz"
        with open(ruta, 'rb') as f_in:
            with gzip.open(ruta_gz, 'wb') as f_out:
                f_out.writelines(f_in)
                
        os.unlink(ruta)
        return ruta_gz
        
    def _encriptar_archivo(
        self,
        ruta: str
    ) -> str:
        """
        Encriptar archivo.
        
        Args:
            ruta: Ruta del archivo
            
        Returns:
            Ruta del archivo encriptado
        """
        ruta_enc = f"{ruta}.enc"
        with open(ruta, 'rb') as f_in:
            datos = f_in.read()
            datos_enc = self.fernet.encrypt(datos)
            with open(ruta_enc, 'wb') as f_out:
                f_out.write(datos_enc)
                
        os.unlink(ruta)
        return ruta_enc
        
    def _subir_archivo(
        self,
        ruta_local: str,
        destino: Dict[str, str]
    ) -> str:
        """
        Subir archivo a destino.
        
        Args:
            ruta_local: Ruta local del archivo
            destino: Configuración de destino
            
        Returns:
            Ruta final del archivo
        """
        nombre_archivo = os.path.basename(ruta_local)
        
        if destino['tipo'] == 'local':
            ruta_final = os.path.join(destino['ruta'], nombre_archivo)
            os.rename(ruta_local, ruta_final)
            
        elif destino['tipo'] == 's3':
            if not self.s3_client:
                self.s3_client = boto3.client('s3')
                
            bucket = destino['bucket']
            key = f"{destino['ruta']}/{nombre_archivo}"
            
            self.s3_client.upload_file(
                ruta_local,
                bucket,
                key
            )
            ruta_final = f"s3://{bucket}/{key}"
            
        elif destino['tipo'] == 'gcs':
            if not self.gcs_client:
                self.gcs_client = storage.Client()
                
            bucket = self.gcs_client.bucket(destino['bucket'])
            blob = bucket.blob(f"{destino['ruta']}/{nombre_archivo}")
            
            blob.upload_from_filename(ruta_local)
            ruta_final = f"gs://{destino['bucket']}/{blob.name}"
            
        elif destino['tipo'] == 'azure':
            if not self.azure_client:
                self.azure_client = BlobServiceClient.from_connection_string(
                    destino['connection_string']
                )
                
            container = self.azure_client.get_container_client(
                destino['container']
            )
            blob = f"{destino['ruta']}/{nombre_archivo}"
            
            with open(ruta_local, 'rb') as data:
                container.upload_blob(blob, data)
                
            ruta_final = f"azure://{destino['container']}/{blob}"
            
        return ruta_final 