import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configura un logger con formato consistente
    
    Args:
        name: Nombre del logger
        log_file: Ruta opcional al archivo de log
        
    Returns:
        Logger configurado
    """
    # Crear logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Crear formateador
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Configurar handler de archivo si se especifica
    if log_file:
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
    
def get_log_file(module_name: str) -> str:
    """
    Genera ruta de archivo de log para un módulo
    
    Args:
        module_name: Nombre del módulo
        
    Returns:
        Ruta al archivo de log
    """
    date_str = datetime.now().strftime('%Y%m%d')
    filename = f"{module_name}_{date_str}.log"
    return os.path.join('logs', filename) 