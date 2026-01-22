from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import logging
from datetime import datetime

# Configuración de Logging
logger = logging.getLogger("Database")

# Base de SQLAlchemy
Base = declarative_base()

# Modelo Signal
class Signal(Base):
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset = Column(String)
    signal = Column(String) # long/short/neutral
    confidence = Column(Float)
    justification = Column(Text)
    raw_data = Column(JSON) # Detalles técnicos

    def __repr__(self):
        return f"<Signal(asset={self.asset}, signal={self.signal}, confidence={self.confidence})>"

# Variables globales para conexión
engine = None
SessionLocal = None

def init_db():
    """Inicializa la conexión a la base de datos y crea tablas si no existen."""
    global engine, SessionLocal
    
    database_url = os.environ.get("DATABASE_URL")
    
    if not database_url:
        logger.warning("DATABASE_URL no está configurada. La persistencia de datos está DESHABILITADA.")
        return

    try:
        # En Render u otros, postgres:// puede necesitar cambiarse a postgresql://
        if database_url.startswith("postgres://"):
            database_url = database_url.replace("postgres://", "postgresql://", 1)

        engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Crear tablas
        Base.metadata.create_all(bind=engine)
        logger.info("Base de datos inicializada y tablas verificadas.")
        
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {str(e)}")
        engine = None
        SessionLocal = None

def save_signal(asset, signal, confidence, justification, raw_data):
    """Guarda una señal en la base de datos."""
    if not SessionLocal:
        # Si no hay DB configurada, fallar silenciosamente (o loggear warning)
        # logger.debug("No hay conexión a DB, saltando persistencia.")
        return

    session = SessionLocal()
    try:
        new_signal = Signal(
            asset=asset,
            signal=signal,
            confidence=confidence,
            justification=justification,
            raw_data=raw_data
        )
        session.add(new_signal)
        session.commit()
        # logger.info(f"Signal guardada: {asset} -> {signal}")
    except Exception as e:
        logger.error(f"Error guardando señal en DB: {str(e)}")
        session.rollback()
    finally:
        session.close()

def get_recent_signals(limit=5):
    """Recupera las últimas señales guardadas."""
    if not SessionLocal:
        return []

    session = SessionLocal()
    try:
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(limit).all()
        # Convertir a dict para JSON
        return [
            {
                "id": s.id,
                "timestamp": s.timestamp.isoformat(),
                "asset": s.asset,
                "signal": s.signal,
                "confidence": s.confidence,
                "justification": s.justification,
                "raw_data": s.raw_data
            }
            for s in signals
        ]
    except Exception as e:
        logger.error(f"Error recuperando historial: {str(e)}")
        return []
    finally:
        session.close()
