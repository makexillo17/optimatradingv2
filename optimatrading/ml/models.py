"""
Data models for machine learning components
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

class CorrelationMatrix(BaseModel):
    """Module correlation matrix with timestamps"""
    matrix: Dict[str, Dict[str, float]]
    last_updated: datetime
    regime: str = "normal"
    confidence: float = Field(ge=0.0, le=1.0)

class ModulePerformance(BaseModel):
    """Historical performance metrics for a module"""
    module_id: str
    accuracy: float = Field(ge=0.0, le=1.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime
    window: str = Field(regex="^(short|medium|long)$")

class MarketAnomaly(BaseModel):
    """Detected market anomaly"""
    timestamp: datetime
    anomaly_type: str
    severity: float = Field(ge=0.0, le=1.0)
    affected_metrics: List[str]
    description: str
    suggested_actions: Optional[List[str]]

class PredictionRecord(BaseModel):
    """Historical prediction record"""
    module_id: str
    timestamp: datetime
    prediction: Dict[str, float]
    actual: Dict[str, float]
    market_conditions: Dict[str, Union[str, float]]
    performance_metrics: Dict[str, float]

class MLConfig(BaseModel):
    """Configuration for ML components"""
    correlation_update_interval: int = Field(3600, gt=0)  # seconds
    weight_update_interval: int = Field(1800, gt=0)  # seconds
    anomaly_detection_sensitivity: float = Field(0.95, ge=0.0, le=1.0)
    performance_windows: Dict[str, int] = Field(
        default_factory=lambda: {
            "short": 24 * 3600,  # 1 day
            "medium": 7 * 24 * 3600,  # 1 week
            "long": 30 * 24 * 3600  # 30 days
        }
    ) 