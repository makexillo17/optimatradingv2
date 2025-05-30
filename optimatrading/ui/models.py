"""
Data models for UI components
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

class ChartConfig(BaseModel):
    """Configuration for chart display"""
    chart_type: str = Field("candlestick", regex="^(candlestick|line|ohlc)$")
    timeframe: str = Field("1h", regex="^[0-9]+[mhd]$")
    indicators: List[str] = []
    show_volume: bool = True
    theme: str = Field("dark", regex="^(dark|light)$")
    height: int = Field(600, gt=200)
    width: Optional[int] = None  # None for responsive

class DashboardLayout(BaseModel):
    """User dashboard layout preferences"""
    layout_name: str
    components: List[str]
    positions: Dict[str, Dict[str, int]]  # component -> {row, col, rowspan, colspan}
    refresh_interval: int = Field(60, ge=5)  # seconds

class AlertConfig(BaseModel):
    """User alert configuration"""
    alert_id: str
    metric: str
    condition: str = Field(regex="^(above|below|crosses)$")
    threshold: float
    cooldown: int = Field(300, ge=60)  # seconds
    notification_channels: List[str]
    enabled: bool = True

class Annotation(BaseModel):
    """User annotation on chart or analysis"""
    timestamp: datetime
    text: str
    category: str
    chart_id: Optional[str]
    coordinates: Optional[Dict[str, float]]
    tags: List[str] = []

class ExplanationConfig(BaseModel):
    """Configuration for explanations"""
    detail_level: str = Field("medium", regex="^(basic|medium|technical)$")
    show_visuals: bool = True
    include_counterfactuals: bool = True
    max_factors: int = Field(5, ge=1, le=10)

class UserPreferences(BaseModel):
    """User interface preferences"""
    default_chart_config: ChartConfig
    default_layout: str
    alerts: List[AlertConfig] = []
    explanation_config: ExplanationConfig
    color_scheme: Dict[str, str]
    language: str = "en"
    timezone: str = "UTC" 