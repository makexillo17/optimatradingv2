"""
Data models for backtesting components
"""

from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
import numpy as np

class MarketScenario(BaseModel):
    """Market scenario configuration"""
    name: str
    description: str
    volatility: float = Field(ge=0.0)
    trend: float  # Positive for uptrend, negative for downtrend
    liquidity: float = Field(ge=0.0, le=1.0)
    correlation_regime: str = Field(regex="^(normal|stress|crisis)$")
    start_date: datetime
    end_date: datetime
    
    @validator("end_date")
    def end_date_must_be_after_start_date(cls, v, values):
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

class BacktestConfig(BaseModel):
    """Backtesting configuration"""
    initial_capital: float = Field(gt=0.0)
    commission_rate: float = Field(ge=0.0, le=0.01)
    slippage_model: str = Field(regex="^(fixed|percentage|smart)$")
    slippage_value: float = Field(ge=0.0)
    position_size_limit: float = Field(ge=0.0, le=1.0)
    max_positions: int = Field(ge=1)
    risk_free_rate: float = 0.02  # Annual
    
    class Config:
        validate_assignment = True

class OptimizationConfig(BaseModel):
    """Parameter optimization configuration"""
    parameter_ranges: Dict[str, Dict[str, Union[float, int]]]  # param -> {min, max, step}
    optimization_metric: str
    cross_validation_folds: int = Field(ge=2, le=10)
    max_iterations: int = Field(ge=100)
    random_state: int = 42

class BacktestResult(BaseModel):
    """Backtesting results"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: int
    win_rate: float
    profit_factor: float
    avg_trade: float
    avg_bars_held: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    positions: List[Dict[str, Union[str, float, datetime]]]
    metrics_by_period: Dict[str, Dict[str, float]]
    
    @validator("sharpe_ratio", "sortino_ratio")
    def validate_ratios(cls, v):
        if not np.isfinite(v):
            return 0.0
        return v

class BenchmarkResult(BaseModel):
    """Benchmark comparison results"""
    strategy_returns: List[float]
    benchmark_returns: List[float]
    alpha: float
    beta: float
    correlation: float
    tracking_error: float
    information_ratio: float
    active_return: float
    
    @validator("alpha", "beta", "correlation")
    def validate_metrics(cls, v):
        if not np.isfinite(v):
            return 0.0
        return v

class OptimizationResult(BaseModel):
    """Parameter optimization results"""
    best_parameters: Dict[str, Union[float, int]]
    best_score: float
    parameter_scores: Dict[str, List[float]]  # param -> list of scores
    cross_validation_scores: List[float]
    optimization_path: List[Dict[str, Union[float, int]]]
    parameter_importance: Dict[str, float]
    
    @validator("best_score")
    def validate_score(cls, v):
        if not np.isfinite(v):
            return 0.0
        return v

class MonteCarloResult(BaseModel):
    """Monte Carlo simulation results"""
    confidence_intervals: Dict[str, Dict[str, float]]  # metric -> {5%, 50%, 95%}
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    probability_profit: float
    expected_return: float
    return_distribution: Dict[str, List[float]]  # bins and frequencies
    
    @validator("var_95", "cvar_95", "probability_profit")
    def validate_risk_metrics(cls, v):
        if not np.isfinite(v):
            return 0.0
        return v 