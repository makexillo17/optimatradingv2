"""
Backtesting and analysis components for Optimatrading
"""

from .models import (
    MarketScenario,
    BacktestConfig,
    OptimizationConfig,
    BacktestResult,
    BenchmarkResult,
    OptimizationResult,
    MonteCarloResult
)
from .engine import BacktestEngine, Position
from .benchmark import BenchmarkManager
from .optimizer import ParameterOptimizer

__all__ = [
    'MarketScenario',
    'BacktestConfig',
    'OptimizationConfig',
    'BacktestResult',
    'BenchmarkResult',
    'OptimizationResult',
    'MonteCarloResult',
    'BacktestEngine',
    'Position',
    'BenchmarkManager',
    'ParameterOptimizer'
] 