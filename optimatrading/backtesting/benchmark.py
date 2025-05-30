"""
Benchmarking system for strategy evaluation
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .models import BenchmarkResult
from .engine import BacktestEngine, BacktestResult
from ..logging import LoggerManager

class BenchmarkManager:
    """
    System for comparing strategy performance against benchmarks
    and analyzing performance attribution.
    """
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Initialize benchmark manager.
        
        Args:
            logger_manager: Logger manager instance
        """
        self.logger = logger_manager.get_logger("BenchmarkManager") if logger_manager else None
        
    def compare_with_benchmark(
        self,
        strategy_result: BacktestResult,
        benchmark_data: pd.Series,
        risk_free_rate: float = 0.02
    ) -> BenchmarkResult:
        """
        Compare strategy performance with benchmark.
        
        Args:
            strategy_result: Strategy backtest result
            benchmark_data: Benchmark price data
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Benchmark comparison results
        """
        try:
            # Calculate returns
            strategy_returns = np.diff(strategy_result.equity_curve) / strategy_result.equity_curve[:-1]
            benchmark_returns = benchmark_data.pct_change().dropna().values
            
            # Align return series
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[-min_len:]
            benchmark_returns = benchmark_returns[-min_len:]
            
            # Calculate metrics
            alpha, beta = self._calculate_alpha_beta(
                strategy_returns,
                benchmark_returns,
                risk_free_rate
            )
            
            correlation = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
            
            tracking_error = np.std(strategy_returns - benchmark_returns) * np.sqrt(252)
            information_ratio = (
                np.mean(strategy_returns - benchmark_returns) * 252 /
                (tracking_error if tracking_error > 0 else float("inf"))
            )
            
            active_return = (
                (1 + strategy_returns).prod() -
                (1 + benchmark_returns).prod()
            )
            
            return BenchmarkResult(
                strategy_returns=strategy_returns.tolist(),
                benchmark_returns=benchmark_returns.tolist(),
                alpha=float(alpha),
                beta=float(beta),
                correlation=float(correlation),
                tracking_error=float(tracking_error),
                information_ratio=float(information_ratio),
                active_return=float(active_return)
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "benchmark_comparison_error",
                    error=str(e)
                )
            raise
            
    def run_buy_hold_benchmark(
        self,
        engine: BacktestEngine,
        market_data: Dict[str, pd.DataFrame],
        initial_weights: Optional[Dict[str, float]] = None
    ) -> BacktestResult:
        """
        Run buy and hold benchmark strategy.
        
        Args:
            engine: Backtest engine instance
            market_data: Market data by symbol
            initial_weights: Initial portfolio weights by symbol
            
        Returns:
            Backtest results for buy and hold
        """
        try:
            if initial_weights is None:
                # Equal weights
                symbols = list(market_data.keys())
                weight = 1.0 / len(symbols)
                initial_weights = {s: weight for s in symbols}
                
            def buy_hold_strategy(data: pd.Series, state: Dict) -> Dict[str, Dict]:
                """Simple buy and hold strategy"""
                if not state["positions"]:
                    # Initial positions
                    signals = {}
                    for symbol, weight in initial_weights.items():
                        signals[symbol] = {
                            "direction": 1,
                            "size": state["equity"] * weight
                        }
                    return signals
                return {}
                
            return engine.run_backtest(market_data, buy_hold_strategy)
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "buy_hold_benchmark_error",
                    error=str(e)
                )
            raise
            
    def analyze_performance_attribution(
        self,
        strategy_result: BacktestResult,
        market_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance attribution.
        
        Args:
            strategy_result: Strategy backtest result
            market_data: Market data by symbol
            benchmark_data: Benchmark price data
            
        Returns:
            Performance attribution analysis
        """
        try:
            # Calculate returns by symbol
            symbol_returns = {}
            for symbol in market_data:
                positions = [
                    p for p in strategy_result.positions
                    if p["symbol"] == symbol
                ]
                
                if positions:
                    pnl = sum(p["pnl"] for p in positions)
                    exposure = sum(
                        p["entry_price"] * p["size"]
                        for p in positions
                    )
                    symbol_returns[symbol] = pnl / exposure if exposure > 0 else 0
                    
            # Calculate benchmark contribution
            benchmark_return = (
                benchmark_data.iloc[-1] /
                benchmark_data.iloc[0] - 1
            )
            
            # Calculate attribution
            total_return = (
                strategy_result.equity_curve[-1] /
                strategy_result.equity_curve[0] - 1
            )
            
            market_contribution = benchmark_return
            selection_contribution = sum(
                ret - benchmark_return
                for ret in symbol_returns.values()
            )
            interaction_contribution = (
                total_return -
                market_contribution -
                selection_contribution
            )
            
            return {
                "total": {
                    "return": float(total_return),
                    "contribution": 1.0
                },
                "market": {
                    "return": float(market_contribution),
                    "contribution": float(market_contribution / total_return)
                },
                "selection": {
                    "return": float(selection_contribution),
                    "contribution": float(selection_contribution / total_return)
                },
                "interaction": {
                    "return": float(interaction_contribution),
                    "contribution": float(interaction_contribution / total_return)
                },
                "by_symbol": {
                    symbol: {
                        "return": float(ret),
                        "contribution": float(ret / total_return)
                    }
                    for symbol, ret in symbol_returns.items()
                }
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "attribution_analysis_error",
                    error=str(e)
                )
            raise
            
    def _calculate_alpha_beta(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float
    ) -> Tuple[float, float]:
        """Calculate strategy alpha and beta"""
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate excess returns
        strategy_excess = strategy_returns - daily_rf
        benchmark_excess = benchmark_returns - daily_rf
        
        # Calculate beta using covariance method
        beta = (
            np.cov(strategy_returns, benchmark_returns)[0, 1] /
            np.var(benchmark_returns)
            if np.var(benchmark_returns) > 0 else 0
        )
        
        # Calculate alpha
        alpha = (
            np.mean(strategy_excess) -
            beta * np.mean(benchmark_excess)
        ) * 252  # Annualize alpha
        
        return alpha, beta 