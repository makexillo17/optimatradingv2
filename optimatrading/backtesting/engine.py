"""
Core backtesting engine
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from .models import (
    BacktestConfig,
    BacktestResult,
    MarketScenario,
    MonteCarloResult
)
from ..logging import LoggerManager

@dataclass
class Position:
    """Trading position"""
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    entry_time: datetime
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    bars_held: int = 0

class BacktestEngine:
    """
    Core backtesting engine with realistic trading simulation.
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration
            logger_manager: Logger manager instance
        """
        self.config = config
        self.logger = logger_manager.get_logger("BacktestEngine") if logger_manager else None
        
        self.reset()
        
    def reset(self) -> None:
        """Reset engine state"""
        self.equity = self.config.initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.equity_curve = [self.equity]
        self.drawdown_curve = [0.0]
        self.current_time: Optional[datetime] = None
        
    def run_backtest(
        self,
        market_data: Dict[str, pd.DataFrame],
        strategy: callable,
        scenario: Optional[MarketScenario] = None
    ) -> BacktestResult:
        """
        Run backtest simulation.
        
        Args:
            market_data: Market data by symbol
            strategy: Strategy function that generates signals
            scenario: Optional market scenario
            
        Returns:
            Backtest results
        """
        try:
            self.reset()
            
            # Prepare data
            aligned_data = self._align_data(market_data)
            if scenario:
                aligned_data = self._apply_scenario(aligned_data, scenario)
                
            # Run simulation
            for timestamp, data in aligned_data.iterrows():
                self.current_time = timestamp
                
                # Update positions
                self._update_positions(data)
                
                # Generate signals
                signals = strategy(data, self._get_state())
                
                # Execute trades
                self._execute_trades(signals, data)
                
                # Update equity curve
                self._update_equity_curve()
                
            # Close remaining positions
            self._close_all_positions(aligned_data.iloc[-1])
            
            # Calculate results
            return self._calculate_results()
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "backtest_error",
                    error=str(e)
                )
            raise
            
    def run_monte_carlo(
        self,
        base_result: BacktestResult,
        n_simulations: int = 1000,
        window_size: int = 20
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            base_result: Base backtest result
            n_simulations: Number of simulations
            window_size: Window size for block bootstrap
            
        Returns:
            Monte Carlo simulation results
        """
        try:
            returns = np.diff(base_result.equity_curve) / base_result.equity_curve[:-1]
            
            # Run simulations in parallel
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._run_single_simulation,
                        returns,
                        window_size
                    )
                    for _ in range(n_simulations)
                ]
                
                simulated_returns = [
                    future.result()
                    for future in futures
                ]
                
            # Calculate confidence intervals
            confidence_intervals = {}
            metrics = ["total_return", "max_drawdown", "sharpe_ratio"]
            
            for metric in metrics:
                values = [
                    self._calculate_metric(returns, metric)
                    for returns in simulated_returns
                ]
                
                confidence_intervals[metric] = {
                    "5%": np.percentile(values, 5),
                    "50%": np.percentile(values, 50),
                    "95%": np.percentile(values, 95)
                }
                
            # Calculate risk metrics
            var_95 = -np.percentile([r.min() for r in simulated_returns], 95)
            cvar_95 = -np.mean([
                r[r < -var_95].mean() if len(r[r < -var_95]) > 0 else 0
                for r in simulated_returns
            ])
            
            probability_profit = np.mean([
                (1 + r).prod() > 1
                for r in simulated_returns
            ])
            
            expected_return = np.mean([
                (1 + r).prod() - 1
                for r in simulated_returns
            ])
            
            # Calculate return distribution
            all_returns = np.concatenate(simulated_returns)
            hist, bins = np.histogram(all_returns, bins=50)
            
            return MonteCarloResult(
                confidence_intervals=confidence_intervals,
                var_95=float(var_95),
                cvar_95=float(cvar_95),
                probability_profit=float(probability_profit),
                expected_return=float(expected_return),
                return_distribution={
                    "bins": bins.tolist(),
                    "frequencies": hist.tolist()
                }
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "monte_carlo_error",
                    error=str(e)
                )
            raise
            
    def _align_data(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Align data from different symbols"""
        aligned = pd.concat(
            [df.add_suffix(f"_{symbol}") for symbol, df in market_data.items()],
            axis=1
        )
        return aligned.dropna()
        
    def _apply_scenario(
        self,
        data: pd.DataFrame,
        scenario: MarketScenario
    ) -> pd.DataFrame:
        """Apply market scenario to data"""
        # Filter by date range
        data = data[
            (data.index >= scenario.start_date) &
            (data.index <= scenario.end_date)
        ].copy()
        
        # Apply volatility adjustment
        returns = data.pct_change()
        vol_ratio = scenario.volatility / returns.std()
        adjusted_returns = returns * vol_ratio
        
        # Apply trend
        trend_returns = adjusted_returns + scenario.trend / len(data)
        
        # Reconstruct prices
        result = (1 + trend_returns).cumprod()
        result.iloc[0] = 1
        
        # Scale back to original price levels
        for col in data.columns:
            result[col] *= data[col].iloc[0]
            
        return result
        
    def _update_positions(
        self,
        data: pd.Series
    ) -> None:
        """Update open positions"""
        for pos in self.positions:
            # Update bars held
            pos.bars_held += 1
            
            # Check stop loss and take profit
            current_price = data[f"close_{pos.symbol}"]
            
            if pos.stop_loss and (
                (pos.direction == 1 and current_price <= pos.stop_loss) or
                (pos.direction == -1 and current_price >= pos.stop_loss)
            ):
                self._close_position(pos, current_price, "stop_loss")
                
            elif pos.take_profit and (
                (pos.direction == 1 and current_price >= pos.take_profit) or
                (pos.direction == -1 and current_price <= pos.take_profit)
            ):
                self._close_position(pos, current_price, "take_profit")
                
    def _execute_trades(
        self,
        signals: Dict[str, Dict[str, Union[int, float]]],
        data: pd.Series
    ) -> None:
        """Execute trading signals"""
        for symbol, signal in signals.items():
            direction = signal.get("direction", 0)
            size = signal.get("size", 0.0)
            
            if direction != 0 and size > 0:
                # Check position limits
                if len(self.positions) >= self.config.max_positions:
                    continue
                    
                # Calculate position size
                price = data[f"close_{symbol}"]
                adjusted_size = min(
                    size,
                    self.equity * self.config.position_size_limit / price
                )
                
                # Apply slippage
                if self.config.slippage_model == "fixed":
                    price += self.config.slippage_value * direction
                elif self.config.slippage_model == "percentage":
                    price *= (1 + self.config.slippage_value * direction)
                else:  # smart slippage
                    volume = data[f"volume_{symbol}"]
                    price *= (1 + self.config.slippage_value * direction * (size / volume))
                    
                # Create position
                position = Position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=price,
                    entry_time=self.current_time,
                    size=adjusted_size,
                    stop_loss=signal.get("stop_loss"),
                    take_profit=signal.get("take_profit")
                )
                
                self.positions.append(position)
                
    def _close_position(
        self,
        position: Position,
        price: float,
        reason: str
    ) -> None:
        """Close a position"""
        position.exit_price = price
        position.exit_time = self.current_time
        
        # Calculate PnL
        position.pnl = (
            position.direction *
            (position.exit_price - position.entry_price) *
            position.size
        )
        
        # Apply commission
        position.pnl -= (
            position.entry_price * position.size *
            self.config.commission_rate
        )
        position.pnl -= (
            position.exit_price * position.size *
            self.config.commission_rate
        )
        
        self.equity += position.pnl
        self.positions.remove(position)
        self.closed_positions.append(position)
        
    def _close_all_positions(
        self,
        data: pd.Series
    ) -> None:
        """Close all open positions"""
        for position in self.positions[:]:
            price = data[f"close_{position.symbol}"]
            self._close_position(position, price, "end_of_test")
            
    def _update_equity_curve(self) -> None:
        """Update equity curve and drawdown"""
        self.equity_curve.append(self.equity)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - self.equity) / peak
        self.drawdown_curve.append(drawdown)
        
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results"""
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate metrics
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        n_days = (self.current_time - self.closed_positions[0].entry_time).days
        annualized_return = (1 + total_return) ** (365 / n_days) - 1
        
        # Risk metrics
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / np.std(excess_returns[excess_returns < 0])
        
        # Trading metrics
        n_trades = len(self.closed_positions)
        winning_trades = len([p for p in self.closed_positions if p.pnl > 0])
        win_rate = winning_trades / n_trades if n_trades > 0 else 0
        
        profit_factor = (
            sum(p.pnl for p in self.closed_positions if p.pnl > 0) /
            abs(sum(p.pnl for p in self.closed_positions if p.pnl < 0))
            if sum(p.pnl for p in self.closed_positions if p.pnl < 0) != 0
            else float("inf")
        )
        
        avg_trade = sum(p.pnl for p in self.closed_positions) / n_trades if n_trades > 0 else 0
        avg_bars = sum(p.bars_held for p in self.closed_positions) / n_trades if n_trades > 0 else 0
        
        # Period metrics
        metrics_by_period = {
            "monthly": self._calculate_period_metrics(returns, 21),
            "quarterly": self._calculate_period_metrics(returns, 63),
            "yearly": self._calculate_period_metrics(returns, 252)
        }
        
        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            max_drawdown=float(max(self.drawdown_curve)),
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            trades=n_trades,
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            avg_trade=float(avg_trade),
            avg_bars_held=float(avg_bars),
            equity_curve=equity_curve.tolist(),
            drawdown_curve=self.drawdown_curve,
            positions=[
                {
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "entry_time": p.entry_time,
                    "exit_price": p.exit_price,
                    "exit_time": p.exit_time,
                    "pnl": p.pnl,
                    "bars_held": p.bars_held
                }
                for p in self.closed_positions
            ],
            metrics_by_period=metrics_by_period
        )
        
    def _calculate_period_metrics(
        self,
        returns: np.ndarray,
        period_length: int
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by period"""
        n_periods = len(returns) // period_length
        periods = {}
        
        for i in range(n_periods):
            start_idx = i * period_length
            end_idx = start_idx + period_length
            period_returns = returns[start_idx:end_idx]
            
            period_metrics = {
                "return": float((1 + period_returns).prod() - 1),
                "volatility": float(period_returns.std() * np.sqrt(252)),
                "sharpe": float(
                    np.sqrt(252) * (period_returns.mean() - self.config.risk_free_rate/252) /
                    period_returns.std()
                    if period_returns.std() > 0 else 0
                )
            }
            
            periods[f"period_{i+1}"] = period_metrics
            
        return periods
        
    def _run_single_simulation(
        self,
        returns: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """Run single Monte Carlo simulation"""
        n_returns = len(returns)
        n_blocks = n_returns // window_size
        
        # Block bootstrap
        block_indices = np.random.randint(
            0,
            n_returns - window_size + 1,
            size=n_blocks
        )
        
        simulated_returns = np.concatenate([
            returns[i:i+window_size]
            for i in block_indices
        ])
        
        return simulated_returns[:n_returns]
        
    def _calculate_metric(
        self,
        returns: np.ndarray,
        metric: str
    ) -> float:
        """Calculate single metric for simulation"""
        if metric == "total_return":
            return float((1 + returns).prod() - 1)
        elif metric == "max_drawdown":
            cum_returns = (1 + returns).cumprod()
            peaks = np.maximum.accumulate(cum_returns)
            drawdowns = (peaks - cum_returns) / peaks
            return float(drawdowns.max())
        elif metric == "sharpe_ratio":
            excess_returns = returns - self.config.risk_free_rate/252
            return float(
                np.sqrt(252) * excess_returns.mean() / excess_returns.std()
                if excess_returns.std() > 0 else 0
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
    def _get_state(self) -> Dict[str, Union[float, List[Dict]]]:
        """Get current engine state"""
        return {
            "equity": self.equity,
            "positions": [
                {
                    "symbol": p.symbol,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "size": p.size,
                    "pnl": p.pnl,
                    "bars_held": p.bars_held
                }
                for p in self.positions
            ]
        } 