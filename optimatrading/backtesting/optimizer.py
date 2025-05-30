"""
Parameter optimization system
"""

from typing import Dict, List, Optional, Union, Tuple, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import TimeSeriesSplit
import optuna

from .models import OptimizationConfig, OptimizationResult, BacktestResult
from .engine import BacktestEngine
from ..logging import LoggerManager

class ParameterOptimizer:
    """
    System for optimizing strategy parameters using various methods.
    """
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Initialize parameter optimizer.
        
        Args:
            logger_manager: Logger manager instance
        """
        self.logger = logger_manager.get_logger("ParameterOptimizer") if logger_manager else None
        
    def optimize_parameters(
        self,
        engine: BacktestEngine,
        market_data: Dict[str, pd.DataFrame],
        strategy_template: Callable,
        config: OptimizationConfig
    ) -> OptimizationResult:
        """
        Optimize strategy parameters.
        
        Args:
            engine: Backtest engine instance
            market_data: Market data by symbol
            strategy_template: Strategy template function
            config: Optimization configuration
            
        Returns:
            Optimization results
        """
        try:
            # Create study
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=config.random_state
                )
            )
            
            # Define objective function
            def objective(trial):
                # Generate parameters
                params = {}
                for name, ranges in config.parameter_ranges.items():
                    if isinstance(ranges["min"], int):
                        params[name] = trial.suggest_int(
                            name,
                            ranges["min"],
                            ranges["max"],
                            ranges.get("step", 1)
                        )
                    else:
                        params[name] = trial.suggest_float(
                            name,
                            ranges["min"],
                            ranges["max"],
                            log=ranges.get("log", False)
                        )
                        
                # Create strategy instance
                strategy = lambda data, state: strategy_template(data, state, params)
                
                # Run cross-validation
                cv_scores = self._run_time_series_cv(
                    engine,
                    market_data,
                    strategy,
                    config.cross_validation_folds
                )
                
                return np.mean(cv_scores)
                
            # Run optimization
            study.optimize(
                objective,
                n_trials=config.max_iterations,
                n_jobs=-1
            )
            
            # Calculate parameter importance
            importance = optuna.importance.get_param_importances(study)
            
            # Extract parameter scores
            parameter_scores = {}
            for param in config.parameter_ranges:
                scores = [
                    trial.params[param]
                    for trial in study.trials
                    if trial.state == optuna.trial.TrialState.COMPLETE
                ]
                parameter_scores[param] = scores
                
            return OptimizationResult(
                best_parameters=study.best_params,
                best_score=float(study.best_value),
                parameter_scores=parameter_scores,
                cross_validation_scores=self._run_time_series_cv(
                    engine,
                    market_data,
                    lambda data, state: strategy_template(data, state, study.best_params),
                    config.cross_validation_folds
                ),
                optimization_path=[
                    {
                        **trial.params,
                        "score": trial.value
                    }
                    for trial in study.trials
                    if trial.state == optuna.trial.TrialState.COMPLETE
                ],
                parameter_importance=importance
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "parameter_optimization_error",
                    error=str(e)
                )
            raise
            
    def analyze_parameter_sensitivity(
        self,
        engine: BacktestEngine,
        market_data: Dict[str, pd.DataFrame],
        strategy_template: Callable,
        base_params: Dict[str, Union[float, int]],
        param_ranges: Dict[str, Dict[str, Union[float, int]]],
        n_points: int = 20
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Analyze parameter sensitivity.
        
        Args:
            engine: Backtest engine instance
            market_data: Market data by symbol
            strategy_template: Strategy template function
            base_params: Base parameter values
            param_ranges: Parameter ranges to test
            n_points: Number of points to test for each parameter
            
        Returns:
            Sensitivity analysis results
        """
        try:
            results = {}
            
            for param_name, ranges in param_ranges.items():
                param_min = ranges["min"]
                param_max = ranges["max"]
                
                # Generate test points
                if isinstance(param_min, int):
                    test_values = np.linspace(
                        param_min,
                        param_max,
                        n_points,
                        dtype=int
                    )
                else:
                    test_values = np.linspace(
                        param_min,
                        param_max,
                        n_points
                    )
                    
                # Test each value
                scores = []
                for value in test_values:
                    # Create parameter set
                    test_params = base_params.copy()
                    test_params[param_name] = value
                    
                    # Create strategy
                    strategy = lambda data, state: strategy_template(
                        data,
                        state,
                        test_params
                    )
                    
                    # Run backtest
                    result = engine.run_backtest(market_data, strategy)
                    
                    # Store score
                    if isinstance(result, BacktestResult):
                        scores.append(result.sharpe_ratio)
                    else:
                        scores.append(0.0)
                        
                results[param_name] = {
                    "values": test_values.tolist(),
                    "scores": scores
                }
                
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "sensitivity_analysis_error",
                    error=str(e)
                )
            raise
            
    def analyze_parameter_robustness(
        self,
        engine: BacktestEngine,
        market_data: Dict[str, pd.DataFrame],
        strategy_template: Callable,
        base_params: Dict[str, Union[float, int]],
        param_ranges: Dict[str, Dict[str, Union[float, int]]],
        n_samples: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze parameter robustness through Monte Carlo sampling.
        
        Args:
            engine: Backtest engine instance
            market_data: Market data by symbol
            strategy_template: Strategy template function
            base_params: Base parameter values
            param_ranges: Parameter ranges for sampling
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Robustness analysis results
        """
        try:
            results = {}
            
            # Generate random samples
            for _ in range(n_samples):
                # Sample parameters
                sample_params = base_params.copy()
                for param_name, ranges in param_ranges.items():
                    if isinstance(ranges["min"], int):
                        value = np.random.randint(
                            ranges["min"],
                            ranges["max"] + 1
                        )
                    else:
                        value = np.random.uniform(
                            ranges["min"],
                            ranges["max"]
                        )
                    sample_params[param_name] = value
                    
                # Create strategy
                strategy = lambda data, state: strategy_template(
                    data,
                    state,
                    sample_params
                )
                
                # Run backtest
                result = engine.run_backtest(market_data, strategy)
                
                # Store results
                if isinstance(result, BacktestResult):
                    for param_name, value in sample_params.items():
                        if param_name not in results:
                            results[param_name] = {
                                "values": [],
                                "scores": []
                            }
                        results[param_name]["values"].append(value)
                        results[param_name]["scores"].append(result.sharpe_ratio)
                        
            # Calculate statistics
            stats = {}
            for param_name, data in results.items():
                values = np.array(data["values"])
                scores = np.array(data["scores"])
                
                # Calculate correlation
                correlation = np.corrcoef(values, scores)[0, 1]
                
                # Calculate stability metrics
                score_std = np.std(scores)
                score_range = np.ptp(scores)
                
                stats[param_name] = {
                    "correlation": float(correlation),
                    "score_std": float(score_std),
                    "score_range": float(score_range),
                    "mean_score": float(np.mean(scores)),
                    "median_score": float(np.median(scores))
                }
                
            return stats
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "robustness_analysis_error",
                    error=str(e)
                )
            raise
            
    def _run_time_series_cv(
        self,
        engine: BacktestEngine,
        market_data: Dict[str, pd.DataFrame],
        strategy: Callable,
        n_splits: int
    ) -> List[float]:
        """Run time series cross-validation"""
        # Prepare data
        dates = pd.DatetimeIndex([
            d for d in market_data[next(iter(market_data))].index
        ])
        
        # Create splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, test_idx in tscv.split(dates):
            # Split data
            train_data = {
                symbol: df.iloc[train_idx]
                for symbol, df in market_data.items()
            }
            test_data = {
                symbol: df.iloc[test_idx]
                for symbol, df in market_data.items()
            }
            
            # Train on training data
            train_result = engine.run_backtest(train_data, strategy)
            
            # Test on test data
            test_result = engine.run_backtest(test_data, strategy)
            
            # Store score
            if isinstance(test_result, BacktestResult):
                scores.append(test_result.sharpe_ratio)
            else:
                scores.append(0.0)
                
        return scores 