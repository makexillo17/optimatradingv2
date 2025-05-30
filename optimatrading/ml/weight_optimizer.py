"""
Dynamic weight optimization system
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

from ..logging import LoggerManager
from .models import ModulePerformance, MLConfig
from ..utils.performance import PerformanceMonitor

class WeightOptimizer:
    """
    Optimizes module weights based on historical performance.
    Supports multiple time horizons and dynamic adjustment.
    """
    
    def __init__(
        self,
        config: MLConfig,
        logger_manager: Optional[LoggerManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize weight optimizer.
        
        Args:
            config: ML configuration
            logger_manager: Logger manager instance
            performance_monitor: Performance monitoring instance
        """
        self.config = config
        self.logger = logger_manager.get_logger("WeightOptimizer") if logger_manager else None
        self.performance_monitor = performance_monitor
        
        self._scaler = MinMaxScaler()
        self._last_update = None
        self._current_weights = None
        
    @performance_monitor.monitor("ml") if performance_monitor else lambda x: x
    def update_weights(
        self,
        performances: Dict[str, List[ModulePerformance]],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Update module weights based on performance metrics.
        
        Args:
            performances: Historical performance metrics by module
            correlation_matrix: Optional correlation matrix for diversification
            min_weight: Minimum weight per module
            max_weight: Maximum weight per module
            
        Returns:
            Dictionary mapping module IDs to weights
        """
        try:
            # Calculate performance scores for each horizon
            horizon_scores = self._calculate_horizon_scores(performances)
            
            # Combine scores across horizons
            combined_scores = self._combine_horizon_scores(horizon_scores)
            
            # Optimize weights considering correlations
            weights = self._optimize_weights(
                combined_scores,
                correlation_matrix,
                min_weight,
                max_weight
            )
            
            self._current_weights = weights
            self._last_update = datetime.now()
            
            if self.logger:
                self.logger.info(
                    "weights_updated",
                    weights=weights,
                    scores=combined_scores
                )
                
            return weights
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "weight_update_error",
                    error=str(e)
                )
            raise
            
    def _calculate_horizon_scores(
        self,
        performances: Dict[str, List[ModulePerformance]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate performance scores for each time horizon"""
        horizon_scores = {}
        
        for module_id, history in performances.items():
            horizon_scores[module_id] = {}
            
            # Group by horizon
            for window in ["short", "medium", "long"]:
                window_metrics = [
                    m for m in history if m.window == window
                ]
                
                if not window_metrics:
                    horizon_scores[module_id][window] = 0.0
                    continue
                    
                # Calculate weighted average of metrics
                scores = []
                weights = []
                
                for metric in window_metrics:
                    # Combine multiple metrics into single score
                    score = (
                        0.4 * metric.accuracy +
                        0.3 * metric.precision +
                        0.2 * metric.recall +
                        0.1 * metric.f1_score
                    )
                    
                    # More recent metrics get higher weight
                    age = (datetime.now() - metric.timestamp).total_seconds()
                    weight = np.exp(-age / self.config.performance_windows[window])
                    
                    scores.append(score)
                    weights.append(weight)
                    
                # Calculate weighted average
                horizon_scores[module_id][window] = np.average(
                    scores,
                    weights=weights
                )
                
        return horizon_scores
        
    def _combine_horizon_scores(
        self,
        horizon_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine scores across horizons"""
        combined_scores = {}
        
        # Weights for different horizons
        horizon_weights = {
            "short": 0.5,    # Recent performance most important
            "medium": 0.3,   # Medium term still significant
            "long": 0.2      # Long term provides stability
        }
        
        for module_id, scores in horizon_scores.items():
            # Weighted average across horizons
            combined_scores[module_id] = sum(
                scores[window] * horizon_weights[window]
                for window in ["short", "medium", "long"]
            )
            
        return combined_scores
        
    def _optimize_weights(
        self,
        scores: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]],
        min_weight: float,
        max_weight: float
    ) -> Dict[str, float]:
        """Optimize weights considering correlations"""
        module_ids = list(scores.keys())
        n_modules = len(module_ids)
        
        # Initial weights proportional to scores
        total_score = sum(scores.values())
        initial_weights = np.array([
            scores[mid] / total_score for mid in module_ids
        ])
        
        # Constraints
        constraints = [
            # Sum to 1
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        ]
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n_modules)]
        
        def objective(weights):
            # Maximize performance while minimizing correlation
            perf_term = -np.sum(weights * [scores[mid] for mid in module_ids])
            
            if correlation_matrix:
                # Add correlation penalty
                corr_term = 0
                for i, mod1 in enumerate(module_ids):
                    for j, mod2 in enumerate(module_ids):
                        if i != j:
                            corr = correlation_matrix[mod1][mod2]
                            corr_term += weights[i] * weights[j] * abs(corr)
                return perf_term + 0.5 * corr_term
            
            return perf_term
            
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if not result.success:
            if self.logger:
                self.logger.warning(
                    "weight_optimization_failed",
                    error=result.message
                )
            # Fall back to normalized scores
            optimized_weights = initial_weights
        else:
            optimized_weights = result.x
            
        # Return as dictionary
        return {
            mid: float(w) for mid, w in zip(module_ids, optimized_weights)
        } 