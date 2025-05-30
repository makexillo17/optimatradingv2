"""
Dynamic correlation learning system
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from ..logging import LoggerManager
from .models import CorrelationMatrix, MLConfig
from ..utils.performance import PerformanceMonitor

class CorrelationLearner:
    """
    Learns and updates correlations between modules dynamically.
    Uses robust estimation and regime detection.
    """
    
    def __init__(
        self,
        config: MLConfig,
        logger_manager: Optional[LoggerManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize correlation learner.
        
        Args:
            config: ML configuration
            logger_manager: Logger manager instance
            performance_monitor: Performance monitoring instance
        """
        self.config = config
        self.logger = logger_manager.get_logger("CorrelationLearner") if logger_manager else None
        self.performance_monitor = performance_monitor
        
        self._last_update = None
        self._current_regime = "normal"
        self._scaler = StandardScaler()
        self._regime_detector = KMeans(n_clusters=3, random_state=42)
        
    @performance_monitor.monitor("ml") if performance_monitor else lambda x: x
    def update_correlations(
        self,
        historical_data: Dict[str, pd.DataFrame],
        min_periods: int = 100
    ) -> CorrelationMatrix:
        """
        Update correlation matrix using recent historical data.
        
        Args:
            historical_data: Dict mapping module IDs to their historical predictions
            min_periods: Minimum number of periods required
            
        Returns:
            Updated correlation matrix
        """
        try:
            # Convert to combined DataFrame
            combined_data = self._prepare_data(historical_data)
            
            if len(combined_data) < min_periods:
                if self.logger:
                    self.logger.warning(
                        "insufficient_data_for_correlation",
                        available_periods=len(combined_data),
                        required_periods=min_periods
                    )
                return None
                
            # Detect current regime
            regime = self._detect_regime(combined_data)
            
            # Use robust correlation estimation
            correlations = self._estimate_correlations(combined_data)
            
            # Create correlation matrix
            matrix = {}
            modules = list(historical_data.keys())
            for i, mod1 in enumerate(modules):
                matrix[mod1] = {}
                for j, mod2 in enumerate(modules):
                    matrix[mod1][mod2] = correlations[i, j]
                    
            # Calculate confidence based on sample size
            confidence = min(len(combined_data) / min_periods, 1.0)
            
            return CorrelationMatrix(
                matrix=matrix,
                last_updated=datetime.now(),
                regime=regime,
                confidence=confidence
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "correlation_update_error",
                    error=str(e)
                )
            raise
            
    def _prepare_data(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Prepare data for correlation analysis"""
        # Combine all module data
        dfs = []
        for module_id, df in historical_data.items():
            # Extract relevant columns and rename
            df = df.copy()
            df.columns = [f"{module_id}_{col}" for col in df.columns]
            dfs.append(df)
            
        combined = pd.concat(dfs, axis=1)
        
        # Handle missing data
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        # Standardize
        return pd.DataFrame(
            self._scaler.fit_transform(combined),
            columns=combined.columns,
            index=combined.index
        )
        
    def _detect_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        # Use rolling volatility and returns to cluster regimes
        vol = data.rolling(window=20).std().mean(axis=1)
        returns = data.diff().mean(axis=1)
        
        features = np.column_stack([vol, returns])
        features = features[~np.isnan(features).any(axis=1)]
        
        if len(features) == 0:
            return "normal"
            
        labels = self._regime_detector.fit_predict(features)
        current_regime = labels[-1]
        
        # Map numeric labels to descriptive regimes
        regime_map = {
            0: "normal",
            1: "high_volatility",
            2: "trending"
        }
        
        return regime_map.get(current_regime, "normal")
        
    def _estimate_correlations(self, data: pd.DataFrame) -> np.ndarray:
        """Estimate correlation matrix robustly"""
        # Use Minimum Covariance Determinant for robust estimation
        robust_cov = MinCovDet(random_state=42).fit(data)
        
        # Convert covariance to correlation
        cov = robust_cov.covariance_
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        
        # Ensure correlation matrix is valid
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1)
        
        return corr 