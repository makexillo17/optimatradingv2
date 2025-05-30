"""
Market anomaly detection system
"""

import numpy as np
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ..logging import LoggerManager
from .models import MarketAnomaly, MLConfig
from ..utils.performance import PerformanceMonitor

class AnomalyDetector:
    """
    Detects market anomalies and suggests parameter adjustments.
    Uses multiple detection methods and adaptive thresholds.
    """
    
    def __init__(
        self,
        config: MLConfig,
        logger_manager: Optional[LoggerManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize anomaly detector.
        
        Args:
            config: ML configuration
            logger_manager: Logger manager instance
            performance_monitor: Performance monitoring instance
        """
        self.config = config
        self.logger = logger_manager.get_logger("AnomalyDetector") if logger_manager else None
        self.performance_monitor = performance_monitor
        
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=0.95)  # Capture 95% variance
        self._isolation_forest = IsolationForest(
            contamination=1 - config.anomaly_detection_sensitivity,
            random_state=42
        )
        
        self._historical_anomalies = []
        self._last_update = None
        
    @performance_monitor.monitor("ml") if performance_monitor else lambda x: x
    def detect_anomalies(
        self,
        market_data: Dict[str, pd.DataFrame],
        lookback_window: int = 100
    ) -> List[MarketAnomaly]:
        """
        Detect anomalies in market data.
        
        Args:
            market_data: Dictionary mapping metrics to their historical values
            lookback_window: Number of periods to analyze
            
        Returns:
            List of detected anomalies
        """
        try:
            # Prepare data
            combined_data = self._prepare_data(market_data, lookback_window)
            
            if len(combined_data) < lookback_window:
                if self.logger:
                    self.logger.warning(
                        "insufficient_data_for_anomaly_detection",
                        available_periods=len(combined_data),
                        required_periods=lookback_window
                    )
                return []
                
            # Detect anomalies using multiple methods
            anomalies = []
            
            # Statistical anomalies
            stat_anomalies = self._detect_statistical_anomalies(combined_data)
            anomalies.extend(stat_anomalies)
            
            # Isolation Forest anomalies
            if_anomalies = self._detect_isolation_forest_anomalies(combined_data)
            anomalies.extend(if_anomalies)
            
            # PCA reconstruction anomalies
            pca_anomalies = self._detect_pca_anomalies(combined_data)
            anomalies.extend(pca_anomalies)
            
            # Merge similar anomalies
            merged = self._merge_anomalies(anomalies)
            
            # Update historical anomalies
            self._historical_anomalies.extend(merged)
            self._last_update = datetime.now()
            
            if self.logger:
                self.logger.info(
                    "anomalies_detected",
                    count=len(merged),
                    anomalies=merged
                )
                
            return merged
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "anomaly_detection_error",
                    error=str(e)
                )
            raise
            
    def suggest_parameter_adjustments(
        self,
        anomalies: List[MarketAnomaly]
    ) -> Dict[str, Dict[str, float]]:
        """
        Suggest parameter adjustments based on detected anomalies.
        
        Args:
            anomalies: List of detected anomalies
            
        Returns:
            Dictionary mapping parameters to suggested adjustments
        """
        adjustments = {}
        
        for anomaly in anomalies:
            # Get base adjustment factor from severity
            factor = self._calculate_adjustment_factor(anomaly.severity)
            
            for metric in anomaly.affected_metrics:
                if metric not in adjustments:
                    adjustments[metric] = {}
                    
                # Adjust parameters based on anomaly type
                if anomaly.anomaly_type == "volatility_spike":
                    adjustments[metric]["smoothing_window"] = factor * 1.5
                    adjustments[metric]["outlier_threshold"] = factor * 2.0
                    
                elif anomaly.anomaly_type == "trend_break":
                    adjustments[metric]["momentum_factor"] = 1.0 / factor
                    adjustments[metric]["mean_reversion_strength"] = factor
                    
                elif anomaly.anomaly_type == "correlation_breakdown":
                    adjustments[metric]["correlation_window"] = factor * 2.0
                    adjustments[metric]["diversification_weight"] = factor
                    
        return adjustments
        
    def _prepare_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        lookback_window: int
    ) -> pd.DataFrame:
        """Prepare data for anomaly detection"""
        # Combine all metrics
        dfs = []
        for metric, df in market_data.items():
            # Extract relevant columns and rename
            df = df.copy()
            df.columns = [f"{metric}_{col}" for col in df.columns]
            dfs.append(df)
            
        combined = pd.concat(dfs, axis=1)
        
        # Handle missing data
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        # Take recent window
        if len(combined) > lookback_window:
            combined = combined.iloc[-lookback_window:]
            
        # Standardize
        return pd.DataFrame(
            self._scaler.fit_transform(combined),
            columns=combined.columns,
            index=combined.index
        )
        
    def _detect_statistical_anomalies(
        self,
        data: pd.DataFrame
    ) -> List[MarketAnomaly]:
        """Detect anomalies using statistical methods"""
        anomalies = []
        
        # Calculate rolling statistics
        roll_mean = data.rolling(window=20).mean()
        roll_std = data.rolling(window=20).std()
        
        # Z-score based detection
        z_scores = (data - roll_mean) / roll_std
        
        for col in data.columns:
            # Find extreme values
            extreme_mask = abs(z_scores[col]) > 3
            if not extreme_mask.any():
                continue
                
            # Group consecutive anomalies
            anomaly_periods = self._group_consecutive(extreme_mask)
            
            for start_idx, end_idx in anomaly_periods:
                severity = float(abs(z_scores[col][start_idx:end_idx]).mean())
                
                anomalies.append(MarketAnomaly(
                    timestamp=data.index[start_idx],
                    anomaly_type="volatility_spike",
                    severity=min(severity / 6.0, 1.0),  # Normalize to [0,1]
                    affected_metrics=[col.split('_')[0]],
                    description=f"Extreme values detected in {col}",
                    suggested_actions=[
                        "Increase smoothing window",
                        "Adjust outlier thresholds"
                    ]
                ))
                
        return anomalies
        
    def _detect_isolation_forest_anomalies(
        self,
        data: pd.DataFrame
    ) -> List[MarketAnomaly]:
        """Detect anomalies using Isolation Forest"""
        # Fit and predict
        predictions = self._isolation_forest.fit_predict(data)
        
        # Find anomalies (-1 indicates anomaly)
        anomaly_mask = predictions == -1
        if not anomaly_mask.any():
            return []
            
        anomalies = []
        anomaly_periods = self._group_consecutive(anomaly_mask)
        
        for start_idx, end_idx in anomaly_periods:
            # Calculate contribution of each feature
            period_data = data.iloc[start_idx:end_idx]
            feature_scores = abs(period_data).mean()
            
            # Get most anomalous features
            top_features = feature_scores.nlargest(3)
            affected_metrics = {f.split('_')[0] for f in top_features.index}
            
            anomalies.append(MarketAnomaly(
                timestamp=data.index[start_idx],
                anomaly_type="correlation_breakdown",
                severity=min(len(affected_metrics) / len(data.columns), 1.0),
                affected_metrics=list(affected_metrics),
                description="Unusual feature interactions detected",
                suggested_actions=[
                    "Review correlation assumptions",
                    "Adjust diversification weights"
                ]
            ))
            
        return anomalies
        
    def _detect_pca_anomalies(
        self,
        data: pd.DataFrame
    ) -> List[MarketAnomaly]:
        """Detect anomalies using PCA reconstruction"""
        # Fit PCA and transform
        transformed = self._pca.fit_transform(data)
        reconstructed = self._pca.inverse_transform(transformed)
        
        # Calculate reconstruction error
        error = np.mean((data - reconstructed) ** 2, axis=1)
        threshold = np.mean(error) + 2 * np.std(error)
        
        anomaly_mask = error > threshold
        if not anomaly_mask.any():
            return []
            
        anomalies = []
        anomaly_periods = self._group_consecutive(anomaly_mask)
        
        for start_idx, end_idx in anomaly_periods:
            # Calculate feature contributions to error
            period_error = (
                data.iloc[start_idx:end_idx] -
                reconstructed[start_idx:end_idx]
            ) ** 2
            feature_errors = period_error.mean()
            
            # Get most affected features
            top_features = feature_errors.nlargest(3)
            affected_metrics = {f.split('_')[0] for f in top_features.index}
            
            anomalies.append(MarketAnomaly(
                timestamp=data.index[start_idx],
                anomaly_type="trend_break",
                severity=min(error[start_idx] / threshold, 1.0),
                affected_metrics=list(affected_metrics),
                description="Structural break in market relationships",
                suggested_actions=[
                    "Adjust trend following parameters",
                    "Review mean reversion assumptions"
                ]
            ))
            
        return anomalies
        
    def _merge_anomalies(
        self,
        anomalies: List[MarketAnomaly]
    ) -> List[MarketAnomaly]:
        """Merge similar anomalies"""
        if not anomalies:
            return []
            
        # Sort by timestamp
        sorted_anomalies = sorted(anomalies, key=lambda x: x.timestamp)
        merged = []
        current = sorted_anomalies[0]
        
        for anomaly in sorted_anomalies[1:]:
            # Check if anomalies are close in time (within 1 hour)
            if (anomaly.timestamp - current.timestamp).total_seconds() <= 3600:
                # Merge if same type or overlapping metrics
                if (anomaly.anomaly_type == current.anomaly_type or
                    set(anomaly.affected_metrics) & set(current.affected_metrics)):
                    # Update current anomaly
                    current.severity = max(current.severity, anomaly.severity)
                    current.affected_metrics = list(set(
                        current.affected_metrics + anomaly.affected_metrics
                    ))
                    current.suggested_actions = list(set(
                        current.suggested_actions + anomaly.suggested_actions
                    ))
                    continue
                    
            merged.append(current)
            current = anomaly
            
        merged.append(current)
        return merged
        
    def _group_consecutive(self, mask: pd.Series) -> List[Tuple[int, int]]:
        """Group consecutive True values in mask"""
        if not mask.any():
            return []
            
        # Find changes in mask
        changes = mask.astype(int).diff()
        
        # Get start and end indices
        starts = changes[changes == 1].index
        ends = changes[changes == -1].index
        
        # Handle edge cases
        if mask.iloc[0]:
            starts = pd.Index([mask.index[0]]).append(starts)
        if mask.iloc[-1]:
            ends = ends.append(pd.Index([mask.index[-1]]))
            
        return list(zip(starts, ends))
        
    def _calculate_adjustment_factor(self, severity: float) -> float:
        """Calculate parameter adjustment factor based on severity"""
        # Non-linear scaling of adjustment factor
        return 1.0 + np.tanh(2 * severity) 