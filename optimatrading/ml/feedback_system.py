"""
Performance feedback and analysis system
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..logging import LoggerManager
from .models import PredictionRecord, ModulePerformance, MLConfig
from ..utils.performance import PerformanceMonitor

class FeedbackSystem:
    """
    Tracks and analyzes historical performance of modules.
    Provides visualization and insights for different market conditions.
    """
    
    def __init__(
        self,
        config: MLConfig,
        logger_manager: Optional[LoggerManager] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """
        Initialize feedback system.
        
        Args:
            config: ML configuration
            logger_manager: Logger manager instance
            performance_monitor: Performance monitoring instance
        """
        self.config = config
        self.logger = logger_manager.get_logger("FeedbackSystem") if logger_manager else None
        self.performance_monitor = performance_monitor
        
        self._predictions = []
        self._last_update = None
        
    @performance_monitor.monitor("ml") if performance_monitor else lambda x: x
    def record_prediction(
        self,
        module_id: str,
        prediction: Dict[str, float],
        actual: Dict[str, float],
        market_conditions: Dict[str, float]
    ) -> None:
        """
        Record a new prediction and its outcome.
        
        Args:
            module_id: ID of the module making prediction
            prediction: Predicted values
            actual: Actual values
            market_conditions: Market conditions at prediction time
        """
        try:
            # Calculate performance metrics
            metrics = self._calculate_metrics(prediction, actual)
            
            record = PredictionRecord(
                module_id=module_id,
                timestamp=datetime.now(),
                prediction=prediction,
                actual=actual,
                market_conditions=market_conditions,
                performance_metrics=metrics
            )
            
            self._predictions.append(record)
            self._last_update = datetime.now()
            
            if self.logger:
                self.logger.debug(
                    "prediction_recorded",
                    module_id=module_id,
                    metrics=metrics
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "prediction_recording_error",
                    error=str(e)
                )
            raise
            
    def calculate_performance(
        self,
        module_id: str,
        window: str = "medium"
    ) -> ModulePerformance:
        """
        Calculate performance metrics for a module.
        
        Args:
            module_id: ID of the module
            window: Time window for calculation
            
        Returns:
            Performance metrics
        """
        try:
            # Get predictions for window
            window_seconds = self.config.performance_windows[window]
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            
            window_predictions = [
                p for p in self._predictions
                if p.module_id == module_id and p.timestamp >= cutoff
            ]
            
            if not window_predictions:
                if self.logger:
                    self.logger.warning(
                        "no_predictions_in_window",
                        module_id=module_id,
                        window=window
                    )
                return None
                
            # Calculate aggregate metrics
            y_true = []
            y_pred = []
            
            for record in window_predictions:
                # Convert continuous predictions to binary for classification metrics
                for key in record.prediction:
                    pred = record.prediction[key] > 0
                    actual = record.actual[key] > 0
                    y_pred.append(pred)
                    y_true.append(actual)
                    
            return ModulePerformance(
                module_id=module_id,
                accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred)),
                recall=float(recall_score(y_true, y_pred)),
                f1_score=float(f1_score(y_true, y_pred)),
                timestamp=datetime.now(),
                window=window
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "performance_calculation_error",
                    error=str(e)
                )
            raise
            
    def analyze_market_conditions(
        self,
        module_id: str,
        window: str = "medium"
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze performance under different market conditions.
        
        Args:
            module_id: ID of the module
            window: Time window for analysis
            
        Returns:
            Analysis results by market condition
        """
        try:
            # Get predictions for window
            window_seconds = self.config.performance_windows[window]
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            
            window_predictions = [
                p for p in self._predictions
                if p.module_id == module_id and p.timestamp >= cutoff
            ]
            
            if not window_predictions:
                return {}
                
            # Convert to DataFrame
            records = []
            for pred in window_predictions:
                record = {
                    "timestamp": pred.timestamp,
                    **pred.market_conditions,
                    **pred.performance_metrics
                }
                records.append(record)
                
            df = pd.DataFrame(records)
            
            # Analyze by market condition
            results = {}
            for condition in pred.market_conditions.keys():
                # Split into quantiles
                quantiles = pd.qcut(df[condition], q=3, labels=["low", "medium", "high"])
                
                # Calculate metrics by quantile
                condition_analysis = df.groupby(quantiles).agg({
                    metric: ["mean", "std", "count"]
                    for metric in pred.performance_metrics.keys()
                })
                
                results[condition] = condition_analysis
                
            return results
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "market_condition_analysis_error",
                    error=str(e)
                )
            raise
            
    def create_performance_dashboard(
        self,
        module_ids: List[str],
        window: str = "medium"
    ) -> go.Figure:
        """
        Create interactive performance visualization.
        
        Args:
            module_ids: List of module IDs to include
            window: Time window for visualization
            
        Returns:
            Plotly figure with performance dashboard
        """
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Accuracy Over Time",
                    "Precision-Recall Trade-off",
                    "Performance by Market Volatility",
                    "Module Correlations"
                )
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            # Accuracy over time
            for i, module_id in enumerate(module_ids):
                module_predictions = [
                    p for p in self._predictions
                    if p.module_id == module_id
                ]
                
                if not module_predictions:
                    continue
                    
                df = pd.DataFrame([
                    {
                        "timestamp": p.timestamp,
                        **p.performance_metrics
                    }
                    for p in module_predictions
                ])
                
                fig.add_trace(
                    go.Scatter(
                        x=df["timestamp"],
                        y=df["accuracy"],
                        name=f"{module_id} Accuracy",
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=1, col=1
                )
                
                # Precision-Recall
                fig.add_trace(
                    go.Scatter(
                        x=[df["recall"].mean()],
                        y=[df["precision"].mean()],
                        name=module_id,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=colors[i % len(colors)]
                        )
                    ),
                    row=1, col=2
                )
                
                # Performance by volatility
                market_analysis = self.analyze_market_conditions(module_id, window)
                if "volatility" in market_analysis:
                    vol_data = market_analysis["volatility"]
                    fig.add_trace(
                        go.Box(
                            y=vol_data["accuracy"]["mean"],
                            name=module_id,
                            marker_color=colors[i % len(colors)]
                        ),
                        row=2, col=1
                    )
                    
            # Module correlations
            if len(module_ids) > 1:
                corr_matrix = np.zeros((len(module_ids), len(module_ids)))
                for i, mod1 in enumerate(module_ids):
                    for j, mod2 in enumerate(module_ids):
                        pred1 = pd.DataFrame([
                            p.performance_metrics for p in self._predictions
                            if p.module_id == mod1
                        ])
                        pred2 = pd.DataFrame([
                            p.performance_metrics for p in self._predictions
                            if p.module_id == mod2
                        ])
                        
                        if not pred1.empty and not pred2.empty:
                            corr_matrix[i, j] = pred1["accuracy"].corr(pred2["accuracy"])
                            
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix,
                        x=module_ids,
                        y=module_ids,
                        colorscale="RdBu"
                    ),
                    row=2, col=2
                )
                
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Module Performance Dashboard"
            )
            
            return fig
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "dashboard_creation_error",
                    error=str(e)
                )
            raise
            
    def _calculate_metrics(
        self,
        prediction: Dict[str, float],
        actual: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate performance metrics for a prediction"""
        metrics = {}
        
        # Calculate error metrics
        errors = []
        for key in prediction:
            if key in actual:
                error = abs(prediction[key] - actual[key])
                errors.append(error)
                
        if errors:
            metrics["mean_absolute_error"] = float(np.mean(errors))
            metrics["max_error"] = float(np.max(errors))
            
        # Calculate directional accuracy
        correct_direction = 0
        total_predictions = 0
        
        for key in prediction:
            if key in actual:
                pred_direction = prediction[key] > 0
                actual_direction = actual[key] > 0
                if pred_direction == actual_direction:
                    correct_direction += 1
                total_predictions += 1
                
        if total_predictions > 0:
            metrics["directional_accuracy"] = correct_direction / total_predictions
            
        return metrics 