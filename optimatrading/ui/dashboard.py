"""
Advanced dashboard system
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

from .models import DashboardLayout, ChartConfig, AlertConfig, Annotation
from .charts import ChartManager
from .explainer import Explainer
from ..logging import LoggerManager

class Dashboard:
    """
    Advanced dashboard system with customizable layouts
    and interactive features.
    """
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None,
        chart_manager: Optional[ChartManager] = None,
        explainer: Optional[Explainer] = None
    ):
        """
        Initialize dashboard.
        
        Args:
            logger_manager: Logger manager instance
            chart_manager: Chart manager instance
            explainer: Explainer instance
        """
        self.logger = logger_manager.get_logger("Dashboard") if logger_manager else None
        self.chart_manager = chart_manager or ChartManager(logger_manager)
        self.explainer = explainer or Explainer(logger_manager, chart_manager)
        
        # Default components
        self.default_components = {
            "price_chart": self._create_price_chart,
            "performance_metrics": self._create_performance_metrics,
            "module_correlations": self._create_module_correlations,
            "alerts": self._create_alerts_panel,
            "annotations": self._create_annotations_panel
        }
        
    def create_dashboard(
        self,
        layout: DashboardLayout,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> Dict[str, Union[go.Figure, Dict]]:
        """
        Create complete dashboard based on layout.
        
        Args:
            layout: Dashboard layout configuration
            market_data: Market data by symbol
            performance_data: Performance metrics by module
            alerts: Active alerts
            annotations: User annotations
            
        Returns:
            Dictionary of dashboard components
        """
        try:
            components = {}
            
            # Create each component based on layout
            for component in layout.components:
                if component not in self.default_components:
                    if self.logger:
                        self.logger.warning(
                            "unknown_component",
                            component=component
                        )
                    continue
                    
                # Get component position
                position = layout.positions.get(component, {
                    "row": 1,
                    "col": 1,
                    "rowspan": 1,
                    "colspan": 1
                })
                
                # Create component
                creator = self.default_components[component]
                components[component] = {
                    "content": creator(
                        market_data,
                        performance_data,
                        alerts,
                        annotations
                    ),
                    "position": position
                }
                
            return components
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "dashboard_creation_error",
                    error=str(e)
                )
            raise
            
    def _create_price_chart(
        self,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> go.Figure:
        """Create main price chart with indicators"""
        # Use first symbol as default
        symbol = next(iter(market_data))
        data = market_data[symbol]
        
        # Create chart config
        config = ChartConfig(
            chart_type="candlestick",
            indicators=["SMA", "RSI", "MACD"],
            show_volume=True
        )
        
        # Get relevant annotations
        chart_annotations = [
            ann for ann in annotations
            if ann.chart_id == symbol
        ]
        
        return self.chart_manager.create_chart(
            data,
            config,
            chart_annotations
        )
        
    def _create_performance_metrics(
        self,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> go.Figure:
        """Create performance metrics panel"""
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Accuracy Over Time",
                "Returns Distribution",
                "Risk Metrics",
                "Module Performance"
            )
        )
        
        # Add accuracy plot
        accuracies = []
        timestamps = []
        for module, data in performance_data.items():
            if "accuracy" in data:
                accuracies.append(data["accuracy"])
                timestamps.append(data.get("timestamp", datetime.now()))
                
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracies,
                name="Accuracy"
            ),
            row=1, col=1
        )
        
        # Add returns distribution
        returns = []
        for data in market_data.values():
            if "returns" in data:
                returns.extend(data["returns"].tolist())
                
        fig.add_trace(
            go.Histogram(
                x=returns,
                name="Returns"
            ),
            row=1, col=2
        )
        
        # Add risk metrics
        risk_metrics = ["sharpe", "sortino", "max_drawdown"]
        values = []
        for metric in risk_metrics:
            values.append(
                np.mean([
                    data.get(metric, 0)
                    for data in performance_data.values()
                ])
            )
            
        fig.add_trace(
            go.Bar(
                x=risk_metrics,
                y=values,
                name="Risk Metrics"
            ),
            row=2, col=1
        )
        
        # Add module performance comparison
        modules = list(performance_data.keys())
        performance = [
            performance_data[m].get("performance", 0)
            for m in modules
        ]
        
        fig.add_trace(
            go.Bar(
                x=modules,
                y=performance,
                name="Module Performance"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark"
        )
        
        return fig
        
    def _create_module_correlations(
        self,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> go.Figure:
        """Create module correlation heatmap"""
        # Extract correlation data
        modules = list(performance_data.keys())
        correlations = pd.DataFrame(
            index=modules,
            columns=modules,
            dtype=float
        )
        
        for i, mod1 in enumerate(modules):
            for j, mod2 in enumerate(modules):
                if i == j:
                    correlations.iloc[i, j] = 1.0
                else:
                    # Calculate correlation between module outputs
                    corr = performance_data[mod1].get(
                        "correlations", {}
                    ).get(mod2, 0)
                    correlations.iloc[i, j] = corr
                    
        # Create heatmap
        config = ChartConfig(
            height=400,
            width=600,
            theme="dark"
        )
        
        return self.chart_manager.create_correlation_heatmap(
            correlations,
            config
        )
        
    def _create_alerts_panel(
        self,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> Dict[str, Union[str, List]]:
        """Create alerts panel"""
        active_alerts = []
        
        for alert in alerts:
            if not alert.enabled:
                continue
                
            # Check if alert condition is met
            for symbol, data in market_data.items():
                if alert.metric in data.columns:
                    current_value = data[alert.metric].iloc[-1]
                    
                    if alert.condition == "above" and current_value > alert.threshold:
                        active_alerts.append({
                            "id": alert.id,
                            "message": f"{symbol} {alert.metric} above {alert.threshold}",
                            "timestamp": datetime.now(),
                            "severity": "high"
                        })
                    elif alert.condition == "below" and current_value < alert.threshold:
                        active_alerts.append({
                            "id": alert.id,
                            "message": f"{symbol} {alert.metric} below {alert.threshold}",
                            "timestamp": datetime.now(),
                            "severity": "high"
                        })
                        
        return {
            "type": "alerts",
            "data": active_alerts
        }
        
    def _create_annotations_panel(
        self,
        market_data: Dict[str, pd.DataFrame],
        performance_data: Dict[str, Dict],
        alerts: List[AlertConfig],
        annotations: List[Annotation]
    ) -> Dict[str, Union[str, List]]:
        """Create annotations panel"""
        # Group annotations by category
        grouped = {}
        for ann in annotations:
            if ann.category not in grouped:
                grouped[ann.category] = []
            grouped[ann.category].append({
                "text": ann.text,
                "timestamp": ann.timestamp,
                "tags": ann.tags
            })
            
        return {
            "type": "annotations",
            "data": grouped
        } 