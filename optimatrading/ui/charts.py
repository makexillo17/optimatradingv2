"""
Advanced charting components
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import talib
from dataclasses import dataclass

from .models import ChartConfig
from ..logging import LoggerManager

@dataclass
class IndicatorSpec:
    """Technical indicator specification"""
    name: str
    function: callable
    params: Dict[str, Union[int, float, str]]
    subplot: bool = False
    color: str = "#2196F3"

class ChartManager:
    """
    Advanced chart manager with technical indicators
    and interactive features.
    """
    
    # Default technical indicators
    INDICATORS = {
        "SMA": IndicatorSpec(
            name="Simple Moving Average",
            function=talib.SMA,
            params={"timeperiod": 20}
        ),
        "EMA": IndicatorSpec(
            name="Exponential Moving Average",
            function=talib.EMA,
            params={"timeperiod": 20}
        ),
        "BBANDS": IndicatorSpec(
            name="Bollinger Bands",
            function=talib.BBANDS,
            params={
                "timeperiod": 20,
                "nbdevup": 2,
                "nbdevdn": 2
            }
        ),
        "RSI": IndicatorSpec(
            name="Relative Strength Index",
            function=talib.RSI,
            params={"timeperiod": 14},
            subplot=True
        ),
        "MACD": IndicatorSpec(
            name="MACD",
            function=talib.MACD,
            params={
                "fastperiod": 12,
                "slowperiod": 26,
                "signalperiod": 9
            },
            subplot=True
        )
    }
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None
    ):
        """
        Initialize chart manager.
        
        Args:
            logger_manager: Logger manager instance
        """
        self.logger = logger_manager.get_logger("ChartManager") if logger_manager else None
        
    def create_chart(
        self,
        data: pd.DataFrame,
        config: ChartConfig,
        annotations: Optional[List[Dict]] = None
    ) -> go.Figure:
        """
        Create interactive chart with indicators.
        
        Args:
            data: OHLCV data
            config: Chart configuration
            annotations: Optional chart annotations
            
        Returns:
            Plotly figure
        """
        try:
            # Calculate number of subplots needed
            subplot_indicators = [
                ind for ind in config.indicators
                if self.INDICATORS[ind].subplot
            ]
            n_subplots = 1 + len(subplot_indicators)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=n_subplots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7] + [0.3] * (n_subplots - 1)
            )
            
            # Add main chart
            self._add_main_chart(fig, data, config)
            
            # Add volume if requested
            if config.show_volume:
                self._add_volume(fig, data)
                
            # Add indicators
            self._add_indicators(fig, data, config.indicators)
            
            # Add annotations if provided
            if annotations:
                self._add_annotations(fig, annotations)
                
            # Update layout
            self._update_layout(fig, config)
            
            return fig
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "chart_creation_error",
                    error=str(e)
                )
            raise
            
    def create_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        config: ChartConfig
    ) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale="RdBu",
            zmid=0
        ))
        
        fig.update_layout(
            title="Module Correlations",
            height=config.height,
            width=config.width,
            template=f"plotly_{config.theme}"
        )
        
        return fig
        
    def create_contribution_chart(
        self,
        contributions: Dict[str, float],
        config: ChartConfig
    ) -> go.Figure:
        """
        Create module contribution waterfall chart.
        
        Args:
            contributions: Module contributions
            config: Chart configuration
            
        Returns:
            Plotly figure
        """
        # Sort contributions
        sorted_items = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="Module Contributions",
            orientation="v",
            measure=["relative"] * len(sorted_items),
            x=[item[0] for item in sorted_items],
            y=[item[1] for item in sorted_items],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        
        fig.update_layout(
            title="Module Contributions to Final Decision",
            showlegend=True,
            height=config.height,
            width=config.width,
            template=f"plotly_{config.theme}"
        )
        
        return fig
        
    def _add_main_chart(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        config: ChartConfig
    ) -> None:
        """Add main price chart"""
        if config.chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="OHLC"
                ),
                row=1, col=1
            )
        elif config.chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["close"],
                    name="Price",
                    line=dict(color="#2196F3")
                ),
                row=1, col=1
            )
        elif config.chart_type == "ohlc":
            fig.add_trace(
                go.Ohlc(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="OHLC"
                ),
                row=1, col=1
            )
            
    def _add_volume(
        self,
        fig: go.Figure,
        data: pd.DataFrame
    ) -> None:
        """Add volume bars"""
        colors = np.where(
            data["close"] > data["open"],
            "#26A69A",  # green
            "#EF5350"   # red
        )
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5
            ),
            row=1, col=1,
            secondary_y=True
        )
        
    def _add_indicators(
        self,
        fig: go.Figure,
        data: pd.DataFrame,
        indicators: List[str]
    ) -> None:
        """Add technical indicators"""
        current_subplot = 2
        
        for indicator in indicators:
            if indicator not in self.INDICATORS:
                continue
                
            spec = self.INDICATORS[indicator]
            
            try:
                if indicator == "BBANDS":
                    upper, middle, lower = spec.function(
                        data["close"],
                        **spec.params
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=upper,
                            name="BB Upper",
                            line=dict(dash="dash")
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=middle,
                            name="BB Middle"
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=lower,
                            name="BB Lower",
                            line=dict(dash="dash")
                        ),
                        row=1, col=1
                    )
                    
                elif indicator == "MACD":
                    macd, signal, hist = spec.function(
                        data["close"],
                        **spec.params
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=macd,
                            name="MACD"
                        ),
                        row=current_subplot, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=signal,
                            name="Signal"
                        ),
                        row=current_subplot, col=1
                    )
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=hist,
                            name="Histogram"
                        ),
                        row=current_subplot, col=1
                    )
                    
                    current_subplot += 1
                    
                else:
                    values = spec.function(
                        data["close"],
                        **spec.params
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=values,
                            name=indicator
                        ),
                        row=1 if not spec.subplot else current_subplot,
                        col=1
                    )
                    
                    if spec.subplot:
                        current_subplot += 1
                        
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        "indicator_calculation_error",
                        indicator=indicator,
                        error=str(e)
                    )
                    
    def _add_annotations(
        self,
        fig: go.Figure,
        annotations: List[Dict]
    ) -> None:
        """Add user annotations to chart"""
        for ann in annotations:
            fig.add_annotation(
                x=ann["timestamp"],
                y=ann.get("coordinates", {}).get("y", 0),
                text=ann["text"],
                showarrow=True,
                arrowhead=1
            )
            
    def _update_layout(
        self,
        fig: go.Figure,
        config: ChartConfig
    ) -> None:
        """Update figure layout"""
        fig.update_layout(
            height=config.height,
            width=config.width,
            template=f"plotly_{config.theme}",
            xaxis_rangeslider_visible=False,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Make it more interactive
        fig.update_layout(
            dragmode="zoom",
            hovermode="x unified",
            hoverdistance=100,
            spikedistance=1000
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ) 