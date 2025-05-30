from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from logging.logger_manager import LoggerManager
from cache.cache_manager import CacheManager

class PerformanceMetrics:
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        self.logger = logger_manager.get_logger("PerformanceMetrics") if logger_manager else None
        self.cache = cache_manager
        
    def calculate_trading_metrics(
        self,
        recommendations: List[Dict[str, Any]],
        actual_returns: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """
        Calcula métricas de rendimiento del trading
        
        Args:
            recommendations: Lista de recomendaciones generadas
            actual_returns: Retornos reales observados
            timestamps: Timestamps de las recomendaciones
            
        Returns:
            Diccionario con métricas calculadas
        """
        try:
            df = pd.DataFrame({
                'timestamp': timestamps,
                'recommendation': [r['recommendation'] for r in recommendations],
                'confidence': [r['confidence'] for r in recommendations],
                'actual_return': actual_returns
            })
            
            # Calcular métricas básicas
            metrics = {
                'total_recommendations': len(recommendations),
                'accuracy': self._calculate_accuracy(df),
                'sharpe_ratio': self._calculate_sharpe_ratio(df),
                'max_drawdown': self._calculate_max_drawdown(df),
                'win_rate': self._calculate_win_rate(df),
                'profit_factor': self._calculate_profit_factor(df),
                'avg_return': np.mean(actual_returns),
                'std_return': np.std(actual_returns)
            }
            
            # Calcular métricas por tipo de señal
            metrics.update(self._calculate_signal_metrics(df))
            
            # Calcular métricas de confianza
            metrics.update(self._calculate_confidence_metrics(df))
            
            # Almacenar métricas en caché si está disponible
            if self.cache:
                self.cache.set(
                    f"metrics:{datetime.now():%Y%m%d}",
                    metrics,
                    namespace="performance"
                )
                
            if self.logger:
                self.logger.info("metrics_calculated", metrics=metrics)
                
            return metrics
            
        except Exception as e:
            if self.logger:
                self.logger.error("metrics_calculation_error", error=str(e))
            return {}
            
    def _calculate_accuracy(self, df: pd.DataFrame) -> float:
        """Calcula la precisión de las recomendaciones"""
        correct_predictions = (
            (df['recommendation'] == 'LONG') & (df['actual_return'] > 0) |
            (df['recommendation'] == 'SHORT') & (df['actual_return'] < 0) |
            (df['recommendation'] == 'NEUTRAL') & (abs(df['actual_return']) < 0.001)
        )
        return float(correct_predictions.mean())
        
    def _calculate_sharpe_ratio(
        self,
        df: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> float:
        """Calcula el Sharpe Ratio"""
        returns = df['actual_return']
        excess_returns = returns - (risk_free_rate / 252)  # Diario
        if len(excess_returns) < 2:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252))
        
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calcula el máximo drawdown"""
        cumulative_returns = (1 + df['actual_return']).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        return float(drawdowns.min())
        
    def _calculate_win_rate(self, df: pd.DataFrame) -> float:
        """Calcula el ratio de operaciones ganadoras"""
        winning_trades = (
            (df['recommendation'] == 'LONG') & (df['actual_return'] > 0) |
            (df['recommendation'] == 'SHORT') & (df['actual_return'] < 0)
        )
        total_trades = (df['recommendation'] != 'NEUTRAL').sum()
        return float(winning_trades.sum() / total_trades) if total_trades > 0 else 0.0
        
    def _calculate_profit_factor(self, df: pd.DataFrame) -> float:
        """Calcula el factor de beneficio"""
        profits = df[df['actual_return'] > 0]['actual_return'].sum()
        losses = abs(df[df['actual_return'] < 0]['actual_return'].sum())
        return float(profits / losses) if losses != 0 else float('inf')
        
    def _calculate_signal_metrics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calcula métricas por tipo de señal"""
        signal_metrics = {}
        
        for signal in ['LONG', 'SHORT', 'NEUTRAL']:
            signal_df = df[df['recommendation'] == signal]
            if len(signal_df) == 0:
                continue
                
            signal_metrics[f"{signal.lower()}_metrics"] = {
                'count': len(signal_df),
                'avg_return': float(signal_df['actual_return'].mean()),
                'std_return': float(signal_df['actual_return'].std()),
                'accuracy': self._calculate_accuracy(signal_df),
                'avg_confidence': float(signal_df['confidence'].mean())
            }
            
        return signal_metrics
        
    def _calculate_confidence_metrics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Calcula métricas por nivel de confianza"""
        confidence_metrics = {}
        
        # Dividir en cuartiles de confianza
        df['confidence_quartile'] = pd.qcut(df['confidence'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            quartile_df = df[df['confidence_quartile'] == quartile]
            if len(quartile_df) == 0:
                continue
                
            confidence_metrics[f"confidence_{quartile.lower()}"] = {
                'count': len(quartile_df),
                'avg_return': float(quartile_df['actual_return'].mean()),
                'accuracy': self._calculate_accuracy(quartile_df),
                'avg_confidence': float(quartile_df['confidence'].mean())
            }
            
        return confidence_metrics
        
    def calculate_module_metrics(
        self,
        module_results: Dict[str, List[Dict[str, Any]]],
        actual_returns: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas por módulo
        
        Args:
            module_results: Resultados por módulo
            actual_returns: Retornos reales observados
            
        Returns:
            Métricas por módulo
        """
        try:
            module_metrics = {}
            
            for module_name, results in module_results.items():
                if not results:
                    continue
                    
                df = pd.DataFrame({
                    'recommendation': [r['recommendation'] for r in results],
                    'confidence': [r['confidence'] for r in results],
                    'actual_return': actual_returns[:len(results)]
                })
                
                module_metrics[module_name] = {
                    'accuracy': self._calculate_accuracy(df),
                    'avg_confidence': float(df['confidence'].mean()),
                    'contribution': self._calculate_module_contribution(
                        results,
                        actual_returns
                    )
                }
                
            if self.logger:
                self.logger.info("module_metrics_calculated", metrics=module_metrics)
                
            return module_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.error("module_metrics_calculation_error", error=str(e))
            return {}
            
    def _calculate_module_contribution(
        self,
        module_results: List[Dict[str, Any]],
        actual_returns: List[float]
    ) -> float:
        """Calcula la contribución de un módulo al rendimiento general"""
        if not module_results or not actual_returns:
            return 0.0
            
        # Correlación entre confianza y retornos absolutos
        confidences = [r['confidence'] for r in module_results]
        abs_returns = np.abs(actual_returns[:len(module_results)])
        
        if len(confidences) < 2:
            return 0.0
            
        return float(np.corrcoef(confidences, abs_returns)[0, 1])
        
    def get_historical_metrics(
        self,
        days: int = 30
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Obtiene métricas históricas
        
        Args:
            days: Número de días a recuperar
            
        Returns:
            Métricas históricas por día
        """
        try:
            if not self.cache:
                return {}
                
            historical_metrics = []
            start_date = datetime.now() - timedelta(days=days)
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                metrics = self.cache.get(
                    f"metrics:{date:%Y%m%d}",
                    namespace="performance"
                )
                
                if metrics:
                    historical_metrics.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'metrics': metrics
                    })
                    
            return {'historical_metrics': historical_metrics}
            
        except Exception as e:
            if self.logger:
                self.logger.error("historical_metrics_error", error=str(e))
            return {} 