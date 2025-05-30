"""
Advanced explanation system
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import shap

from .models import ExplanationConfig
from ..logging import LoggerManager
from .charts import ChartManager

class Explainer:
    """
    Advanced explanation system for trading decisions.
    Uses SHAP values and decision trees for interpretability.
    """
    
    def __init__(
        self,
        logger_manager: Optional[LoggerManager] = None,
        chart_manager: Optional[ChartManager] = None
    ):
        """
        Initialize explainer.
        
        Args:
            logger_manager: Logger manager instance
            chart_manager: Chart manager instance
        """
        self.logger = logger_manager.get_logger("Explainer") if logger_manager else None
        self.chart_manager = chart_manager or ChartManager(logger_manager)
        self._scaler = StandardScaler()
        
    def explain_decision(
        self,
        features: Dict[str, float],
        model_outputs: Dict[str, float],
        historical_data: pd.DataFrame,
        config: ExplanationConfig
    ) -> Dict[str, Union[str, Dict, List]]:
        """
        Generate comprehensive explanation for a trading decision.
        
        Args:
            features: Current feature values
            model_outputs: Model outputs/predictions
            historical_data: Historical feature data
            config: Explanation configuration
            
        Returns:
            Dictionary containing explanation components
        """
        try:
            # Convert features to DataFrame
            current_features = pd.DataFrame([features])
            
            # Calculate feature importance
            importance = self._calculate_importance(
                historical_data,
                model_outputs["final_decision"]
            )
            
            # Generate text explanation
            text_explanation = self._generate_text_explanation(
                features,
                model_outputs,
                importance,
                config
            )
            
            # Generate counterfactuals if requested
            counterfactuals = None
            if config.include_counterfactuals:
                counterfactuals = self._generate_counterfactuals(
                    current_features,
                    historical_data,
                    model_outputs
                )
                
            # Create visual explanations if requested
            visuals = None
            if config.show_visuals:
                visuals = self._create_visual_explanations(
                    features,
                    importance,
                    historical_data
                )
                
            return {
                "text_explanation": text_explanation,
                "feature_importance": importance,
                "counterfactuals": counterfactuals,
                "visuals": visuals
            }
            
        except Exception as e:
            if self.logger:
                self.logger.error(
                    "explanation_generation_error",
                    error=str(e)
                )
            raise
            
    def _calculate_importance(
        self,
        historical_data: pd.DataFrame,
        target: float,
        n_samples: int = 1000
    ) -> Dict[str, float]:
        """Calculate feature importance using SHAP values"""
        try:
            # Prepare data
            X = self._scaler.fit_transform(historical_data)
            
            # Train simple model for SHAP
            model = DecisionTreeRegressor(max_depth=5)
            model.fit(X, [target] * len(X))
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Convert to importance scores
            importance = {}
            for i, col in enumerate(historical_data.columns):
                importance[col] = float(np.abs(shap_values[:, i]).mean())
                
            # Normalize to percentages
            total = sum(importance.values())
            importance = {
                k: v / total * 100
                for k, v in importance.items()
            }
            
            return importance
            
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "importance_calculation_error",
                    error=str(e)
                )
            return {}
            
    def _generate_text_explanation(
        self,
        features: Dict[str, float],
        outputs: Dict[str, float],
        importance: Dict[str, float],
        config: ExplanationConfig
    ) -> str:
        """Generate text explanation at appropriate detail level"""
        try:
            # Get top factors
            top_factors = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:config.max_factors]
            
            # Basic explanation
            if config.detail_level == "basic":
                explanation = (
                    f"The system {'recommends' if outputs['final_decision'] > 0 else 'does not recommend'} "
                    f"taking a position, with a confidence of {abs(outputs['final_decision'])*100:.1f}%.\n\n"
                    f"The main factors influencing this decision are:\n"
                )
                
                for factor, importance in top_factors:
                    explanation += f"- {factor}: {importance:.1f}% importance\n"
                    
            # Medium detail
            elif config.detail_level == "medium":
                explanation = (
                    f"Decision Analysis:\n"
                    f"{'Positive' if outputs['final_decision'] > 0 else 'Negative'} recommendation "
                    f"with {abs(outputs['final_decision'])*100:.1f}% confidence\n\n"
                    f"Key Factors:\n"
                )
                
                for factor, importance in top_factors:
                    value = features.get(factor, "N/A")
                    explanation += (
                        f"- {factor}:\n"
                        f"  * Current value: {value}\n"
                        f"  * Importance: {importance:.1f}%\n"
                    )
                    
            # Technical detail
            else:
                explanation = (
                    f"Technical Analysis Report:\n"
                    f"Decision: {outputs['final_decision']:.4f}\n"
                    f"Confidence: {abs(outputs['final_decision'])*100:.1f}%\n\n"
                    f"Factor Analysis:\n"
                )
                
                for factor, importance in top_factors:
                    value = features.get(factor, "N/A")
                    normalized = self._scaler.transform([[value]])[0][0] if value != "N/A" else "N/A"
                    explanation += (
                        f"- {factor}:\n"
                        f"  * Raw value: {value}\n"
                        f"  * Normalized value: {normalized:.4f}\n"
                        f"  * Importance: {importance:.1f}%\n"
                        f"  * Impact direction: {'positive' if value > 0 else 'negative'}\n"
                    )
                    
            return explanation
            
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "text_explanation_error",
                    error=str(e)
                )
            return "Could not generate detailed explanation."
            
    def _generate_counterfactuals(
        self,
        current: pd.DataFrame,
        historical: pd.DataFrame,
        outputs: Dict[str, float]
    ) -> List[Dict[str, Union[str, float]]]:
        """Generate counterfactual explanations"""
        try:
            counterfactuals = []
            decision = outputs["final_decision"]
            
            # Find similar historical cases with opposite decisions
            X = self._scaler.fit_transform(historical)
            current_normalized = self._scaler.transform(current)
            
            # Calculate distances
            distances = np.linalg.norm(X - current_normalized, axis=1)
            closest_indices = np.argsort(distances)
            
            # Find counterfactuals
            for idx in closest_indices[:10]:
                cf_features = historical.iloc[idx]
                
                # Check if it would lead to opposite decision
                if (decision > 0 and cf_features.mean() < 0) or \
                   (decision < 0 and cf_features.mean() > 0):
                    
                    # Calculate key differences
                    differences = {}
                    for col in historical.columns:
                        curr_val = current[col].iloc[0]
                        cf_val = cf_features[col]
                        if abs(curr_val - cf_val) > 0.1:  # Threshold for significant difference
                            differences[col] = {
                                "from": curr_val,
                                "to": cf_val,
                                "change": (cf_val - curr_val) / curr_val * 100
                            }
                            
                    if differences:
                        counterfactuals.append({
                            "scenario": f"Scenario {len(counterfactuals)+1}",
                            "differences": differences,
                            "outcome": "opposite decision"
                        })
                        
                    if len(counterfactuals) >= 3:
                        break
                        
            return counterfactuals
            
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "counterfactual_generation_error",
                    error=str(e)
                )
            return []
            
    def _create_visual_explanations(
        self,
        features: Dict[str, float],
        importance: Dict[str, float],
        historical_data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Create visual explanations"""
        try:
            visuals = {}
            
            # Feature importance chart
            sorted_importance = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            visuals["importance"] = {
                "type": "bar",
                "data": {
                    "x": [x[0] for x in sorted_importance],
                    "y": [x[1] for x in sorted_importance]
                },
                "layout": {
                    "title": "Feature Importance",
                    "xaxis_title": "Feature",
                    "yaxis_title": "Importance (%)"
                }
            }
            
            # Historical distribution
            for feature, value in features.items():
                if feature in historical_data.columns:
                    hist_data = historical_data[feature]
                    visuals[f"distribution_{feature}"] = {
                        "type": "histogram",
                        "data": {
                            "x": hist_data.tolist(),
                            "current_value": value
                        },
                        "layout": {
                            "title": f"{feature} Distribution",
                            "xaxis_title": feature,
                            "yaxis_title": "Frequency"
                        }
                    }
                    
            return visuals
            
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    "visual_explanation_error",
                    error=str(e)
                )
            return {} 