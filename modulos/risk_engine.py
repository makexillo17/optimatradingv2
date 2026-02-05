import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field, field_validator
import logging

# Setup Logger usually handled globally, but we'll instantiate for the engine
logger = logging.getLogger("RiskEngine")

# --- 1. CONFIG MODELS (Pydantic V2) ---

class RiskConfig(BaseModel):
    max_account_risk_per_trade: float = Field(0.02, ge=0.001, le=0.05, description="Max % of equity to lose in 1 trade")
    kelly_fraction: float = Field(0.5, ge=0.1, le=1.0, description="Half-Kelly or Quarter-Kelly multiplier")
    base_atr_multiplier: float = 2.0
    risk_free_rate: float = 0.0 # For Sharpe/Kelly theoretical adjustments
    
class TradeContext(BaseModel):
    current_equity: float = Field(..., gt=0)
    win_rate: float = Field(..., ge=0.0, le=1.0) # Historical or Rolling
    payoff_ratio: float = Field(..., ge=0.0) # Avg Win / Avg Loss
    hurst_exponent: float = Field(0.5, ge=0.0, le=1.0)
    current_atr: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    side: str = Field(..., pattern="^(long|short)$")
    micro_structure_gaps: List[float] = Field(default_factory=list, description="Price levels of known gaps/LVNs")

# --- 2. RISK ENGINE LOGIC ---

class InstitutionalRiskMotor:
    """
    Gestiona el riesgo y el dimensionamiento de posición utilizando estándares institucionales.
    - Fractional Kelly Criterion para sizing.
    - Volatility-Adjusted Stops (ATR + Hurst).
    - Microstructure-Aware placement.
    """
    def __init__(self, config: RiskConfig = None):
        if config is None:
            self.config = RiskConfig()
        else:
            self.config = config
            
    def calculate_position_size(self, context: TradeContext) -> Dict[str, float]:
        """
        Calcula el tamaño de posición óptimo usando Fractional Kelly.
        $f = p - (q / b)$
        donde p = win_rate, q = 1-p, b = payoff_ratio.
        """
        p = context.win_rate
        b = context.payoff_ratio
        
        # Kelly Formula
        if b <= 0:
            optimal_f = 0.0
        else:
            q = 1 - p
            optimal_f = p - (q / b)
            
        # Apply Fraction (Safety)
        kelly_size_fraction = max(0.0, optimal_f) * self.config.kelly_fraction
        
        # Hard Cap (Risk per Trade Constraint)
        # Why? Kelly says 'how much to bet', but we usually interpret 'f' as % of bankroll to risk? 
        # Ideally Kelly gives leverage/size directly. 
        # If Optimal F is 0.10, we bet 10% of equity. 
        # But commonly in trading we cap 'Risk Amount' (Loss at Stop), not just 'Notional Size'.
        # Let's interpret 'Kelly Fraction' as the Risk Amount % we are willing to take?
        # Standard Kelly gives size. 
        # Let's align with "Max Risk 2%". If Kelly says risk 10%, we clip to 2%.
        
        # Risk of Ruin Check
        if optimal_f <= 0:
            logger.warning("Kelly Criterion negativo. No operar.")
            return {'size_asset': 0.0, 'risk_amount': 0.0, 'reason': 'Negative Kelly'}

        final_risk_fraction = min(kelly_size_fraction, self.config.max_account_risk_per_trade)
        
        risk_amount_usd = context.current_equity * final_risk_fraction
        
        # Calculate Asset Size based on STOP DISTANCE (Volatility Sizing)
        # Size = Risk_Amount / Distance_to_Stop
        stop_price = self.get_dynamic_stop_loss(context)
        stop_distance = abs(context.entry_price - stop_price)
        
        if stop_distance == 0:
             return {'size_asset': 0.0, 'risk_amount': 0.0, 'reason': 'Zero Stop Distance'}
             
        size_asset = risk_amount_usd / stop_distance
        
        # Risk of Ruin Projection (Simplified Approximation)
        # e^(-2*E*B / Var) ... complex. 
        # Simple rule: if win_rate < 0.3 and payoff < 2, risk is high.
        risk_of_ruin_warning = False
        if context.win_rate < 0.35 and context.payoff_ratio < 1.5:
             logger.warning(f"HIGH RISK OF RUIN PROJECTION: WR {context.win_rate:.2f} / Payoff {context.payoff_ratio:.2f}")
             risk_of_ruin_warning = True

        return {
            'size_asset': size_asset,
            'risk_amount': risk_amount_usd,
            'risk_percent': final_risk_fraction,
            'stop_price': stop_price,
            'kelly_raw': optimal_f,
            'warning_ruin': risk_of_ruin_warning
        }

    def get_dynamic_stop_loss(self, context: TradeContext) -> float:
        """
        Calcula el Stop Loss basado en ATR ajustado por Régimen (Hurst).
        """
        base_mult = self.config.base_atr_multiplier
        hurst = context.hurst_exponent
        
        # Hurst Adaptation
        # High Hurst (Trend) -> Tighter Stops (We expect follow through, don't give back)
        # Low Hurst (Mean Rev) -> Wider Stops (Expect noise/excursion)
        
        if hurst > 0.65:
            final_mult = base_mult * 0.75 # Tighten
        elif hurst < 0.45:
            final_mult = base_mult * 1.5 # Widen
        else:
            final_mult = base_mult
            
        distance = context.current_atr * final_mult
        
        if context.side == 'long':
            raw_stop = context.entry_price - distance
        else:
            raw_stop = context.entry_price + distance
            
        # Microstructure Validation (Paradoja del Stop-Loss)
        # Si el stop cae en un "Gap" o nivel de liquidez conocido, mejor moverlo DETRÁS.
        # "Avoid placing stops in liquidity gaps (LVN), place them behind HVN".
        # Simplified: If raw_stop appears in context.micro_structure_gaps (approx), move it further.
        
        adjusted_stop = self._adjust_for_microstructure(raw_stop, context.side, context.micro_structure_gaps)
        
        return adjusted_stop

    def _adjust_for_microstructure(self, stop_price: float, side: str, gaps: List[float]) -> float:
        """
        Si el stop está cerca de un gap, lo empuja un poco más lejos para evitar ser barrido 
        justo antes de que el precio llene el gap y revierta.
        """
        if not gaps:
            return stop_price
            
        # Threshold de cercanía (ej: 0.2% del precio)
        threshold = stop_price * 0.002 
        
        for gap_level in gaps:
            if abs(stop_price - gap_level) < threshold:
                # Conflicto: Stop en Gap.
                # Push it further out.
                push = stop_price * 0.005 # Empujar 0.5% extra
                if side == 'long':
                    logger.info(f"MicroStructure: Ajustando Stop Long de {stop_price:.2f} a {stop_price-push:.2f} (Gap en {gap_level:.2f})")
                    return stop_price - push
                else:
                    logger.info(f"MicroStructure: Ajustando Stop Short de {stop_price:.2f} a {stop_price+push:.2f} (Gap en {gap_level:.2f})")
                    return stop_price + push
                    
        return stop_price

    def update_trailing_exit(self, current_price: float, high_water_mark: float, atr: float, side: str) -> float:
        """
        Calcula nivel de Chandelier Exit Dinámico.
        Long Exit = High Water Mark - (Multiplier * ATR)
        """
        mult = 3.0 # Chandelier standard implies keeping profits loose
        # Could be config driven
        
        if side == 'long':
            exit_price = high_water_mark - (mult * atr)
            # Ensure exit never goes down (handled by caller usually, but logic here implies strictly based on HWM)
            return exit_price
        else:
            # Short Exit = Low Water Mark + (Multiplier * ATR)
            # Assuming 'high_water_mark' passed here is actually the 'lowest low' for shorts
            exit_price = high_water_mark + (mult * atr)
            return exit_price

# --- 3. DEMO / TEST ---
if __name__ == "__main__":
    # Test Data
    cfg = RiskConfig(max_account_risk_per_trade=0.02, kelly_fraction=0.5)
    motor = InstitutionalRiskMotor(cfg)
    
    ctx = TradeContext(
        current_equity=10000.0,
        win_rate=0.55,
        payoff_ratio=2.0, # 2:1 Reward Risk
        hurst_exponent=0.70, # Trending
        current_atr=50.0,
        entry_price=45000.0,
        side='long',
        micro_structure_gaps=[44900.0] # Gap just below entry
    )
    
    # 1. Calculate Stop
    stop = motor.get_dynamic_stop_loss(ctx)
    print(f"Entry: {ctx.entry_price}")
    print(f"ATR: {ctx.current_atr}")
    print(f"Hurst: {ctx.hurst_exponent} (High -> Tighten Monitor)")
    print(f"Calculated Stop: {stop:.2f}")
    
    # 2. Calculate Sizing
    res = motor.calculate_position_size(ctx)
    print("\n--- POSITION SIZING ---")
    print(f"Kelly Raw: {res.get('kelly_raw'):.4f}")
    print(f"Risk Amount: ${res['risk_amount']:.2f} ({res['risk_percent']*100:.2f}%)")
    print(f"Position Size (Asset): {res['size_asset']:.4f} BTC")
