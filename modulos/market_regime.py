import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands

def detect_regime(df: pd.DataFrame) -> str:
    """
    Detects the market regime based on ADX and Bollinger Bandwidth.
    
    Regimes:
    - TRENDING: ADX(14) > 25
    - RANGING: ADX(14) < 20 AND BBW < BBW_SMA(50)
    - NOISE: Everything else
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        
    Returns:
        str: 'TRENDING', 'RANGING', or 'NOISE'
    """
    if len(df) < 50:
        return 'NOISE' # Not enough data
        
    try:
        # Calculate ADX (14)
        # ta expects Series
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        adx = adx_indicator.adx().iloc[-1]
        
        # Calculate Bollinger Bands (20, 2) for BBW
        bb_indicator = BollingerBands(close=df['close'], window=20, window_dev=2)
        bb_upper = bb_indicator.bollinger_hband()
        bb_lower = bb_indicator.bollinger_lband()
        bb_mavg = bb_indicator.bollinger_mavg()
        
        # Calculate BBW: (Upper - Lower) / Middle
        # Handle division by zero just in case
        bbw_series = (bb_upper - bb_lower) / bb_mavg
        
        # We need current BBW and Average BBW over last 50 periods
        current_bbw = bbw_series.iloc[-1]
        
        # Calculate BBW SMA 50
        # If we don't have enough data for 50 period SMA of BBW, we might get NaN.
        # We need at least 50 periods + 20 for BB calculation = 70 candles ideally.
        bbw_sma_50 = bbw_series.rolling(window=50).mean().iloc[-1]
        
        # Classification Logic
        if adx > 25:
            return 'TRENDING'
        
        if adx < 20 and current_bbw < bbw_sma_50:
            return 'RANGING'
            
        return 'NOISE'
        
    except Exception as e:
        print(f"Error in detect_regime: {e}")
        return 'NOISE'
