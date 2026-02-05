import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# --- 1. DATA MODELS & CONSTANTS ---
# Using strict slots for HFT-like performance (minimizing overhead)

STATUS_TRENDING = "TRENDING"
STATUS_RANGING = "RANGING"
STATUS_IDLE = "NOISE"  # Kill Switch

@dataclass(slots=True)
class RegimeState:
    """
    Market State Snapshot acts as the central nervous signal.
    
    Attributes:
        regime (str): The classification (TRENDING, RANGING, NOISE).
        hurst_exponent (float): The persistence metric (0-1).
        kalman_price (float): The de-noised price estimate.
        kalman_velocity (float): The trend speed (dPrice/dt).
        is_kill_switch_active (bool): True if Entropy > Threshold (H ~ 0.5).
        risk_multiplier (float): Position sizing adjustor (e.g., 0.0 to 1.0).
        atr_stop_multiplier (float): Dynamic Stop Loss width (Tight vs Wide).
        explanation (str): Human-readable justification.
    """
    regime: str
    hurst_exponent: float
    kalman_price: float
    kalman_velocity: float
    is_kill_switch_active: bool
    risk_multiplier: float
    atr_stop_multiplier: float
    explanation: str

# --- 2. PERSISTENCE ENGINE (Hurst) ---

class HurstCalculator:
    """
    Calculates the Hurst Exponent to measure time-series persistence (Long Term Memory).
    Uses Rescaled Range (R/S) Analysis on rolling windows.
    """
    def __init__(self, min_chunk_size: int = 8):
        self.min_chunk_size = min_chunk_size

    def calculate_rs(self, series: np.ndarray) -> Tuple[float, float]:
        """Calculates (R/S) statistic for a given series."""
        # Clean data needed? Assuming clean numpy array inputs.
        
        # 1. Calculate Mean returns (or difference)
        # Often H is calc on log-returns for financial data
        # But R/S can be on the series itself if detrended. 
        # Standard approach: Log returns r_t = ln(P_t / P_{t-1})
        # Note: If series passed is already returns, skip. 
        # But caller usually passes price. We'll handle price -> returns inside calculate_rolling.
        
        # Actually R/S on differences:
        mean = np.mean(series)
        
        # 2. Deviations from mean
        y = series - mean
        
        # 3. Cumulative Deviations
        z = np.cumsum(y)
        
        # 4. Range
        r = np.max(z) - np.min(z)
        
        # 5. Standard Deviation
        s = np.std(series) # Use population or sample? std(ddof=1)
        
        if s == 0: return 0.0, 0.0
        
        return r, s

    def compute(self, price_series: pd.Series, max_lag: int = 20) -> float:
        """
        Estimates Hurst Exponent using the R/S method over multiple lags.
        ln(R/S) ~ H * ln(n)
        
        Args:
            price_series: pd.Series of Closing Prices.
            max_lag: not used in simplified R/S, but we usually slice different windows.
                     For a robust rolling H, we usually take a window (e.g. 100) and 
                     subdivide it or just use the whole window for one R/S point?
                     
        Simplified Fast Hurst for Rolling Window (100 candles):
        We can't do full regression on every candle efficiently in Python without lag.
        
        Implementation: 
        We use the simplified formula or a small regression over dyadic subdivisions.
        """
        # Convert prices to log returns
        returns = np.log(price_series / price_series.shift(1)).dropna()
        if len(returns) < self.min_chunk_size * 2:
            return 0.5 # Default Random Walk

        # Classic R/S Analysis
        # We perform R/S calc on the window.
        # Ideally we need regression of log(R/S) vs log(n).
        # We will split the data into chunks of size n.
        
        vals = returns.values
        N = len(vals)
        
        # Ranges of n (chunk sizes) to test
        # e.g., N=100 -> chunks: [10, 20, 50, 100]
        min_n = self.min_chunk_size
        max_n = N
        
        rs_values = []
        n_values = []
        
        # Create a few scales
        scales = [int(N / (2**i)) for i in range(0, 4) if int(N / (2**i)) > min_n]
        scales = sorted(list(set(scales))) # Unique
        
        for n in scales:
            # Split data into chunks of size n
            num_chunks = N // n
            chunks_rs = []
            
            for i in range(num_chunks):
                chunk = vals[i*n : (i+1)*n]
                r, s = self.calculate_rs(chunk)
                if s > 0 and r > 0:
                    chunks_rs.append(r/s)
            
            if chunks_rs:
                avg_rs = np.mean(chunks_rs)
                rs_values.append(avg_rs)
                n_values.append(n)
                
        if len(rs_values) < 2:
            return 0.5
            
        # Log-Log Regression
        # ln(R/S) = H * ln(n) + c
        y_reg = np.log(rs_values)
        x_reg = np.log(n_values)
        
        # Polyfit degree 1
        A = np.vstack([x_reg, np.ones(len(x_reg))]).T
        m, c = np.linalg.lstsq(A, y_reg, rcond=None)[0]
        
        return float(m)

# --- 3. TREND TRACKING ENGINE (Kalman) ---

class KalmanTrendFilter:
    """
    Adaptive Kalman Filter for Price Estimation.
    Model: Constant Velocity (Locally Linear Trend).
    x = [price, velocity]'
    F = [[1, dt], [0, 1]]
    """
    def __init__(self, process_noise_scale: float = 1e-3):
        self.process_noise_scale = process_noise_scale
        
        # State Vector [p, v]
        self.x = np.zeros(2) 
        
        # Covariance Matrix P
        self.P = np.eye(2) 
        
        # Transition Matrix F (dt = 1)
        self.F = np.array([[1.0, 1.0], 
                           [0.0, 1.0]])
        
        # Measurement Matrix H (we only measure price)
        self.H = np.array([1.0, 0.0])
        
        # Process Noise Q
        # Assumes small random jerks in velocity
        self.Q = np.array([[process_noise_scale, 0],
                           [0, process_noise_scale]])
        
        self.initialized = False

    def update(self, measurement_price: float, volatility_atr: float):
        """
        Updates the Kalman state.
        
        Args:
            measurement_price: Current Close price.
            volatility_atr: Current ATR(14) used to adapt Measurement Noise R.
        """
        if not self.initialized:
            self.x = np.array([measurement_price, 0.0])
            self.P = np.eye(2) * volatility_atr
            self.initialized = True
            return

        # 1. Prediction Step
        # x_pred = F * x
        x_pred = self.F @ self.x
        # P_pred = F * P * F.T + Q
        P_pred = (self.F @ self.P @ self.F.T) + self.Q
        
        # 2. Update Step
        # Adaptive R based on ATR. 
        # If High Volatility (Noise), R is High -> We trust Model (Trend) more than Measurement.
        # If Low Volatility, R is Low -> We trust Measurement more.
        # R = sigma_meas^2. We can approximate sigma_meas ~ ATR.
        R = (volatility_atr) ** 2 if volatility_atr > 0 else 1.0
        
        # Innovation/Residual: y = z - H * x_pred
        z = measurement_price
        y = z - (self.H @ x_pred)
        
        # Innovation Covariance: S = H * P_pred * H.T + R
        S = (self.H @ P_pred @ self.H.T) + R
        
        # Kalman Gain: K = P_pred * H.T * inv(S)
        K = (P_pred @ self.H.T) / S
        
        # New State: x = x_pred + K * y
        self.x = x_pred + (K * y)
        
        # New Covariance: P = (I - K * H) * P_pred
        I = np.eye(2)
        # Numerical stability form: P = (I-KH)P(I-KH)' + KRK' (Joseph form) is better but standard is ok here
        self.P = (I - np.outer(K, self.H)) @ P_pred
        
    def get_state(self) -> Tuple[float, float]:
        """Returns (Estimated Price, Estimated Velocity)"""
        return self.x[0], self.x[1]

# --- 4. MAIN MODULE LOGIC ---

class MarketRegimeModule:
    def __init__(self):
        self.hurst_calc = HurstCalculator(min_chunk_size=8)
        self.kalman_filter = KalmanTrendFilter(process_noise_scale=0.001)
        self.last_atr = 0.0
        
    def analyze(self, df: pd.DataFrame) -> RegimeState:
        """
        Central processing unit for Regime Detection.
        """
        if len(df) < 50:
            return self._get_fallback_state()
            
        current_close = df['close'].iloc[-1]
        
        # 1. Calculate ATR for Kalman Adaptability
        # Simple rolling approximation if ta lib not available or for speed
        # Assuming df has 'high', 'low', 'close'
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_series = tr.rolling(window=14).mean()
        current_atr = atr_series.iloc[-1]
        
        if np.isnan(current_atr): current_atr = current_close * 0.01
        
        # 2. Update Kalman Filter
        # Ideally we feed recent history to warm up if it's the first run, 
        # but here we assume sequential calls or we handle warmup.
        # For this standalone simplified fn, we might reset KF or loop.
        # In a real BacktestEngine, we call .analyze() sequentially.
        # If 'initialized' is false, let's fast-forward KF over last 10 candles?
        if not self.kalman_filter.initialized:
            warmup_window = min(len(df), 30)
            for i in range(len(df)-warmup_window, len(df)):
                c_p = df['close'].iloc[i]
                c_atr = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else c_p * 0.01
                self.kalman_filter.update(c_p, c_atr)
        else:
            self.kalman_filter.update(current_close, current_atr)
            
        k_price, k_velocity = self.kalman_filter.get_state()
        
        # 3. Calculate Hurst Exponent (Memory)
        # Use last 100 periods
        window_size = 100
        if len(df) > window_size:
            hurst_window = df['close'].iloc[-window_size:]
        else:
            hurst_window = df['close']
            
        hurst = self.hurst_calc.compute(hurst_window)
        
        # 4. Classification Logic
        # H > 0.65       -> Persistence (Trend)
        # H < 0.45       -> Anti-Persistence (Mean Rev)
        # 0.45 <= H <= 0.55 -> Geometric Brownian Motion (Random Walk/Noise)
        
        regime = STATUS_IDLE
        risk_mult = 0.0
        stop_mult = 3.5 # Default safety
        kill_switch = False
        explanation = ""
        
        if hurst > 0.65:
            regime = STATUS_TRENDING
            risk_mult = 1.0 # Aggressive (Kelly 50% applied elsewhere? Or here?)
            # Prompt says "Riesgo Kelly del 50%". We'll signal 1.0 factor for the sizing engine to use.
            stop_mult = 2.0 # Tight stops in trend
            explanation = f"🚀 TENDENCIA FUERTE (H={hurst:.2f}). Mercado Persistente. Activar SMC/Trend."
            
        elif hurst < 0.45:
            regime = STATUS_RANGING
            risk_mult = 0.5 # Reduce size in Mean Reversion
            stop_mult = 3.5 # Wide stops
            explanation = f"🔁 RANGO / REVERSIÓN (H={hurst:.2f}). Mercado Antipersistente. Activar Gaps/Bollinger."
            
        else:
            # Random Walk Zone (0.45 - 0.55) -> MAX ENTROPY
            regime = STATUS_IDLE
            kill_switch = True
            risk_mult = 0.0
            stop_mult = 5.0 # Irrelevant if no trade
            explanation = f"⚠️ KILL SWITCH ACTIVADO (H={hurst:.2f} ~ 0.5). Alta Entropía / Ruido Aleatorio. PROTECCIÓN CAPITAL."
            
        return RegimeState(
            regime=regime,
            hurst_exponent=hurst,
            kalman_price=k_price,
            kalman_velocity=k_velocity,
            is_kill_switch_active=kill_switch,
            risk_multiplier=risk_mult,
            atr_stop_multiplier=stop_mult,
            explanation=explanation
        )

    def _get_fallback_state(self) -> RegimeState:
        return RegimeState(
            STATUS_IDLE, 0.5, 0.0, 0.0, True, 0.0, 3.0, "Datos insuficientes"
        )

# --- 5. COMPATIBILITY WRAPPER (Facade) ---

_global_regime_module = MarketRegimeModule()

def get_market_regime(df: pd.DataFrame) -> RegimeState:
    """Institutional entry point returning full state object."""
    return _global_regime_module.analyze(df)

def detect_regime(df: pd.DataFrame) -> str:
    """Legacy wrapper for backward compatibility."""
    state = get_market_regime(df)
    return state.regime

# --- 6. EXAMPLE EXECUTION (Kill Switch Demo) ---
if __name__ == "__main__":
    print("--- SIMULATING MARKET REGIME ---")
    
    # 1. Create Synthetic Random Walk (Kill Switch Scenario)
    np.random.seed(42)
    # Generate geometric brownian motion close to H=0.5
    returns = np.random.normal(0, 0.01, 200)
    prices = 100 * np.exp(np.cumsum(returns))
    
    df_noise = pd.DataFrame({
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'open': prices # Approx
    })
    
    state = get_market_regime(df_noise)
    print(f"\nScenario 1: Random Walk (Noise)")
    print(f"Hurst: {state.hurst_exponent:.2f}")
    print(f"Regime: {state.regime}")
    print(f"Kill Switch: {state.is_kill_switch_active}")
    print(f"Msg: {state.explanation}")
    
    if state.is_kill_switch_active:
        print(">>> BLOCKED: Order rejected by Entropy Filter.")
        
    # 2. Create Trending Data (H > 0.65)
    t = np.linspace(0, 10, 200)
    trend = t * 2 # Deterministic trend
    noise = np.random.normal(0, 0.5, 200)
    trend_prices = 100 + trend + noise
    df_trend = pd.DataFrame({'high': trend_prices+1,'low':trend_prices-1,'close':trend_prices})
    
    # Need to reset filter or re-instantiate for clean test
    # (In live mod, the internal filter keeps state, which is good)
    state_trend = _global_regime_module.analyze(df_trend)
    print(f"\nScenario 2: Strong Trend")
    print(f"Hurst: {state_trend.hurst_exponent:.2f}")
    print(f"Regime: {state_trend.regime}")
    print(f"Kill Switch: {state_trend.is_kill_switch_active}")
    print(f"Msg: {state_trend.explanation}")
