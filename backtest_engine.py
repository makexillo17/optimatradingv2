import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import time
import yaml

# Force UTF-8 output on Windows console (prevents emoji encoding errors)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Debug: verificar carga de .env ──────────────────────────────────
from dotenv import load_dotenv
from pathlib import Path

# Ruta absoluta al .env del proyecto (evita problemas con espacios en ruta)
_PROJECT_ROOT = Path(r"c:\Users\chump\OneDrive\proyecto personal")
_env_path = _PROJECT_ROOT / ".env"

print(f"Buscando .env en: {_env_path}  (existe: {_env_path.exists()})")
load_dotenv(dotenv_path=str(_env_path), override=True)

_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
if _key:
    print(f"Llave encontrada: {_key[:5]}...{_key[-4:]}  (len={len(_key)})")
else:
    print("ADVERTENCIA: ANTHROPIC_API_KEY no encontrada. Verifica tu archivo .env")

# ── Verificar dependencia ta ────────────────────────────────────────
try:
    import ta
    print("[OK] Libreria 'ta' cargada correctamente.")
except ModuleNotFoundError:
    print(
        "\n[ERROR] No se encontro el modulo 'ta' (Technical Analysis).\n"
        "Instalalo con:  pip install ta\n"
    )
    import sys
    sys.exit(1)

# Import Modules
from modulos.smc_ict import SmcIctModule
from modulos.broker_behavior import BrokerBehaviorModule
from modulos.yield_anomaly import YieldAnomalyModule
from modulos.carry_trade import CarryTradeModule
from modulos.dynamic_hedging import DynamicHedgingModule
from modulos.gap_sniper import GapSniperModule
from modulos.volatility_arb import VolatilityArbModule
from modulos.stat_arb import StatArbModule
from modulos.liquidity_provision import LiquidityProvisionModule
from modulos.market_making import MarketMakingModule
from modulos.market_regime import detect_regime
from modulos.volatility_guard import VolatilityGuardModule

from main.consensus import ConsensusAnalyzer
from llm_client import ClaudeTrader

class BacktestEngine:
    def __init__(self, data_file='data/btc_history.csv', initial_capital=10000.0, commission_rate=0.001):
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.balance = initial_capital # Acts as "Free Cash" for Longs, and "Collateral" for Shorts
        self.commission_rate = commission_rate
        self.equity_curve = []
        self.trades = []
        self.position = None 
        self.verbose = True
        
        # ── Forensic Analysis Log ───────────────────────────────────
        self.forensic_log = []  # List of dicts for brutal_analysis_log.csv
        self._last_exit_reason = None  # Tracks most recent exit reason per candle
        
        # ── Engine Isolation Mode (Benchmarking) ────────────────────
        self.isolation_enabled = False
        self.isolation_target = None
        self.volume_threshold = 1.5
        self.sweep_lookback_hours = 24
        self.vix_percentile = 90.0
        self.break_even_ratio = 1.0
        self._load_isolation_config()
        
        # Modules
        self.modules = {
            'smc_ict': SmcIctModule(),
            'broker_behavior': BrokerBehaviorModule(),
            'yield_anomaly': YieldAnomalyModule(),
            'dynamic_hedging': DynamicHedgingModule(),
            'gap_sniper': GapSniperModule(
                volume_threshold=self.volume_threshold,
                sweep_lookback_hours=self.sweep_lookback_hours
            ),
            'volatility_arb': VolatilityArbModule(),
            'carry_trade': CarryTradeModule(),
            'stat_arb': StatArbModule(),
            'liquidity_provision': LiquidityProvisionModule(),
            'volatility_guard': VolatilityGuardModule(
                vix_percentile=self.vix_percentile
            )
        }
        self.consensus = ConsensusAnalyzer()

        # AI Engine (skip in isolation mode — no Claude needed)
        if not self.isolation_enabled:
            self.trader = ClaudeTrader()
        else:
            self.trader = None
            print(f"\n{'='*60}")
            print(f"  🔬 MODO AISLAMIENTO ACTIVO: {self.isolation_target.upper()}")
            print(f"  ConsensusAnalyzer: BYPASS")
            print(f"  ClaudeTrader:      BYPASS")
            print(f"  Motor único:       {self.isolation_target}")
            print(f"{'='*60}\n")
        
    def _load_isolation_config(self):
        """Lee testing_mode.engine_isolation y parámetros de señales desde config.yaml."""
        config_path = _PROJECT_ROOT / "config" / "config.yaml"
        try:
            with open(str(config_path), 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            testing = cfg.get('ai_engine', {}).get('testing_mode', {})
            isolation = testing.get('engine_isolation', {})
            self.isolation_enabled = isolation.get('enabled', False)
            self.isolation_target = isolation.get('target_engine', 'gap_sniper')
            
            # Parametros de senales (configurable desde YAML)
            self.volume_threshold = testing.get('volume_threshold', 1.5)
            self.sweep_lookback_hours = testing.get('sweep_lookback_hours', 24)
            self.vix_percentile = testing.get('vix_percentile', 90.0)
            
            inst_logic = cfg.get('ai_engine', {}).get('institutional_logic', {})
            self.break_even_ratio = inst_logic.get('break_even_ratio', 1.0)
        except Exception as e:
            print(f"[Config] No se pudo leer testing_mode: {e} — modo normal.")
            self.isolation_enabled = False
            self.isolation_target = None

    def load_data(self):
        if not os.path.exists(self.data_file):
            print(f"Error: {self.data_file} not found. Run utils/data_loader.py first.")
            return None
        
        df = pd.read_csv(self.data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Data Loaded: {len(df)} candles.")
        return df

    def run(self):
        df = self.load_data()
        if df is None: return
        
        print("Starting Simulation...")
        
        # PRE-CALCULATE EMA 200 FOR TREND FILTER
        # Using pandas ewm or rolling. standard is EMA.
        # df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        # Or using ta library if preferred, but pandas is faster/easier here without extra imports if not needed.
        # Let's use simple pandas calculation for speed.
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        start_idx = 205
        self.cooldown = 0 # Cooldown counter in candles
        
        for i in range(start_idx, len(df)):
            # Update cooldown
            if self.cooldown > 0:
                self.cooldown -= 1
                
            current_df = df.iloc[:i+1].copy()
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            current_time = current_candle['timestamp']
            current_ema200 = current_candle['ema200']
            
            # Reset per-candle exit reason tracker
            self._last_exit_reason = None
            
            # ══════════════════════════════════════════════════════════
            # MODO AISLAMIENTO vs MODO NORMAL
            # ══════════════════════════════════════════════════════════
            
            market_input = {'market_data': current_df}
            current_regime = detect_regime(current_df)
            
            if self.isolation_enabled:
                # ── ISOLATION MODE: Solo el motor target ───────────────
                module_results = {}
                
                # Ejecutar SOLO el motor objetivo + dynamic_hedging (para stops)
                target = self.isolation_target
                if target in self.modules:
                    try:
                        module_results[target] = self.modules[target].analyze(market_input)
                    except Exception as e:
                        print(f"[{current_time}] Error en {target}: {e}")
                
                # Dynamic hedging siempre se ejecuta (necesario para stops)
                if target != 'dynamic_hedging' and 'dynamic_hedging' in self.modules:
                    try:
                        module_results['dynamic_hedging'] = self.modules['dynamic_hedging'].analyze(market_input)
                    except Exception:
                        pass
                
                # Volatility Guard siempre se ejecuta (filtro macro)
                if target != 'volatility_guard' and 'volatility_guard' in self.modules:
                    try:
                        module_results['volatility_guard'] = self.modules['volatility_guard'].analyze(market_input)
                    except Exception:
                        pass
                
                # Convertir señal del motor directamente a decisión de trading
                engine_result = module_results.get(target, {})
                engine_rec = engine_result.get('recommendation', 'neutral')
                engine_conf = engine_result.get('confidence', 0.0)
                
                # Mapeo directo: long→BUY, short→SELL, neutral→HOLD
                if engine_rec == 'long' and engine_conf > 0.0:
                    ai_decision = "BUY"
                elif engine_rec == 'short' and engine_conf > 0.0:
                    ai_decision = "SELL"
                else:
                    ai_decision = "HOLD"
                
                # -- HTF ALIGNMENT FILTER (EMA200 Hierarchy) --------
                if ai_decision == "BUY" and current_price < current_ema200:
                    ai_decision = "HOLD"
                    if self.verbose:
                        print(f"[{current_time}] HTF FILTER: BUY bloqueado (precio {current_price:.2f} < EMA200 {current_ema200:.2f})")
                elif ai_decision == "SELL" and current_price > current_ema200:
                    ai_decision = "HOLD"
                    if self.verbose:
                        print(f"[{current_time}] HTF FILTER: SELL bloqueado (precio {current_price:.2f} > EMA200 {current_ema200:.2f})")
                
                # -- VOLATILITY GUARD (Delta-VIX + RSI HTF) ----------
                vg = module_results.get('volatility_guard', {})
                if vg.get('force_hold'):
                    if ai_decision != "HOLD":
                        print(f"[{current_time}] CRISIS VOL: {ai_decision} bloqueado (Vol Regime: {vg.get('vol_regime')})")
                    ai_decision = "HOLD"
                elif ai_decision == "BUY" and vg.get('block_buy'):
                    print(f"[{current_time}] RSI HTF: BUY bloqueado (RSI={vg.get('daily_rsi', 0):.1f} > {self.modules.get('volatility_guard', VolatilityGuardModule()).rsi_overbought})")
                    ai_decision = "HOLD"
                elif ai_decision == "SELL" and vg.get('block_sell'):
                    print(f"[{current_time}] RSI HTF: SELL bloqueado (RSI={vg.get('daily_rsi', 0):.1f} < {self.modules.get('volatility_guard', VolatilityGuardModule()).rsi_oversold})")
                    ai_decision = "HOLD"
                
                # -- NOISE REVERSION MODE ----------------------------
                if current_regime == 'NOISE' and target in ('carry_trade',):
                    ai_decision = "HOLD"
                
                # Sin consenso en aislamiento
                consensus_signal = 0.0
                recommendation = engine_rec
                consensus_result = {'confidence': engine_conf, 'signal': 0.0}
                
                if self.verbose and i % 24 == 0:
                    print(f"[{current_time}] 🔬 ISOLATION [{target}]: {engine_rec} (conf: {engine_conf:.2f}) → {ai_decision} | Regime: {current_regime} | HTF: {'BULL' if current_price > current_ema200 else 'BEAR'}")
                
            else:
                # ── MODO NORMAL: Consenso + Claude + Defensas ─────────
                module_results = {}
                for name, module in self.modules.items():
                    try:
                        module_results[name] = module.analyze(market_input)
                    except Exception:
                        pass
                
                # 2. Consensus
                consensus_result = self.consensus.analyze(module_results, market_regime=current_regime)
                consensus_signal = consensus_result.get('signal', consensus_result.get('details', {}).get('avg_signal', 0.0))
                recommendation = consensus_result.get('recommendation', 'NEUTRAL')
                
                if 'signal' not in consensus_result:
                    if recommendation == 'long': consensus_signal = 0.8
                    elif recommendation == 'short': consensus_signal = -0.8

                # 3. AI Decision — ClaudeTrader
                current_row = {
                    'timestamp': str(current_time),
                    'open': float(current_candle['open']),
                    'high': float(current_candle['high']),
                    'low': float(current_candle['low']),
                    'close': float(current_price),
                    'volume': float(current_candle['volume']),
                    'ema200': float(current_ema200),
                    'regime': current_regime,
                    'consensus_signal': float(consensus_signal),
                }

                try:
                    ai_decision = self.trader.analyze_market_data(current_row, market_regime=current_regime)
                except Exception as e:
                    print(f"[{current_time}] Error en ClaudeTrader: {e} — defaulting to HOLD")
                    ai_decision = "HOLD"

                print(f"[{current_time}] Claude decidio: {ai_decision}")
                time.sleep(1.5)  # Rate limit: evitar Error 429 de Anthropic

                # ── VEDA NOISE: Bloquea tendencia, permite reversión (FVG/Sweep) ──
                if current_regime == 'NOISE':
                    # En NOISE, Claude no puede operar tendencia.
                    # Pero si gap_sniper o smc_ict detectan reversión, el consenso ya los habría filtrado.
                    # En modo normal, forzamos HOLD total para Claude.
                    if ai_decision != "HOLD":
                        print(f"[{current_time}] 🚫 VEDA NOISE: Claude dijo {ai_decision} → forzado a HOLD")
                    ai_decision = "HOLD"
                
                # ── HTF ALIGNMENT FILTER (Normal Mode) ────────────────
                if ai_decision == "BUY" and current_price < current_ema200:
                    ai_decision = "HOLD"
                elif ai_decision == "SELL" and current_price > current_ema200:
                    ai_decision = "HOLD"

                # --- VERBOSE DEBUG ---
                if self.verbose and i % 24 == 0:
                     if abs(consensus_signal) > 0.3 or current_regime == 'NOISE':
                        trend_status = "BULL" if current_price > current_ema200 else "BEAR"
                        print(f"[{current_time}] Regime: {current_regime} | Score: {consensus_signal:.2f} | Trend: {trend_status} | SMC: {module_results.get('smc_ict', {}).get('recommendation')}")

            # 4. Risk Params (shared by both modes)
            hedging_info = module_results.get('dynamic_hedging', {})
            stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
            stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
            
            # 5. Position Management
            self._manage_positions(
                current_price, current_time, 
                consensus_signal, recommendation, 
                stop_long, stop_short, 
                module_results,
                current_ema200,
                current_regime,
                ai_decision
            )
            
            # 5. Track Equity
            current_equity = self.balance
            
            if self.position:
                if self.position['type'] == 'long':
                    asset_value = self.position['size'] * current_price
                    current_equity = self.balance + asset_value
                elif self.position['type'] == 'short':
                    unrealized_pnl = (self.position['entry_price'] - current_price) * self.position['size']
                    current_equity = self.balance + unrealized_pnl
            
            self.equity_curve.append({'timestamp': current_time, 'equity': current_equity})

            # ── 6. FORENSIC LOG — brutal_analysis_log.csv ──────────
            carry_result = module_results.get('carry_trade', {})
            smc_result = module_results.get('smc_ict', {})
            
            carry_signal = carry_result.get('recommendation', 'N/A')
            carry_conf = carry_result.get('confidence', 0.0)
            carry_just = carry_result.get('justification', '')
            
            smc_signal = smc_result.get('recommendation', 'N/A')
            smc_conf = smc_result.get('confidence', 0.0)
            smc_structure = smc_result.get('structure', 'N/A')
            smc_ob_type = smc_result.get('nearest_ob_type', 'None')
            
            is_noise = current_regime == 'NOISE'
            
            # Identify the culprit module on exit rows
            culprit = ''
            if self._last_exit_reason:
                last_trade = self.trades[-1] if self.trades else None
                if last_trade and last_trade['pnl'] < 0:
                    trade_type = last_trade['type']
                    trade_dir = 'BUY' if trade_type == 'long' else 'SELL'
                    
                    if self.isolation_enabled:
                        # ── ATRIBUCIÓN DINÁMICA: motor aislado ────────
                        culprit = f'{self.isolation_target}({trade_dir})'
                    else:
                        # ── ATRIBUCIÓN NORMAL: módulos + Claude ───────
                        if trade_type == 'long':
                            if carry_signal == 'long':
                                culprit += 'CarryTrade(long) '
                            if smc_signal == 'long':
                                culprit += 'SMC(long) '
                            culprit += 'Claude(BUY) '
                        elif trade_type == 'short':
                            if carry_signal == 'short':
                                culprit += 'CarryTrade(short) '
                            if smc_signal == 'short':
                                culprit += 'SMC(short) '
                            culprit += 'Claude(SELL) '
                    
                    if is_noise:
                        culprit += ' ⚠️NOISE_REGIME'
                    culprit = culprit.strip()

            self.forensic_log.append({
                'Fecha': str(current_time),
                'Precio': round(current_price, 2),
                'EMA200': round(current_ema200, 2),
                'Regimen': current_regime,
                'Hurst_Score': round(consensus_signal, 4),
                'Señal_CarryTrade': carry_signal,
                'Confianza_CarryTrade': round(carry_conf, 2),
                'Señal_SMC': smc_signal,
                'Confianza_SMC': round(smc_conf, 2),
                'Estructura_SMC': smc_structure,
                'OB_Tipo_SMC': smc_ob_type,
                'Decision_Claude': ai_decision,
                'Consenso_Recomendacion': recommendation,
                'Confianza_Consenso': round(consensus_result.get('confidence', 0.0), 4),
                'Posicion_Activa': self.position['type'] if self.position else 'NINGUNA',
                'Razon_de_Salida': self._last_exit_reason if self._last_exit_reason else '',
                'PnL_Trade': round(self.trades[-1]['pnl'], 2) if (self._last_exit_reason and self.trades) else '',
                'Culpable_Perdida': culprit,
                'FLAG_NOISE': '⚠️ PROHIBIR_OPERAR' if is_noise else '',
                'POI_Quality': self.position.get('poi_quality', '') if self.position else (
                    self.trades[-1].get('poi_quality', '') if (self._last_exit_reason and self.trades) else ''
                )
            })

        # --- FORCE CLOSE AT END ---
        if self.position:
            last_price = df.iloc[-1]['close']
            last_time = df.iloc[-1]['timestamp']
            print("Force closing remaining position at end of simulation.")
            self._close_position(last_price, last_time, "Force Close (End of Data)")

        # ── Save Forensic CSV ───────────────────────────────────────
        self._save_forensic_log()
        self._generate_report()

    def _manage_positions(self, price, timestamp, consensus_signal, recommendation, stop_long, stop_short, module_results, ema200, current_regime, ai_decision="HOLD"):
        
        # ── Check Exits ──────────────────────────────────────────────
        if self.position:
            # ── BREAK-EVEN LOGIC (Defensa de Autor) ───────────────────
            if not self.position.get('break_even_triggered', False):
                initial_sl = self.position.get('initial_stop_loss', 0)
                entry_price = self.position['entry_price']
                if self.position['type'] == 'long' and initial_sl > 0:
                    risk = entry_price - initial_sl
                    profit = price - entry_price
                    if risk > 0 and profit >= (risk * self.break_even_ratio):
                        self.position['stop_loss'] = entry_price
                        self.position['break_even_triggered'] = True
                        print(f"[{timestamp}] 🛡️ BREAK-EVEN TRIGGERED: Profit >= 1:{self.break_even_ratio} R:R. Stop Loss movido a {entry_price:.2f}")
                elif self.position['type'] == 'short' and initial_sl > 0:
                    risk = initial_sl - entry_price
                    profit = entry_price - price
                    if risk > 0 and profit >= (risk * self.break_even_ratio):
                        self.position['stop_loss'] = entry_price
                        self.position['break_even_triggered'] = True
                        print(f"[{timestamp}] 🛡️ BREAK-EVEN TRIGGERED: Profit >= 1:{self.break_even_ratio} R:R. Stop Loss movido a {entry_price:.2f}")

            # STOP LOSS (always active)
            if self.position['type'] == 'long':
                if price <= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            elif self.position['type'] == 'short':
                if price >= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            
            # ── MATHEMATICAL TAKE PROFIT (1:1.5 R:R from Sniper) ────
            if self.position.get('take_profit'):
                tp = self.position['take_profit']
                if self.position['type'] == 'long' and price >= tp:
                    self._close_position(price, timestamp, "Take Profit (1:1.5 R:R)")
                    return
                elif self.position['type'] == 'short' and price <= tp:
                    self._close_position(price, timestamp, "Take Profit (1:1.5 R:R)")
                    return
            
            # TAKE PROFIT / REVERSAL (signal-based exits)
            divergence = module_results.get('yield_anomaly', {}).get('divergence_detected', False)
            div_rec = module_results.get('yield_anomaly', {}).get('recommendation', 'neutral')
            
            # ── TAKE PROFIT MÍNIMO (ATR-Based) ─────────────────────
            current_atr = module_results.get('dynamic_hedging', {}).get('current_atr', 0)
            min_profit_threshold = current_atr * 1.0
            
            if self.position['type'] == 'long':
                unrealized_per_unit = price - self.position['entry_price']
                unrealized_pnl = unrealized_per_unit * self.position['size']
                
                if unrealized_per_unit > 0 and unrealized_per_unit < min_profit_threshold and current_atr > 0:
                    print(f"[{timestamp}] 🔒 TP MÍNIMO: Ganancia ${unrealized_pnl:.2f} < ATR ${min_profit_threshold:.2f} — bloqueando cierre")
                elif ai_decision == "SELL" or consensus_signal < -0.50 or (divergence and div_rec == 'short'):
                    self._close_position(price, timestamp, f"AI/Reversal ({ai_decision})")
                    return
                    
            elif self.position['type'] == 'short':
                unrealized_per_unit = self.position['entry_price'] - price
                unrealized_pnl = unrealized_per_unit * self.position['size']
                
                if unrealized_per_unit > 0 and unrealized_per_unit < min_profit_threshold and current_atr > 0:
                    print(f"[{timestamp}] 🔒 TP MÍNIMO: Ganancia ${unrealized_pnl:.2f} < ATR ${min_profit_threshold:.2f} — bloqueando cierre")
                elif ai_decision == "BUY" or consensus_signal > 0.50 or (divergence and div_rec == 'long'):
                    self._close_position(price, timestamp, f"AI/Reversal ({ai_decision})")
                    return

        # ── Check Entries ──────────────────────────────────────────────
        # In isolation mode: allow entries in NOISE (reversion via FVG/Sweep)
        # In normal mode: NOISE is blocked (Claude veda already forced HOLD)
        noise_allowed = self.isolation_enabled and self.isolation_target in ('gap_sniper', 'smc_ict')
        if not self.position and self.cooldown == 0 and (current_regime != 'NOISE' or noise_allowed):
            
            # Get sniper's mathematical TP/SL if available
            sniper_result = module_results.get('gap_sniper', module_results.get(self.isolation_target, {}))
            sniper_tp = sniper_result.get('take_profit', 0.0)
            sniper_sl = sniper_result.get('stop_loss', 0.0)
            
            # Determine source label for logging
            source = self.isolation_target.upper() if self.isolation_enabled else "Claude"
            
            # LONG ENTRY
            if ai_decision == "BUY":
                capital = self.balance
                size_asset = (capital * 0.99) / price
                
                cost = size_asset * price
                fee = cost * self.commission_rate
                
                if (cost + fee) > self.balance:
                    size_asset = (self.balance - fee) / price
                
                self.balance -= (size_asset * price + fee)
                
                # Use sniper's SL if available, else fallback to hedging stop
                entry_sl = sniper_sl if sniper_sl > 0 else stop_long
                entry_tp = sniper_tp if sniper_tp > 0 else None
                
                self.position = {
                    'type': 'long',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': entry_sl,
                    'initial_stop_loss': entry_sl,
                    'take_profit': entry_tp,
                    'entry_time': timestamp,
                    'poi_quality': sniper_result.get('poi_quality', 'UNKNOWN')
                }
                tp_str = f" | TP: {entry_tp:.2f}" if entry_tp else ""
                print(f"[{timestamp}] BUY LONG @ {price:.2f} ({source}: BUY | Regime: {current_regime} | SL: {entry_sl:.2f}{tp_str} | POI: {self.position['poi_quality']})")

            # SHORT ENTRY
            elif ai_decision == "SELL":
                equity = self.balance
                size_asset = (equity * 0.99) / price
                
                notional_value = size_asset * price
                fee = notional_value * self.commission_rate
                
                if fee > self.balance:
                     return
                
                self.balance -= fee 
                
                entry_sl = sniper_sl if sniper_sl > 0 else stop_short
                entry_tp = sniper_tp if sniper_tp > 0 else None
                
                self.position = {
                    'type': 'short',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': entry_sl,
                    'initial_stop_loss': entry_sl,
                    'take_profit': entry_tp,
                    'entry_time': timestamp,
                    'poi_quality': sniper_result.get('poi_quality', 'UNKNOWN')
                }
                tp_str = f" | TP: {entry_tp:.2f}" if entry_tp else ""
                print(f"[{timestamp}] SELL SHORT @ {price:.2f} ({source}: SELL | Regime: {current_regime} | SL: {entry_sl:.2f}{tp_str} | POI: {self.position['poi_quality']})")

    def _close_position(self, price, timestamp, reason):
        if not self.position: return
        
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        # Calculate Economics
        if self.position['type'] == 'long':
            # Sell Asset -> Cash
            revenue = size * price
            fee = revenue * self.commission_rate
            
            # Cash In
            self.balance += (revenue - fee)
            
            # PnL (for stats)
            # We assume entry fee was paid from balance at start
            # entry cost was size * entry_price
            # PnL = (revenue - fee) - (entry_cost + entry_fee)
            # But entry_fee is already gone from balance.
            # So Delta Balance = (revenue - fee) - entry_cost? 
            # No, Delta Balance is just the PnL of the round trip.
            # Let's verify:
            # Start: Bal = 10000.
            # Buy: Cost 9000. Fee 9. Bal = 991. Position = 9000 val.
            # Sell: Rev 9500. Fee 9.5. Bal += 9490.5 -> 10481.5.
            # Total Profit = 481.5.
            # Calc: (9500 - 9000) - 9 - 9.5 = 500 - 18.5 = 481.5. Correct.
            
            gross_pnl = (price - entry_price) * size
            entry_fee = (size * entry_price) * self.commission_rate
            net_pnl = gross_pnl - fee - entry_fee
            
        else: # SHORT
            # Cover Short
            # PnL = (Entry - Exit) * Size
            gross_pnl = (entry_price - price) * size
            
            cover_cost = size * price
            exit_fee = cover_cost * self.commission_rate
            
            # Update Balance (which held Collateral)
            # Balance += Net PnL - Exit Fee
            # (Note: Entry fee was already deducted)
            
            net_pnl = gross_pnl - exit_fee 
            
            # Note on PnL reporting: we want to subtract entry fee too for 'Net Trade PnL'
            entry_fee = (size * entry_price) * self.commission_rate
            # But entry fee was ALREADY deducted from balance.
            
            # So to update Balance correctly:
            # We just add the PnL from the price movement and subtract exit fee.
            self.balance += (gross_pnl - exit_fee)
            
            # For reported trade PnL statistic, we include entry fee
            trade_pnl_report = gross_pnl - exit_fee - entry_fee
            net_pnl = trade_pnl_report # Use this for the logs

        # Track exit reason for forensic log
        self._last_exit_reason = reason
        
        self.trades.append({
            'type': self.position['type'],
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'entry_price': entry_price,
            'exit_price': price,
            'pnl': net_pnl,
            'reason': reason,
            'poi_quality': self.position.get('poi_quality', 'UNKNOWN')
        })
        
        print(f"[{timestamp}] CLOSE {self.position['type'].upper()} @ {price:.2f} ({reason}). PnL: ${net_pnl:.2f}")
        self.position = None

    def _generate_report(self):
        print("\n--- PERFORMANCE REPORT ---")
        percent_return = ((self.balance - self.initial_capital) / self.initial_capital) * 100
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        losses = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Max Drawdown
        equity_series = pd.DataFrame(self.equity_curve)
        max_dd = 0.0
        if not equity_series.empty:
            equity_series['peak'] = equity_series['equity'].cummax()
            equity_series['dd'] = (equity_series['peak'] - equity_series['equity']) / equity_series['peak']
            max_dd = equity_series['dd'].max() * 100
        
        # ── Cabecera de Aislamiento ─────────────────────────────────
        isolation_header = ""
        if self.isolation_enabled:
            isolation_header = f"\n--- AISLAMIENTO ACTIVO: MODO {self.isolation_target.upper()} ---\n"
            
        report_str = f"""{isolation_header}
--- PERFORMANCE REPORT ---
Final Balance:  ${self.balance:,.2f}
Total Return:   {percent_return:.2f}%
Win Rate:       {win_rate:.2f}%
Profit Factor:  {profit_factor:.2f}
Max Drawdown:   {max_dd:.2f}%
Total Trades:   {len(self.trades)}
"""
        print(report_str)
        
        # Nombre dinámico de archivo según modo
        if self.isolation_enabled:
            report_file = f'backtest_report_ISOLATION_{self.isolation_target}.txt'
            chart_file = f'equity_curve_ISOLATION_{self.isolation_target}.png'
        else:
            report_file = 'backtest_report.txt'
            chart_file = 'equity_curve.png'
        
        with open(report_file, 'w') as f:
            f.write(report_str)
        print(f"Report saved to {report_file}")
        
        if not equity_series.empty:
            plt.figure(figsize=(10,6))
            plt.plot(equity_series['timestamp'], equity_series['equity'])
            title = f"Backtest: Return {percent_return:.2f}% | PF {profit_factor:.2f}"
            if self.isolation_enabled:
                title = f"[ISOLATION: {self.isolation_target.upper()}] {title}"
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Capital ($)")
            plt.grid(True)
            plt.savefig(chart_file)
            print(f"Chart saved to {chart_file}")

    def _save_forensic_log(self):
        """Guarda brutal_analysis_log.csv con diagnóstico forense de cada vela."""
        if not self.forensic_log:
            print("[Forensic] No hay datos para guardar.")
            return
        
        csv_path = 'brutal_analysis_log.csv'
        fieldnames = [
            'Fecha', 'Precio', 'EMA200', 'Regimen', 'Hurst_Score',
            'Señal_CarryTrade', 'Confianza_CarryTrade',
            'Señal_SMC', 'Confianza_SMC', 'Estructura_SMC', 'OB_Tipo_SMC',
            'Decision_Claude', 'Consenso_Recomendacion', 'Confianza_Consenso',
            'Posicion_Activa', 'Razon_de_Salida', 'PnL_Trade',
            'Culpable_Perdida', 'FLAG_NOISE', 'POI_Quality'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.forensic_log)
        
        # ── Summary Stats ───────────────────────────────────────────
        total_rows = len(self.forensic_log)
        noise_rows = sum(1 for r in self.forensic_log if r['FLAG_NOISE'])
        loss_rows = [r for r in self.forensic_log if r['Culpable_Perdida']]
        noise_losses = [r for r in loss_rows if '⚠️NOISE_REGIME' in r.get('Culpable_Perdida', '')]
        
        print(f"\n{'='*60}")
        print(f"  BRUTAL ANALYSIS LOG — FORENSIC SUMMARY")
        print(f"{'='*60}")
        print(f"  Archivo guardado: {csv_path}")
        print(f"  Total velas analizadas: {total_rows}")
        print(f"  Velas en NOISE:         {noise_rows} ({noise_rows/total_rows*100:.1f}%)")
        print(f"  Trades con pérdida:     {len(loss_rows)}")
        print(f"  Pérdidas en NOISE:      {len(noise_losses)}")
        if loss_rows:
            print(f"\n  --- CULPABLES POR PÉRDIDA ---")
            for r in loss_rows:
                print(f"  [{r['Fecha']}] PnL: ${r['PnL_Trade']} | Régimen: {r['Regimen']} | Culpable: {r['Culpable_Perdida']}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    eng = BacktestEngine()
    eng.run()
