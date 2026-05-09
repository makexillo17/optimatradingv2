import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import time
import yaml

from dataclasses import dataclass, field
from queue import PriorityQueue
from datetime import timedelta

# Tipos de Eventos
MARKET_TICK = 1
SIGNAL_GENERATED = 2
ORDER_PLACED = 3
ORDER_FILLED = 4

@dataclass(order=True)
class Event:
    timestamp: pd.Timestamp
    priority: int
    event_type: int = field(compare=False)
    data: dict = field(compare=False, default_factory=dict)


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
from modulos.feedback_loop import FeedbackLoop
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
        self.feedback = FeedbackLoop(bored_threshold_candles=48)
        
        # ── Event-Driven State ──
        self.events = PriorityQueue()
        self.latency_ms = 50
        self.slippage_model = "volatility_adaptive"
        self.total_latency_loss = 0.0


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

            sim_settings = cfg.get('simulation_settings', {})
            self.latency_ms = sim_settings.get('simulated_latency_ms', 50)
            self.slippage_model = sim_settings.get('slippage_model', "volatility_adaptive")

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
        
        print("Starting Event-Driven Simulation...")
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        start_idx = 205
        self.cooldown = 0
        self.current_df_full = df
        
        # ── 1. Enqueue MARKET_TICK events ──
        for i in range(start_idx, len(df)):
            candle = df.iloc[i]
            t = candle['timestamp']
            
            # Interpolador de Ticks (4 por vela)
            self.events.put(Event(t, 1, MARKET_TICK, {'price': candle['open'], 'tick_type': 'open', 'idx': i, 'candle': candle}))
            self.events.put(Event(t + timedelta(minutes=15), 1, MARKET_TICK, {'price': candle['high'], 'tick_type': 'high', 'idx': i, 'candle': candle}))
            self.events.put(Event(t + timedelta(minutes=30), 1, MARKET_TICK, {'price': candle['low'], 'tick_type': 'low', 'idx': i, 'candle': candle}))
            self.events.put(Event(t + timedelta(minutes=59, seconds=59), 1, MARKET_TICK, {'price': candle['close'], 'tick_type': 'close', 'idx': i, 'candle': candle}))

        # ── 2. Event Loop ──
        while not self.events.empty():
            event = self.events.get()
            
            if event.event_type == MARKET_TICK:
                self._handle_market_tick(event)
            elif event.event_type == SIGNAL_GENERATED:
                self._handle_signal(event)
            elif event.event_type == ORDER_PLACED:
                self._handle_order_placed(event)
                
        # Force Close at end
        if self.position:
            last_price = df.iloc[-1]['close']
            last_time = df.iloc[-1]['timestamp']
            self._close_position(last_price, last_time, "Force Close (End of Data)")

        self._save_forensic_log()
        self._generate_report()

    def _handle_market_tick(self, event):
        data = event.data
        price = data['price']
        timestamp = event.timestamp
        idx = data['idx']
        tick_type = data['tick_type']
        
        self.current_price = price
        self.current_time = timestamp
        self.current_idx = idx
        
        # 1. Manage Exits first (SL/TP)
        self._manage_exits(price, timestamp)
        
        # 2. Generar Señal al Cierre de Vela
        if tick_type == 'close':
            if self.cooldown > 0:
                self.cooldown -= 1
            
            # Look-ahead Bias Protection: slice precisely up to this candle
            current_df = self.current_df_full.iloc[:idx+1].copy()
            self.events.put(Event(timestamp, 2, SIGNAL_GENERATED, {'df': current_df, 'candle': data['candle']}))

    def _handle_signal(self, event):
        current_df = event.data['df']
        current_candle = event.data['candle']
        current_time = event.timestamp
        current_price = current_candle['close']
        current_ema200 = current_candle['ema200']
        
        self.feedback.update_mood()
        self._last_exit_reason = None
        market_input = {'market_data': current_df}
        current_regime = detect_regime(current_df)
        
        module_results = {}
        ai_decision = "HOLD"
        consensus_signal = 0.0
        recommendation = "NEUTRAL"
        
        if self.isolation_enabled:
            target = self.isolation_target
            if target in self.modules:
                try:
                    if current_regime == 'NOISE' and target == 'gap_sniper':
                        self.modules['gap_sniper'].volume_threshold = self.volume_threshold * 0.8
                    module_results[target] = self.modules[target].analyze(market_input)
                except Exception: pass
                
            if target != 'dynamic_hedging' and 'dynamic_hedging' in self.modules:
                try: module_results['dynamic_hedging'] = self.modules['dynamic_hedging'].analyze(market_input)
                except Exception: pass
                
            if target != 'volatility_guard' and 'volatility_guard' in self.modules:
                try: module_results['volatility_guard'] = self.modules['volatility_guard'].analyze(market_input)
                except Exception: pass
                
            engine_result = module_results.get(target, {})
            engine_rec = engine_result.get('recommendation', 'neutral')
            engine_conf = engine_result.get('confidence', 0.0)
            
            if engine_rec == 'long' and engine_conf > 0.0: ai_decision = "BUY"
            elif engine_rec == 'short' and engine_conf > 0.0: ai_decision = "SELL"
            
            if ai_decision == "BUY" and current_price < current_ema200: ai_decision = "HOLD"
            elif ai_decision == "SELL" and current_price > current_ema200: ai_decision = "HOLD"
            
            vg = module_results.get('volatility_guard', {})
            if vg.get('force_hold'): ai_decision = "HOLD"
            elif ai_decision == "BUY" and vg.get('block_buy'): ai_decision = "HOLD"
            elif ai_decision == "SELL" and vg.get('block_sell'): ai_decision = "HOLD"
            
            if current_regime == 'NOISE' and target in ('carry_trade',): ai_decision = "HOLD"
            recommendation = engine_rec
            
        else:
            for name, module in self.modules.items():
                try:
                    if current_regime == 'NOISE' and name == 'gap_sniper':
                        self.modules['gap_sniper'].volume_threshold = self.volume_threshold * 0.8
                    module_results[name] = module.analyze(market_input)
                except Exception: pass
                
            consensus_result = self.consensus.analyze(module_results, market_regime=current_regime)
            consensus_signal = consensus_result.get('signal', consensus_result.get('details', {}).get('avg_signal', 0.0))
            recommendation = consensus_result.get('recommendation', 'NEUTRAL')
            
            if 'signal' not in consensus_result:
                if recommendation == 'long': consensus_signal = 0.8
                elif recommendation == 'short': consensus_signal = -0.8
                
            current_row = {
                'timestamp': str(current_time), 'open': float(current_candle['open']),
                'high': float(current_candle['high']), 'low': float(current_candle['low']),
                'close': float(current_price), 'volume': float(current_candle['volume']),
                'ema200': float(current_ema200), 'regime': current_regime,
                'consensus_signal': float(consensus_signal),
                'obi_score': float(module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0))
            }
            try: ai_decision = self.trader.analyze_market_data(current_row, market_regime=current_regime)
            except Exception: ai_decision = "HOLD"
            
            if current_regime == 'NOISE' and ai_decision != "HOLD": ai_decision = "HOLD"
            if ai_decision == "BUY" and current_price < current_ema200: ai_decision = "HOLD"
            elif ai_decision == "SELL" and current_price > current_ema200: ai_decision = "HOLD"

        self.last_module_results = module_results
        self.last_ai_decision = ai_decision
        self.last_consensus_signal = consensus_signal
        
        hedging_info = module_results.get('dynamic_hedging', {})
        stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
        stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
        self.current_atr = hedging_info.get('current_atr', current_price * 0.01)

        # Update Equity Curve
        current_equity = self.balance
        if self.position:
            if self.position['type'] == 'long': current_equity = self.balance + (self.position['size'] * current_price)
            elif self.position['type'] == 'short': current_equity = self.balance + ((self.position['entry_price'] - current_price) * self.position['size'])
        self.equity_curve.append({'timestamp': current_time, 'equity': current_equity})

        # Forensic Log Update
        self._append_forensic_log(current_time, current_price, current_ema200, current_regime, consensus_signal, ai_decision, recommendation, module_results, consensus_result)

        # Place Order Event if needed
        noise_allowed = self.isolation_enabled and self.isolation_target in ('gap_sniper', 'smc_ict')
        if not self.position and self.cooldown == 0 and (current_regime != 'NOISE' or noise_allowed):
            sniper_result = module_results.get('gap_sniper', module_results.get(self.isolation_target, {}))
            sniper_tp = sniper_result.get('take_profit', 0.0)
            sniper_sl = sniper_result.get('stop_loss', 0.0)
            
            if ai_decision in ("BUY", "SELL"):
                order_data = {
                    'direction': ai_decision,
                    'sniper_sl': sniper_sl, 'stop_long': stop_long, 'stop_short': stop_short,
                    'sniper_tp': sniper_tp, 'poi_quality': sniper_result.get('poi_quality', 'UNKNOWN')
                }
                exec_time = current_time + timedelta(milliseconds=self.latency_ms)
                self.events.put(Event(exec_time, 3, ORDER_PLACED, order_data))

    def _handle_order_placed(self, event):
        if self.position: return
        order = event.data
        timestamp = event.timestamp
        market_price = self.current_price
        
        # Slippage Dinamico + Latency Impact
        atr = self.current_atr if hasattr(self, 'current_atr') and self.current_atr > 0 else market_price * 0.01
        slippage_pct = (atr / market_price) * 0.05
        
        if order['direction'] == 'BUY':
            execution_price = market_price * (1 + slippage_pct)
            latency_loss_per_unit = execution_price - market_price
            
            capital = self.balance
            size_asset = (capital * 0.99) / execution_price
            total_loss = latency_loss_per_unit * size_asset
            self.total_latency_loss += total_loss
            
            fee = size_asset * execution_price * self.commission_rate
            if (size_asset * execution_price + fee) > self.balance: 
                size_asset = (self.balance - fee) / execution_price
            self.balance -= (size_asset * execution_price + fee)
            
            entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_long']
            self.position = {
                'type': 'long', 'entry_price': execution_price, 'size': size_asset,
                'stop_loss': entry_sl, 'initial_stop_loss': entry_sl,
                'take_profit': order['sniper_tp'] if order['sniper_tp'] > 0 else None,
                'entry_time': timestamp, 'poi_quality': order['poi_quality']
            }
            print(f"[{timestamp}] BUY LONG @ {execution_price:.2f} (Slippage/Latency Loss: ${total_loss:.2f})")
            self.feedback.reset_inactivity()
            
        elif order['direction'] == 'SELL':
            execution_price = market_price * (1 - slippage_pct)
            latency_loss_per_unit = market_price - execution_price
            
            equity = self.balance
            size_asset = (equity * 0.99) / execution_price
            total_loss = latency_loss_per_unit * size_asset
            self.total_latency_loss += total_loss
            
            fee = size_asset * execution_price * self.commission_rate
            if fee > self.balance: return
            self.balance -= fee 
            
            entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_short']
            self.position = {
                'type': 'short', 'entry_price': execution_price, 'size': size_asset,
                'stop_loss': entry_sl, 'initial_stop_loss': entry_sl,
                'take_profit': order['sniper_tp'] if order['sniper_tp'] > 0 else None,
                'entry_time': timestamp, 'poi_quality': order['poi_quality']
            }
            print(f"[{timestamp}] SELL SHORT @ {execution_price:.2f} (Slippage/Latency Loss: ${total_loss:.2f})")
            self.feedback.reset_inactivity()

    def _manage_exits(self, price, timestamp):
        if not self.position: return
        
        # Break-Even Logic
        if not self.position.get('break_even_triggered', False):
            initial_sl = self.position.get('initial_stop_loss', 0)
            entry_price = self.position['entry_price']
            if self.position['type'] == 'long' and initial_sl > 0:
                if price - entry_price >= (entry_price - initial_sl) * self.break_even_ratio:
                    self.position['stop_loss'] = entry_price
                    self.position['break_even_triggered'] = True
            elif self.position['type'] == 'short' and initial_sl > 0:
                if entry_price - price >= (initial_sl - entry_price) * self.break_even_ratio:
                    self.position['stop_loss'] = entry_price
                    self.position['break_even_triggered'] = True

        # Stop Loss
        if self.position['type'] == 'long' and price <= self.position['stop_loss']:
            self._close_position(price, timestamp, "Stop Loss")
            return
        elif self.position['type'] == 'short' and price >= self.position['stop_loss']:
            self._close_position(price, timestamp, "Stop Loss")
            return
            
        # Take Profit
        if self.position.get('take_profit'):
            tp = self.position['take_profit']
            if self.position['type'] == 'long' and price >= tp:
                self._close_position(price, timestamp, "Take Profit")
                return
            elif self.position['type'] == 'short' and price <= tp:
                self._close_position(price, timestamp, "Take Profit")
                return
                
        # Reversal
        if hasattr(self, 'last_module_results'):
            ai_decision = self.last_ai_decision
            consensus_signal = self.last_consensus_signal
            divergence = self.last_module_results.get('yield_anomaly', {}).get('divergence_detected', False)
            div_rec = self.last_module_results.get('yield_anomaly', {}).get('recommendation', 'neutral')
            
            if self.position['type'] == 'long':
                if ai_decision == "SELL" or consensus_signal < -0.50 or (divergence and div_rec == 'short'):
                    self._close_position(price, timestamp, f"AI/Reversal ({ai_decision})")
            elif self.position['type'] == 'short':
                if ai_decision == "BUY" or consensus_signal > 0.50 or (divergence and div_rec == 'long'):
                    self._close_position(price, timestamp, f"AI/Reversal ({ai_decision})")

    def _append_forensic_log(self, current_time, current_price, current_ema200, current_regime, consensus_signal, ai_decision, recommendation, module_results, consensus_result):
        carry_result = module_results.get('carry_trade', {})
        smc_result = module_results.get('smc_ict', {})
        is_noise = current_regime == 'NOISE'
        
        culprit = ''
        if self._last_exit_reason:
            last_trade = self.trades[-1] if self.trades else None
            if last_trade:
                trade_dir = 'BUY' if last_trade['type'] == 'long' else 'SELL'
                if self.isolation_enabled: culprit = f'{self.isolation_target}({trade_dir})'
                else: culprit = f"Claude({trade_dir})"
                if is_noise: culprit += ' ⚠️NOISE_REGIME'
                
        self.forensic_log.append({
            'Fecha': str(current_time), 'Precio': round(current_price, 2),
            'EMA200': round(current_ema200, 2), 'Regimen': current_regime,
            'Hurst_Score': round(consensus_signal, 4),
            'Señal_CarryTrade': carry_result.get('recommendation', 'N/A'),
            'Confianza_CarryTrade': round(carry_result.get('confidence', 0.0), 2),
            'Señal_SMC': smc_result.get('recommendation', 'N/A'),
            'Confianza_SMC': round(smc_result.get('confidence', 0.0), 2),
            'Estructura_SMC': smc_result.get('structure', 'N/A'),
            'OB_Tipo_SMC': smc_result.get('nearest_ob_type', 'None'),
            'OBI_Score': round(smc_result.get('details', {}).get('obi_score', 0.0), 4),
            'Decision_Claude': ai_decision,
            'Consenso_Recomendacion': recommendation,
            'Confianza_Consenso': round(consensus_result.get('confidence', 0.0) if type(consensus_result) == dict else 0.0, 4),
            'Posicion_Activa': self.position['type'] if self.position else 'NINGUNA',
            'Razon_de_Salida': self._last_exit_reason if self._last_exit_reason else '',
            'PnL_Trade': round(self.trades[-1]['pnl'], 2) if (self._last_exit_reason and self.trades) else '',
            'Culpable_Perdida': culprit,
            'FLAG_NOISE': '⚠️ PROHIBIR_OPERAR' if is_noise else '',
            'POI_Quality': self.position.get('poi_quality', '') if self.position else (self.trades[-1].get('poi_quality', '') if (self._last_exit_reason and self.trades) else ''),
            'Lesson_Learned': self.trades[-1].get('lesson', '') if (self._last_exit_reason and self.trades) else ''
        })

    def _close_position(self, price, timestamp, reason):
        if not self.position: return
        entry_price = self.position['entry_price']
        size = self.position['size']

        market_price = price
        atr = self.current_atr if hasattr(self, 'current_atr') and self.current_atr > 0 else market_price * 0.01
        slippage_pct = (atr / market_price) * 0.05
        
        if self.position['type'] == 'long':
            price = market_price * (1 - slippage_pct)
            latency_loss_per_unit = market_price - price
            self.total_latency_loss += latency_loss_per_unit * size
        else:
            price = market_price * (1 + slippage_pct)
            latency_loss_per_unit = price - market_price
            self.total_latency_loss += latency_loss_per_unit * size

        
        if self.position['type'] == 'long':
            revenue = size * price
            fee = revenue * self.commission_rate
            self.balance += (revenue - fee)
            gross_pnl = (price - entry_price) * size
            entry_fee = (size * entry_price) * self.commission_rate
            net_pnl = gross_pnl - fee - entry_fee
        else: # SHORT
            gross_pnl = (entry_price - price) * size
            cover_cost = size * price
            exit_fee = cover_cost * self.commission_rate
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
Latency Loss:   ${self.total_latency_loss:,.2f}
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
            'Culpable_Perdida', 'FLAG_NOISE', 'POI_Quality', 'Lesson_Learned', 'OBI_Score'
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
