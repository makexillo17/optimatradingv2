import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import time

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
        
        # Modules
        self.modules = {
            'smc_ict': SmcIctModule(),
            'broker_behavior': BrokerBehaviorModule(),
            'yield_anomaly': YieldAnomalyModule(),
            'dynamic_hedging': DynamicHedgingModule(),
            'gap_sniper': GapSniperModule(),
            'volatility_arb': VolatilityArbModule(),
            'carry_trade': CarryTradeModule(),
            'stat_arb': StatArbModule(),
            'liquidity_provision': LiquidityProvisionModule()
        }
        self.consensus = ConsensusAnalyzer()

        # AI Engine
        self.trader = ClaudeTrader()
        
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
            
            # 1. Run Analysis (modules still used for exits / risk)
            module_results = {}
            market_input = {'market_data': current_df}
            
            for name, module in self.modules.items():
                try:
                    module_results[name] = module.analyze(market_input)
                except Exception:
                    pass
            
            # 2. Consensus (kept for exit signals & regime detection)
            current_regime = detect_regime(current_df)
            
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

            # ── VEDA TOTAL: NOISE = HOLD FORZADO ──────────────────────
            if current_regime == 'NOISE':
                if ai_decision != "HOLD":
                    print(f"[{current_time}] 🚫 VEDA NOISE: Claude dijo {ai_decision} → forzado a HOLD")
                ai_decision = "HOLD"

            # --- VERBOSE DEBUG ---
            if self.verbose and i % 24 == 0:
                 if abs(consensus_signal) > 0.3 or current_regime == 'NOISE':
                    trend_status = "BULL" if current_price > current_ema200 else "BEAR"
                    print(f"[{current_time}] Regime: {current_regime} | Score: {consensus_signal:.2f} | Trend: {trend_status} | SMC: {module_results.get('smc_ict', {}).get('recommendation')}")

            # 4. Risk Params
            hedging_info = module_results.get('dynamic_hedging', {})
            stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
            stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
            
            # 5. Position Management (entries driven by AI, exits by modules)
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
                # Find which module likely caused a losing trade
                last_trade = self.trades[-1] if self.trades else None
                if last_trade and last_trade['pnl'] < 0:
                    # Determine blame: was the entry signal wrong?
                    trade_type = last_trade['type']  # 'long' or 'short'
                    if trade_type == 'long':
                        # Entered long — who recommended long?
                        if carry_signal == 'long':
                            culprit += 'CarryTrade(long) '
                        if smc_signal == 'long':
                            culprit += 'SMC(long) '
                        culprit += f'Claude(BUY) '
                    elif trade_type == 'short':
                        if carry_signal == 'short':
                            culprit += 'CarryTrade(short) '
                        if smc_signal == 'short':
                            culprit += 'SMC(short) '
                        culprit += f'Claude(SELL) '
                    if is_noise:
                        culprit += '⚠️NOISE_REGIME '
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
        
        # ── Check Exits (unchanged — driven by modules & risk params) ──
        if self.position:
            # STOP LOSS
            if self.position['type'] == 'long':
                if price <= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            elif self.position['type'] == 'short':
                if price >= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            
            # TAKE PROFIT / REVERSAL
            divergence = module_results.get('yield_anomaly', {}).get('divergence_detected', False)
            div_rec = module_results.get('yield_anomaly', {}).get('recommendation', 'neutral')
            
            # ── TAKE PROFIT MÍNIMO (ATR-Based) ─────────────────────
            # No cerrar posición ganadora si la ganancia < 1.0 * ATR
            current_atr = module_results.get('dynamic_hedging', {}).get('current_atr', 0)
            min_profit_threshold = current_atr * 1.0  # 1 ATR mínimo de ganancia
            
            if self.position['type'] == 'long':
                unrealized_per_unit = price - self.position['entry_price']
                unrealized_pnl = unrealized_per_unit * self.position['size']
                
                # Si hay ganancia pero es menor al ATR, bloquear cierre por señal
                if unrealized_per_unit > 0 and unrealized_per_unit < min_profit_threshold and current_atr > 0:
                    print(f"[{timestamp}] 🔒 TP MÍNIMO: Ganancia ${unrealized_pnl:.2f} < ATR ${min_profit_threshold:.2f} — bloqueando cierre")
                elif ai_decision == "SELL" or consensus_signal < -0.50 or (divergence and div_rec == 'short'):
                    self._close_position(price, timestamp, f"AI/Reversal (Claude: {ai_decision})")
                    return
                    
            elif self.position['type'] == 'short':
                unrealized_per_unit = self.position['entry_price'] - price
                unrealized_pnl = unrealized_per_unit * self.position['size']
                
                if unrealized_per_unit > 0 and unrealized_per_unit < min_profit_threshold and current_atr > 0:
                    print(f"[{timestamp}] 🔒 TP MÍNIMO: Ganancia ${unrealized_pnl:.2f} < ATR ${min_profit_threshold:.2f} — bloqueando cierre")
                elif ai_decision == "BUY" or consensus_signal > 0.50 or (divergence and div_rec == 'long'):
                    self._close_position(price, timestamp, f"AI/Reversal (Claude: {ai_decision})")
                    return

        # ── Check Entries (driven by Claude AI decision) ──────────────
        if not self.position and self.cooldown == 0 and current_regime != 'NOISE':
            
            # LONG ENTRY — Claude says BUY
            if ai_decision == "BUY":
                capital = self.balance
                size_asset = (capital * 0.99) / price
                
                cost = size_asset * price
                fee = cost * self.commission_rate
                
                if (cost + fee) > self.balance:
                    size_asset = (self.balance - fee) / price
                
                self.balance -= (size_asset * price + fee)
                
                self.position = {
                    'type': 'long',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_long,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] BUY LONG @ {price:.2f} (Claude: BUY | Regime: {current_regime})")

            # SHORT ENTRY — Claude says SELL
            elif ai_decision == "SELL":
                equity = self.balance
                size_asset = (equity * 0.99) / price
                
                notional_value = size_asset * price
                fee = notional_value * self.commission_rate
                
                if fee > self.balance:
                     return
                
                self.balance -= fee 
                
                self.position = {
                    'type': 'short',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_short,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] SELL SHORT @ {price:.2f} (Claude: SELL | Regime: {current_regime})")

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
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': entry_price,
            'exit_price': price,
            'pnl': net_pnl,
            'reason': reason
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
            
        report_str = f"""
--- PERFORMANCE REPORT ---
Final Balance:  ${self.balance:,.2f}
Total Return:   {percent_return:.2f}%
Win Rate:       {win_rate:.2f}%
Profit Factor:  {profit_factor:.2f}
Max Drawdown:   {max_dd:.2f}%
Total Trades:   {len(self.trades)}
"""
        print(report_str)
        with open('backtest_report.txt', 'w') as f:
            f.write(report_str)
        
        if not equity_series.empty:
            plt.figure(figsize=(10,6))
            plt.plot(equity_series['timestamp'], equity_series['equity'])
            plt.title(f"Backtest: Return {percent_return:.2f}% | PF {profit_factor:.2f}")
            plt.xlabel("Date")
            plt.ylabel("Capital ($)")
            plt.grid(True)
            plt.savefig('equity_curve.png')
            print("Chart saved to equity_curve.png")

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
            'Culpable_Perdida', 'FLAG_NOISE'
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
