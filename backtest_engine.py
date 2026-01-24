import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure root directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
            
            # 1. Run Analysis
            module_results = {}
            market_input = {'market_data': current_df}
            
            for name, module in self.modules.items():
                try:
                    module_results[name] = module.analyze(market_input)
                except Exception:
                    pass
            
            # 2. Consensus
            # Calculate Market Regime
            current_regime = detect_regime(current_df)
            
            consensus_result = self.consensus.analyze(module_results, market_regime=current_regime)
            consensus_signal = consensus_result.get('signal', consensus_result.get('details', {}).get('avg_signal', 0.0))
            recommendation = consensus_result.get('recommendation', 'NEUTRAL')
            
            # Use 'signal' key if available directly
            if 'signal' not in consensus_result:
                if recommendation == 'long': consensus_signal = 0.8
                elif recommendation == 'short': consensus_signal = -0.8
            
            # --- VERBOSE DEBUG ---
            if self.verbose and i % 24 == 0:
                 if abs(consensus_signal) > 0.3 or current_regime == 'NOISE':
                    trend_status = "BULL" if current_price > current_ema200 else "BEAR"
                    print(f"[{current_time}] Regime: {current_regime} | Score: {consensus_signal:.2f} | Trend: {trend_status} | SMC: {module_results.get('smc_ict', {}).get('recommendation')}")

            # 3. Risk Params
            hedging_info = module_results.get('dynamic_hedging', {})
            stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
            stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
            
            # 4. Logic
            self._manage_positions(
                current_price, current_time, 
                consensus_signal, recommendation, 
                stop_long, stop_short, 
                module_results,
                current_ema200
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

        # --- FORCE CLOSE AT END ---
        if self.position:
            last_price = df.iloc[-1]['close']
            last_time = df.iloc[-1]['timestamp']
            print("Force closing remaining position at end of simulation.")
            self._close_position(last_price, last_time, "Force Close (End of Data)")

        self._generate_report()

    def _manage_positions(self, price, timestamp, consensus_signal, recommendation, stop_long, stop_short, module_results, ema200):
        
        # Check Exits
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
            
            if self.position['type'] == 'long':
                if consensus_signal < -0.30 or (divergence and div_rec == 'short'):
                    self._close_position(price, timestamp, "Take Profit/Reversal")
                    return
            elif self.position['type'] == 'short':
                if consensus_signal > 0.30 or (divergence and div_rec == 'long'):
                    self._close_position(price, timestamp, "Take Profit/Reversal")
                    return

        # Check Entries
        # COOLDOWN CHECK and Trend Filter
        if not self.position and self.cooldown == 0:
            
            # LONG ENTRY
            # Signal > 0.30 AND Price > EMA 200
            if consensus_signal > 0.30:
                if price > ema200:
                    capital = self.balance
                    size_asset = (capital * 0.99) / price # Use available cash
                    
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
                    print(f"[{timestamp}] BUY LONG @ {price:.2f} (Score: {consensus_signal:.2f} > 0.3 | > EMA200)")
                else:
                    if self.verbose and consensus_signal > 0.4:
                        print(f"[{timestamp}] FILTERED LONG: Price ({price:.2f}) < EMA200 ({ema200:.2f})")

            # SHORT ENTRY
            # Signal < -0.30 AND Price < EMA 200
            elif consensus_signal < -0.30:
                if price < ema200:
                    equity = self.balance
                    size_asset = (equity * 0.99) / price
                    
                    notional_value = size_asset * price
                    fee = notional_value * self.commission_rate
                    
                    if fee > self.balance:
                         # Not enough for fee
                         return
                    
                    self.balance -= fee 
                    
                    self.position = {
                        'type': 'short',
                        'entry_price': price,
                        'size': size_asset,
                        'stop_loss': stop_short,
                        'entry_time': timestamp
                    }
                    print(f"[{timestamp}] SELL SHORT @ {price:.2f} (Score: {consensus_signal:.2f} < -0.3 | < EMA200)")
                else:
                    if self.verbose and consensus_signal < -0.4:
                         print(f"[{timestamp}] FILTERED SHORT: Price ({price:.2f}) > EMA200 ({ema200:.2f})")

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

if __name__ == "__main__":
    eng = BacktestEngine()
    eng.run()
