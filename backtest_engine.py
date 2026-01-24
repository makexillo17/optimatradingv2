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

from main.consensus import ConsensusAnalyzer

class BacktestEngine:
    def __init__(self, data_file='btc_history.csv', initial_capital=10000.0):
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.equity_curve = []
        self.trades = []
        self.position = None # None or {'type': 'long'/'short', 'entry_price': float, 'size': float, 'stop_loss': float}
        
        self.logger = None # Optional logging

        # Initialize Modules
        self.modules = {
            'smc_ict': SmcIctModule(),
            'broker_behavior': BrokerBehaviorModule(),
            'yield_anomaly': YieldAnomalyModule(),
            'carry_trade': CarryTradeModule(),
            'dynamic_hedging': DynamicHedgingModule(),
            'gap_sniper': GapSniperModule(),
            'volatility_arb': VolatilityArbModule(),
            'stat_arb': StatArbModule(),
            'liquidity_provision': LiquidityProvisionModule(),
            'market_making': MarketMakingModule()
        }
        
        self.consensus = ConsensusAnalyzer()
        
    def load_data(self):
        if not os.path.exists(self.data_file):
            print(f"Error: {self.data_file} not found. Run data_loader.py first.")
            return None
        
        df = pd.read_csv(self.data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Data Loaded: {len(df)} candles.")
        return df

    def run(self):
        df = self.load_data()
        if df is None: return
        
        print("Starting Backtest Simulation... (This may take a few minutes)")
        
        # Start from index 200 to ensure enough history for indicators (EMA200, etc)
        start_idx = 205 
        
        for i in range(start_idx, len(df)):
            # Slice historical data up to current moment (simulating live feed)
            # Optimization: Most libs need full history, so we slice.
            # Warning: This is O(N^2) complexity. For 8000 candles it's acceptable but slow.
            current_df = df.iloc[:i+1].copy() # Copy to avoid SettingWithCopy warnings inside modules
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            current_time = current_candle['timestamp']
            
            # 1. Run Analysis Modules
            module_results = {}
            market_input = {'market_data': current_df}
            
            for name, module in self.modules.items():
                try:
                    module_results[name] = module.analyze(market_input)
                except Exception as e:
                    # print(f"Error in {name}: {e}")
                    pass
            
            # 2. Get Consensus
            consensus_result = self.consensus.analyze(module_results)
            recommendation = consensus_result.get('recommendation', 'NEUTRAL')
            
            # 3. Get Risk Management Info
            hedging_info = module_results.get('dynamic_hedging', {})
            suggested_stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
            suggested_stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
            risk_factor = hedging_info.get('risk_factor', 1.0)
            
            # 4. Portfolio Logic
            self._manage_positions(
                current_price, 
                current_time, 
                recommendation, 
                suggested_stop_long, 
                suggested_stop_short,
                risk_factor,
                module_results
            )
            
            # Record Equity
            current_equity = self.balance
            if self.position:
                # Unrealized PnL
                if self.position['type'] == 'long':
                    pnl = (current_price - self.position['entry_price']) * self.position['size']
                else:
                    pnl = (self.position['entry_price'] - current_price) * self.position['size']
                current_equity += pnl
            
            self.equity_curve.append({'timestamp': current_time, 'equity': current_equity})
            
            if i % 100 == 0:
                print(f"Computed {i}/{len(df)} bars. Equity: ${current_equity:.2f}")

        self._generate_report()

    def _manage_positions(self, price, timestamp, recommendation, stop_long, stop_short, risk_factor, module_results):
        
        # Check Exits first (Stop Loss or Take Profit signals)
        if self.position:
            # STOP LOSS CHECK
            if self.position['type'] == 'long':
                if price <= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            elif self.position['type'] == 'short':
                if price >= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            
            # REVERSAL SIGNALS (Take Profit or Flip)
            # Divergence from Yield Anomaly?
            div_detected = module_results.get('yield_anomaly', {}).get('divergence_detected', False)
            
            if self.position['type'] == 'long':
                if recommendation in ['STRONG_SELL_GAP', 'SELL_TREND'] or (div_detected and module_results['yield_anomaly']['recommendation'] == 'short'):
                    self._close_position(price, timestamp, "Signal Reversal")
                    return
            elif self.position['type'] == 'short':
                if recommendation in ['STRONG_BUY_GAP', 'BUY_TREND'] or (div_detected and module_results['yield_anomaly']['recommendation'] == 'long'):
                    self._close_position(price, timestamp, "Signal Reversal")
                    return

        # Check Entries
        if not self.position:
            # We only enter if we have a strong signal
            # Risk factor check:
            if risk_factor == 0.0: return # Hedge mode / Safe mode
            
            size_usd = self.balance * 0.95 * risk_factor # Use 95% of capital * risk factor
            size_asset = size_usd / price
            
            if recommendation in ['STRONG_BUY_GAP', 'BUY_TREND']:
                self.position = {
                    'type': 'long',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_long,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] OPEN LONG @ {price:.2f} (SL: {stop_long:.2f})")
                
            elif recommendation in ['STRONG_SELL_GAP', 'SELL_TREND']:
                self.position = {
                    'type': 'short',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_short,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] OPEN SHORT @ {price:.2f} (SL: {stop_short:.2f})")

    def _close_position(self, price, timestamp, reason):
        if not self.position: return
        
        pnl = 0
        if self.position['type'] == 'long':
            pnl = (price - self.position['entry_price']) * self.position['size']
        else:
            pnl = (self.position['entry_price'] - price) * self.position['size']
            
        self.balance += pnl
        
        trade_record = {
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': self.position['entry_price'],
            'exit_price': price,
            'pnl': pnl,
            'reason': reason
        }
        self.trades.append(trade_record)
        
        print(f"[{timestamp}] CLOSE {self.position['type'].upper()} @ {price:.2f} ({reason}). PnL: ${pnl:.2f}")
        self.position = None

    def _generate_report(self):
        print("\n--- BACKTEST REPORT ---")
        percent_return = ((self.balance - self.initial_capital) / self.initial_capital) * 100
        
        wins = [t for t in self.trades if t['pnl'] > 0]
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0.0
        
        # Max Drawdown
        equity_series = pd.DataFrame(self.equity_curve)
        if not equity_series.empty:
            equity_series['peak'] = equity_series['equity'].cummax()
            equity_series['drawdown'] = (equity_series['equity'] - equity_series['peak']) / equity_series['peak']
            max_drawdown = equity_series['drawdown'].min() * 100
        else:
            max_drawdown = 0.0
            
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Balance:   ${self.balance:,.2f}")
        print(f"Total Return:    {percent_return:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        print(f"Win Rate:        {win_rate:.2f}%")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        
        # Plot
        if not equity_series.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_series['timestamp'], equity_series['equity'], label='Equity')
            plt.title(f"Backtest Requirements - Return: {percent_return:.2f}%")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.legend()
            plt.grid(True)
            plt.savefig('backtest_result.png')
            print("Chart saved as backtest_result.png")

if __name__ == "__main__":
    engine = BacktestEngine()
    engine.run()
