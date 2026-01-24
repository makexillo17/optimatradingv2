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
    def __init__(self, data_file='data/btc_history.csv', initial_capital=10000.0, commission_rate=0.001):
        self.data_file = data_file
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.commission_rate = commission_rate
        self.equity_curve = []
        self.trades = []
        self.position = None 
        self.verbose = True # ACTIVAR MODO VERBOSE
        
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
        
        start_idx = 205
        
        for i in range(start_idx, len(df)):
            current_df = df.iloc[:i+1].copy()
            current_candle = current_df.iloc[-1]
            current_price = current_candle['close']
            current_time = current_candle['timestamp']
            
            # 1. Run Analysis
            module_results = {}
            market_input = {'market_data': current_df}
            
            for name, module in self.modules.items():
                try:
                    module_results[name] = module.analyze(market_input)
                except Exception:
                    pass
            
            # 2. Consensus
            consensus_result = self.consensus.analyze(module_results)
            # Extraer signal numérico si existe, o usar lógica interna de consensus
            # El consensus retorna detalles['avg_signal'] en algunos casos o 'signal' en el return directo
            consensus_signal = consensus_result.get('signal', consensus_result.get('details', {}).get('avg_signal', 0.0))
            recommendation = consensus_result.get('recommendation', 'NEUTRAL')
            
            # --- DEBUGGING / VERBOSE ---
            if self.verbose:
                # Filtrar solo casos interesantes (cerca de trigger o muy negativos)
                if abs(consensus_signal) > 0.3:
                    smc_rec = module_results.get('smc_ict', {}).get('recommendation', 'neutral')
                    vsa_rec = module_results.get('broker_behavior', {}).get('recommendation', 'neutral')
                    adx_val = module_results.get('carry_trade', {}).get('adx_value', 0.0)
                    
                    print(f"[{current_time}] Score: {consensus_signal:.2f} | Rec: {recommendation} | SMC: {smc_rec} | VSA: {vsa_rec} | ADX: {adx_val:.1f}")
            
            # 3. Risk Params
            hedging_info = module_results.get('dynamic_hedging', {})
            stop_long = hedging_info.get('suggested_stop_long', current_price * 0.95)
            stop_short = hedging_info.get('suggested_stop_short', current_price * 1.05)
            
            # 4. Logic
            self._manage_positions(
                current_price, current_time, 
                consensus_signal, recommendation, 
                stop_long, stop_short, 
                module_results
            )
            
            # Track Equity
            current_equity = self.balance
            if self.position:
                if self.position['type'] == 'long':
                    current_equity += (current_price - self.position['entry_price']) * self.position['size']
                # Short
                # value = size * price. (Borrowed asset value to repay).
                # Equity = Balance + Unrealized PnL.
                # Since Balance adjusted at Entry (simulated cash out), logic holds roughly.
            
            self.equity_curve.append({'timestamp': current_time, 'equity': current_equity})
            
            if i % 1000 == 0:
                print(f"Computed {i}/{len(df)} bars. Equity: ${current_equity:.2f}")

        self._generate_report()

    def _manage_positions(self, price, timestamp, consensus_signal, recommendation, stop_long, stop_short, module_results):
        
        # Check Exits
        if self.position:
            # STOP LOSS
            if self.position['type'] == 'long':
                if price <= self.position['stop_loss']:
                    self._close_position(price, timestamp, "Stop Loss")
                    return
            
            # TAKE PROFIT / REVERSAL
            divergence = module_results.get('yield_anomaly', {}).get('divergence_detected', False)
            div_rec = module_results.get('yield_anomaly', {}).get('recommendation', 'neutral')
            
            if self.position['type'] == 'long':
                if recommendation == 'short' or (divergence and div_rec == 'short'):
                    self._close_position(price, timestamp, "Take Profit/Reversal")
                    return

        # Check Entries
        if not self.position:
            # ENTRY: Consensus > 0.25 (Relaxed from 0.5) OR Score < -0.25 (Short)
            
            # LONG ENTRY
            if consensus_signal > 0.25:
                # Calculate size
                capital = self.balance
                size_asset = (capital * 0.99) / price
                cost = size_asset * price
                fee = cost * self.commission_rate
                
                # Check affordable
                total_cost = cost + fee
                if total_cost > self.balance:
                    size_asset = (self.balance - fee) / price
                
                self.balance -= (size_asset * price + fee)
                
                self.position = {
                    'type': 'long',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_long,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] BUY LONG @ {price:.2f} (Signal: {consensus_signal:.2f})")
                
            # SHORT ENTRY
            elif consensus_signal < -0.25:
                # Simplified Short: We sell asset we don't have (Borrow).
                # Cash (Balance) increases by Price * Size. Debt (Position) is Size.
                # Here we model "Inverse Long" PnL for simplicity or proper Short?
                # Proper Short:
                # Initial Margin = Balance. We sell size worth roughly margin (1x leverage).
                # Cash += Size * Price.
                # Exit: We Buy back. Cash -= Size * Price.
                # Profit = EntryPrice - ExitPrice.
                
                capital = self.balance
                size_asset = (capital * 0.99) / price
                revenue = size_asset * price
                fee = revenue * self.commission_rate
                
                # We need margin to open.
                # If balance is 10k, we sell 10k worth.
                # Cash becomes 20k? (10k collateral + 10k proceeds).
                # Let's keep simpler logic: Position tracks entry. PnL calculated at exit.
                # Balance stays separate or acts as collateral.
                # For CONSISTENCY with Long logic above (Balance -= cost):
                # Let's say we put up COLLATERAL equal to trade value.
                # cost = size * price.
                # balance -= cost + fee. (Collateral locked).
                # Exit: Return collateral +/- pnl.
                
                cost = size_asset * price
                fee = cost * self.commission_rate
                
                if (cost + fee) > self.balance:
                     size_asset = (self.balance - fee) / price
                
                self.balance -= (size_asset * price + fee) # Lock collateral
                
                self.position = {
                    'type': 'short',
                    'entry_price': price,
                    'size': size_asset,
                    'stop_loss': stop_short,
                    'entry_time': timestamp
                }
                print(f"[{timestamp}] SELL SHORT @ {price:.2f} (Signal: {consensus_signal:.2f})")

    def _close_position(self, price, timestamp, reason):
        if not self.position: return
        
        # Calculate PnL based on type
        entry_price = self.position['entry_price']
        size = self.position['size']
        
        # Calculate Release of Capital
        # We locked (Size * Entry) + EntryFee.
        # We need to return: (Size * Entry) +/- PnL - ExitFee
        
        entry_val = size * entry_price
        exit_val = size * price
        
        if self.position['type'] == 'long':
            gross_pnl = exit_val - entry_val
        else:
            gross_pnl = entry_val - exit_val # Short: Profit if Entry > Exit

        exit_fee = exit_val * self.commission_rate
        net_pnl = gross_pnl - exit_fee
        
        # Return Capital (EntryVal) + NetPnL
        # Note: We already deducted entry fee from balance.
        # So we return: EntryVal + NetPnL
        
        returned_capital = entry_val + net_pnl
        self.balance += returned_capital
        
        # For Reporting (Net PnL of the full trade cycle)
        # Cycle PnL = NetPnL - EntryFee
        # We paid EntryFee at start.
        entry_fee = entry_val * self.commission_rate
        trade_pnl_net = net_pnl - entry_fee
        
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': timestamp,
            'type': self.position['type'],
            'entry_price': entry_price,
            'exit_price': price,
            'pnl': trade_pnl_net,
            'reason': reason
        })
        
        print(f"[{timestamp}] CLOSE {self.position['type'].upper()} @ {price:.2f} ({reason}). PnL: ${trade_pnl_net:.2f}")
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
            
        print(f"Final Balance:  ${self.balance:,.2f}")
        print(f"Total Return:   {percent_return:.2f}%")
        print(f"Win Rate:       {win_rate:.2f}%")
        print(f"Profit Factor:  {profit_factor:.2f}")
        print(f"Max Drawdown:   {max_dd:.2f}%")
        print(f"Total Trades:   {len(self.trades)}")
        
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
