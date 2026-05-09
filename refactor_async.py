import re
import os

def refactor_llm_client():
    with open('llm_client.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = content.replace("from anthropic import Anthropic", "from anthropic import Anthropic, AsyncAnthropic")
    content = content.replace("self.client = Anthropic(api_key=self.api_key)", "self.client = AsyncAnthropic(api_key=self.api_key)")
    content = content.replace("def analyze_market_data(", "async def analyze_market_data(")
    content = content.replace("message = self.client.messages.create(", "message = await self.client.messages.create(")
    
    with open('llm_client.py', 'w', encoding='utf-8') as f:
        f.write(content)

def refactor_backtest_engine():
    with open('backtest_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Imports
    if "import asyncio" not in content:
        content = content.replace("from queue import PriorityQueue", "from queue import PriorityQueue\nimport asyncio\nfrom collections import deque")

    # Change def run to async def run
    content = content.replace("    def run(self):", "    async def run(self):")
    
    # Change handle_signal to async
    content = content.replace("    def _handle_signal(self, event):", "    async def _handle_signal(self, event):")
    
    # In run loop: await _handle_signal
    content = content.replace("self._handle_signal(event)", "await self._handle_signal(event)")

    # Inside _handle_signal: await trader.analyze
    content = content.replace("ai_decision = self.trader.analyze_market_data(current_row, market_regime=current_regime)",
                              "ai_decision = await self.trader.analyze_market_data(current_row, market_regime=current_regime)")

    # Asyncio Consensus
    consensus_sync = "consensus_result = self.consensus.analyze(module_results, market_regime=current_regime)"
    consensus_async = "consensus_result = await asyncio.get_event_loop().run_in_executor(None, lambda: self.consensus.analyze(module_results, market_regime=current_regime))"
    content = content.replace(consensus_sync, consensus_async)

    # Deque implementation
    if "self.current_df_full = deque(" not in content:
        content = content.replace("self.current_df_full = df", "self.current_df_full = df")

    # Monitor Position Task (Zero Latency Exit)
    # We add a global event for ticks to awake the monitor
    if "self.tick_event =" not in content:
        init_addition = """
        self.tick_event = asyncio.Event()
        self.monitor_task = None"""
        content = content.replace("self.total_latency_loss = 0.0", "self.total_latency_loss = 0.0\n" + init_addition)

    if "self.monitor_task = asyncio.create_task" not in content:
        run_start = "        self.cooldown = 0"
        run_new = """        self.cooldown = 0
        self.tick_event.clear()
        self.monitor_task = asyncio.create_task(self._monitor_position())"""
        content = content.replace(run_start, run_new)
        
    # _handle_market_tick updates event
    content = content.replace("self.current_idx = idx", "self.current_idx = idx\n        self.tick_event.set()")

    # Stop Monitor Task at the end
    end_run = "self._generate_report()"
    end_run_new = "self._generate_report()\n        if self.monitor_task:\n            self.monitor_task.cancel()"
    content = content.replace(end_run, end_run_new)

    monitor_def = """
    async def _monitor_position(self):
        # Tarea de salida de latencia cero
        while True:
            try:
                await self.tick_event.wait()
                self.tick_event.clear()
                
                if self.position:
                    price = self.current_price
                    timestamp = self.current_time
                    
                    # Zero-Latency Check SL / TP
                    initial_sl = self.position.get('initial_stop_loss', 0)
                    entry_price = self.position['entry_price']
                    
                    if not self.position.get('break_even_triggered', False):
                        if self.position['type'] == 'long' and initial_sl > 0:
                            if price - entry_price >= (entry_price - initial_sl) * self.break_even_ratio:
                                self.position['stop_loss'] = entry_price
                                self.position['break_even_triggered'] = True
                        elif self.position['type'] == 'short' and initial_sl > 0:
                            if entry_price - price >= (initial_sl - entry_price) * self.break_even_ratio:
                                self.position['stop_loss'] = entry_price
                                self.position['break_even_triggered'] = True
                                
                    if self.position['type'] == 'long' and price <= self.position['stop_loss']:
                        self._close_position(price, timestamp, "Stop Loss")
                    elif self.position['type'] == 'short' and price >= self.position['stop_loss']:
                        self._close_position(price, timestamp, "Stop Loss")
                        
                    tp = self.position.get('take_profit')
                    if tp:
                        if self.position['type'] == 'long' and price >= tp:
                            self._close_position(price, timestamp, "Take Profit")
                        elif self.position['type'] == 'short' and price <= tp:
                            self._close_position(price, timestamp, "Take Profit")
            except asyncio.CancelledError:
                break
"""
    if "async def _monitor_position(self):" not in content:
        content = content + "\n" + monitor_def

    # Remove _manage_exits calls for SL/TP as they are handled by monitor
    manage_exits_mod = """
    def _manage_exits(self, price, timestamp):
        if not self.position: return
        
        # SL/TP are now handled by _monitor_position task in zero-latency
        # Only Reversals are handled here
        
        if hasattr(self, 'last_module_results'):
"""
    content = re.sub(r'    def _manage_exits\(self, price, timestamp\):.*?if hasattr\(self, \'last_module_results\'\):', manage_exits_mod, content, flags=re.DOTALL)

    # Change __main__ to use asyncio.run
    content = content.replace("    eng.run()", "    asyncio.run(eng.run())")
    
    with open('backtest_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    refactor_llm_client()
    refactor_backtest_engine()
    print("Refactored to Async/Await with asyncio successfully.")
