import os

def integrate_anemona():
    with open('backtest_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Imports
    if "import grpc" not in content:
        content = content.replace("import asyncio", "import asyncio\nimport grpc\nimport protos.anemona_pb2 as anemona_pb2\nimport protos.anemona_pb2_grpc as anemona_pb2_grpc")

    # Init variables
    if "self.alpha_factor =" not in content:
        content = content.replace("self.last_prob_direccion = 0.5", "self.last_prob_direccion = 0.5\n        self.alpha_factor = 0.0\n        self.position_multiplier = 1.0\n        self.witching_warning = False")

    # Anemona Listener Task
    if "_listen_anemona" not in content:
        run_task = """        self.monitor_task = asyncio.create_task(self._monitor_position())
        self.anemona_task = asyncio.create_task(self._listen_anemona())"""
        content = content.replace("        self.monitor_task = asyncio.create_task(self._monitor_position())", run_task)
        
        cancel_task = """        if self.monitor_task:
            self.monitor_task.cancel()
        if hasattr(self, 'anemona_task'):
            self.anemona_task.cancel()"""
        content = content.replace("""        if self.monitor_task:
            self.monitor_task.cancel()""", cancel_task)
            
        listener_method = """
    async def _listen_anemona(self):
        try:
            async with grpc.aio.insecure_channel('localhost:50051') as channel:
                stub = anemona_pb2_grpc.AlphaSignalEngineStub(channel)
                request = anemona_pb2.SignalRequest(client_id="OptimaV2")
                async for update in stub.SubscribePositionMultipliers(request):
                    self.alpha_factor = update.raw_current_alpha_factor
                    self.position_multiplier = update.terminal_position_multiplier
                    self.witching_warning = update.severe_macro_rebalance_warning
        except Exception as e:
            pass # gRPC no disponible
"""
        content += listener_method
        
    # Bayesian logic modification
    if "base_prob += (self.alpha_factor * 0.25)" not in content:
        content = content.replace("self.last_prob_direccion = max(0.0, min(1.0, base_prob))", 
                                  "base_prob += (self.alpha_factor * 0.25)\n            self.last_prob_direccion = max(0.0, min(1.0, base_prob))")

    # Position sizing and SL modification
    if "capital * self.position_multiplier" not in content:
        content = content.replace("capital = self.balance", "capital = self.balance * self.position_multiplier")
        content = content.replace("equity = self.balance", "equity = self.balance * self.position_multiplier")
        
        # SL aggressive
        sl_agg = """
            if self.witching_warning:
                # Aumentar agresividad de Stop Loss 30%
                entry_sl = execution_price - ((execution_price - entry_sl) * 0.7) if order['direction'] == 'BUY' else execution_price + ((entry_sl - execution_price) * 0.7)
"""
        content = content.replace("entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_long']", 
                                  "entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_long']" + sl_agg)
        content = content.replace("entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_short']", 
                                  "entry_sl = order['sniper_sl'] if order['sniper_sl'] > 0 else order['stop_short']" + sl_agg)
        
    with open('backtest_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    integrate_anemona()
    print("Anemona gRPC integration applied.")
