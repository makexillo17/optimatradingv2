import re

def add_panic_protocol():
    with open('backtest_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Imports
    if "from modulos.execution_manager import execute_hard_veto" not in content:
        content = content.replace("from modulos.execution_manager import check_hard_veto",
                                  "from modulos.execution_manager import check_hard_veto, execute_hard_veto")

    # State vars
    if "self.event_impact =" not in content:
        content = content.replace("self.witching_warning = False", "self.witching_warning = False\n        self.event_impact = 1\n        self.event_sentiment = 0.0\n        self.system_status = 'ACTIVE'")

    # Update _listen_anemona
    listen_old = """                    self.alpha_factor = update.raw_current_alpha_factor
                    self.position_multiplier = update.terminal_position_multiplier
                    self.witching_warning = update.severe_macro_rebalance_warning"""
    listen_new = """                    self.alpha_factor = update.raw_current_alpha_factor
                    self.position_multiplier = update.terminal_position_multiplier
                    self.witching_warning = update.severe_macro_rebalance_warning
                    self.event_impact = update.event_impact_level
                    self.event_sentiment = update.event_sentiment"""
    content = content.replace(listen_old, listen_new)

    # Monitor Position - Panic check
    monitor_old = "                if self.position:"
    monitor_new = """                # PANIC PROTOCOL
                if self.system_status == 'ACTIVE' and getattr(self, 'event_impact', 1) == 5 and getattr(self, 'event_sentiment', 0.0) < -0.8:
                    execute_hard_veto("BTCUSD")
                    self.system_status = 'SUSPENDED'
                    if self.position:
                        self._close_position(self.current_price, self.current_time, "FLASH CLOSE (Panic Protocol)")
                        
                if self.system_status == 'SUSPENDED':
                    continue

                if self.position:"""
    content = content.replace(monitor_old, monitor_new)

    # _handle_signal SMC Logic
    # Old logic
    old_signal_check = "if vetoed:"
    new_signal_check = """
            # SMC Liquidity Catalyst
            smc_just = module_results.get('smc_ict', {}).get('justification', '')
            if getattr(self, 'event_impact', 1) == 4:
                if 'ORDER BLOCK' in smc_just.upper() or 'BREAKER' in smc_just.upper():
                    if (recommendation == 'long' and getattr(self, 'event_sentiment', 0.0) > 0.5) or \\
                       (recommendation == 'short' and getattr(self, 'event_sentiment', 0.0) < -0.5):
                        self.position_multiplier *= 1.2
                    elif (recommendation == 'long' and obi_score < -0.3) or \\
                         (recommendation == 'short' and obi_score > 0.3):
                        vetoed = True
                        veto_reason = "FAKE_OUT_RISK: Noticia alto impacto pero divergencia con OBI."
                        
            if vetoed:"""
    content = content.replace(old_signal_check, new_signal_check)

    # _handle_market_tick suspend check
    tick_old = "    async def _handle_market_tick(self, event):"
    tick_new = "    async def _handle_market_tick(self, event):\n        if getattr(self, 'system_status', 'ACTIVE') == 'SUSPENDED': return"
    content = content.replace(tick_old, tick_new)

    with open('backtest_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    add_panic_protocol()
    print("Panic Protocol and SMC Interface applied.")
