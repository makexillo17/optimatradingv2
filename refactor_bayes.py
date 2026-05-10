import os
import re

def create_execution_manager():
    content = """
def check_hard_veto(module_results, recommendation, is_toxic):
    if is_toxic:
        return True, "REJECTED_BY_MATH: Toxic Flow detectado."
        
    obi_score = module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0)
    
    if recommendation == 'long' and obi_score < -0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente negativo (< -0.4) para compras."
    if recommendation == 'short' and obi_score > 0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente positivo (> 0.4) para ventas."
        
    return False, ""
"""
    with open('modulos/execution_manager.py', 'w', encoding='utf-8') as f:
        f.write(content)

def update_llm_client():
    with open('llm_client.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace the prompt builder
    old_prompt_builder = """        # ── Construir el user prompt ────────────────────────────────
        user_prompt = self._build_user_prompt(current_data, context_docs, market_regime)"""
        
    new_prompt_builder = """        obi_score = current_data.get('obi_score', 0.0)
        rvol = current_data.get('rvol', 1.0)
        structure = current_data.get('structure', 'Estructura Neutra')
        
        user_prompt = f"Contexto: OBI muestra absorción del {abs(obi_score*100):.0f}%. WVDI confirma volumen institucional ({rvol:.2f}). El precio está en {structure}. Valida si la estructura de velas confirma la dirección del mercado."
        user_prompt += "\\nIMPORTANTE: Retorna ÚNICAMENTE un número decimal entre 0.0 y 1.0 representando tu Score Estético IA. No agregues texto."
"""
    content = content.replace(old_prompt_builder, new_prompt_builder)
    
    # Replace parsing logic
    old_parse = """            # Extraer texto de la respuesta
            raw_response = message.content[0].text
            decision = raw_response.strip().upper()

            # Validar que sea una decisión esperada
            if decision not in ("BUY", "SELL", "HOLD"):
                logger.warning(
                    "Respuesta inesperada de Claude: '%s' — defaulting to HOLD",
                    raw_response,
                )
                return "HOLD"

            logger.info("ClaudeTrader decisión: %s", decision)
            return decision"""
            
    new_parse = r"""            raw_response = message.content[0].text.strip()
            try:
                import re
                match = re.search(r"0\.\d+|1\.0|0|1", raw_response)
                score_estetico = float(match.group(0)) if match else 0.5
            except ValueError:
                score_estetico = 0.5
                
            return score_estetico"""
            
    content = content.replace(old_parse, new_parse)
    
    with open('llm_client.py', 'w', encoding='utf-8') as f:
        f.write(content)

def update_backtest_engine():
    with open('backtest_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()

    if "from modulos.execution_manager import check_hard_veto" not in content:
        content = content.replace("from modulos.market_regime import detect_regime",
                                  "from modulos.market_regime import detect_regime\nfrom modulos.execution_manager import check_hard_veto\nfrom modulos.microstructure import is_flow_toxic")

    # Add prob tracking in init
    if "self.last_prob_direccion =" not in content:
        content = content.replace("self.total_latency_loss = 0.0", "self.total_latency_loss = 0.0\n        self.last_prob_direccion = 0.5")

    # Change how consensus is handled
    # Remove the old ai_decision block and replace with the new Bayesian logic
    
    bayes_logic = """
            # BAYESIAN PROBABILITY LOGIC
            obi_score = float(module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0))
            rvol = float(module_results.get('broker_behavior', {}).get('metrics', {}).get('rvol', 1.0))
            structure = module_results.get('smc_ict', {}).get('structure', 'Neutral')
            
            base_prob = 0.5
            if recommendation == 'long':
                base_prob += (obi_score * 0.3)
                if rvol > 1.2: base_prob += 0.1
            elif recommendation == 'short':
                base_prob += (abs(obi_score) * 0.3)
                if rvol > 1.2: base_prob += 0.1
                
            self.last_prob_direccion = max(0.0, min(1.0, base_prob))
            
            is_toxic = is_flow_toxic(current_df, obi_score)
            vetoed, veto_reason = check_hard_veto(module_results, recommendation, is_toxic)
            
            if vetoed:
                ai_decision = "HOLD"
                print(f"[{current_time}] {veto_reason}")
            else:
                digest = {
                    'obi_score': obi_score,
                    'rvol': rvol,
                    'structure': structure
                }
                try: 
                    score_estetico_IA = await self.trader.analyze_market_data(digest, market_regime=current_regime)
                except Exception as e: 
                    print("Error Claude:", e)
                    score_estetico_IA = 0.5
                    
                prob_combinada = (self.last_prob_direccion + score_estetico_IA) / 2.0
                
                if prob_combinada > 0.75:
                    ai_decision = recommendation
                else:
                    ai_decision = "HOLD"
"""

    # We need to replace the call to self.trader.analyze_market_data in backtest_engine
    # Let's find it:
    old_claude_call = """            current_row = {
                'timestamp': str(current_time), 'open': float(current_candle['open']),
                'high': float(current_candle['high']), 'low': float(current_candle['low']),
                'close': float(current_price), 'volume': float(current_candle['volume']),
                'ema200': float(current_ema200), 'regime': current_regime,
                'consensus_signal': float(consensus_signal),
                'obi_score': float(module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0))
            }
            try: ai_decision = await self.trader.analyze_market_data(current_row, market_regime=current_regime)
            except Exception: ai_decision = "HOLD"
            
            if current_regime == 'NOISE' and ai_decision != "HOLD": ai_decision = "HOLD"
            if ai_decision == "BUY" and current_price < current_ema200: ai_decision = "HOLD"
            elif ai_decision == "SELL" and current_price > current_ema200: ai_decision = "HOLD" """

    content = content.replace(old_claude_call, bayes_logic)
    
    # In _monitor_position, add Exit on Weakness
    exit_on_weakness = """                    if not self.position.get('break_even_triggered', False):
                        # Exit on weakness
                        if self.last_prob_direccion < 0.4:
                            self._close_position(price, timestamp, "Exit on Weakness (Prob < 0.4)")
                            continue"""
                            
    content = content.replace("                    if not self.position.get('break_even_triggered', False):", exit_on_weakness)
    
    with open('backtest_engine.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    create_execution_manager()
    update_llm_client()
    update_backtest_engine()
    print("Bayesian Orchestration applied.")
