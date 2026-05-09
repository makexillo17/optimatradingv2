import re

def modify_smc():
    with open('modulos/smc_ict.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Imports
    if "from modulos.microstructure import calculate_obi, is_flow_toxic" not in content:
        content = content.replace("import pandas as pd\nimport numpy as np", 
                                  "import pandas as pd\nimport numpy as np\nfrom modulos.microstructure import calculate_obi, is_flow_toxic")

    # 2. Analyze method
    # Find start of analyze
    analyze_sig = "def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:"
    if "obi_score = calculate_obi(df)" not in content:
        content = content.replace(
            "current_close = df.iloc[-1]['close']",
            "current_close = df.iloc[-1]['close']\n        obi_score = calculate_obi(df)"
        )
        
    # 3. Spoofing logic during mitigation
    bull_mitigation = "if (z['bottom'] * 0.999) <= current_close <= (z['top'] * 1.001):"
    if "spoofing" not in content:
        bull_spoof = """
                    if obi_score < -0.1:
                        # Spoofing: Retiro de Bids institucionales
                        z['state'] = OB_INVALIDATED
                        continue
                    if (z['bottom'] * 0.999) <= current_close <= (z['top'] * 1.001):"""
        content = content.replace(bull_mitigation, bull_spoof, 1)
        
    bear_mitigation = "if (z['bottom'] * 0.999) <= current_close <= z['top']:"
    if "Spoofing: Retiro de Asks" not in content:
        bear_spoof = """
                    if obi_score > 0.1:
                        # Spoofing: Retiro de Asks institucionales
                        z['state'] = OB_INVALIDATED
                        continue
                    if (z['bottom'] * 0.999) <= current_close <= z['top']:"""
        content = content.replace(bear_mitigation, bear_spoof, 1)

    # 4. Toxic Flow and details
    if "Toxic Flow" not in content:
        ret_block = "return {\n            'recommendation': signal,"
        new_ret_block = """
        if signal == 'long' and is_flow_toxic(df, obi_score):
            signal = 'neutral'
            justification = "Bloqueado por Toxic Flow (Precio sube pero OBI negativo)."
        elif signal == 'short' and is_flow_toxic(df, obi_score):
            signal = 'neutral'
            justification = "Bloqueado por Toxic Flow (Precio baja pero OBI positivo)."
            
        return {
            'recommendation': signal,"""
        content = content.replace(ret_block, new_ret_block)
        
        # Add obi_score to details
        content = content.replace(
            "'structure': structure,",
            "'structure': structure,\n                'obi_score': obi_score,"
        )
        
    # 5. _detect_order_blocks Absorption
    if "obi_score_ob" not in content:
        bull_ob_creation = "if close_price > open_price and prev_close < prev_open:"
        new_bull_ob = """obi_score_ob = calculate_obi(df.iloc[:i+1])
            if close_price > open_price and prev_close < prev_open and obi_score_ob > 0.3:"""
        content = content.replace(bull_ob_creation, new_bull_ob)
        
        bear_ob_creation = "if close_price < open_price and prev_close > prev_open:"
        new_bear_ob = """obi_score_ob = calculate_obi(df.iloc[:i+1])
            if close_price < open_price and prev_close > prev_open and obi_score_ob < -0.3:"""
        content = content.replace(bear_ob_creation, new_bear_ob)

    with open('modulos/smc_ict.py', 'w', encoding='utf-8') as f:
        f.write(content)


def modify_gap_sniper():
    with open('modulos/gap_sniper.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "from modulos.microstructure import calculate_obi, is_flow_toxic" not in content:
        content = content.replace("import pandas as pd\nimport numpy as np", 
                                  "import pandas as pd\nimport numpy as np\nfrom modulos.microstructure import calculate_obi, is_flow_toxic")

    if "obi_score = calculate_obi(df)" not in content:
        content = content.replace(
            "current_close = current_candle['close']",
            "current_close = current_candle['close']\n        obi_score = calculate_obi(df)"
        )
        
    if "Toxic Flow" not in content:
        ret_block = "return {\n            'recommendation': signal,"
        new_ret_block = """
        if signal == 'long' and is_flow_toxic(df, obi_score):
            signal = 'neutral'
            justification = "Bloqueado por Toxic Flow (Gap Alcista pero OBI negativo)."
        elif signal == 'short' and is_flow_toxic(df, obi_score):
            signal = 'neutral'
            justification = "Bloqueado por Toxic Flow (Gap Bajista pero OBI positivo)."
            
        return {
            'recommendation': signal,"""
        content = content.replace(ret_block, new_ret_block)
        
        # Add obi_score to details
        content = content.replace(
            "'gap_size_percent': gap_pct",
            "'gap_size_percent': gap_pct,\n                'obi_score': obi_score"
        )
        
    with open('modulos/gap_sniper.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    modify_smc()
    modify_gap_sniper()
    print("Modifications applied successfully.")
