
def check_hard_veto(module_results, recommendation, is_toxic):
    if is_toxic:
        return True, "REJECTED_BY_MATH: Toxic Flow detectado."
        
    obi_score = module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0)
    
    if recommendation == 'long' and obi_score < -0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente negativo (< -0.4) para compras."
    if recommendation == 'short' and obi_score > 0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente positivo (> 0.4) para ventas."
        
    return False, ""
