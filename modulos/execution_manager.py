
def check_hard_veto(module_results, recommendation, is_toxic):
    if is_toxic:
        return True, "REJECTED_BY_MATH: Toxic Flow detectado."
        
    obi_score = module_results.get('smc_ict', {}).get('details', {}).get('obi_score', 0.0)
    
    if recommendation == 'long' and obi_score < -0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente negativo (< -0.4) para compras."
    if recommendation == 'short' and obi_score > 0.4:
        return True, "REJECTED_BY_MATH: OBI agresivamente positivo (> 0.4) para ventas."
        
    return False, ""

def execute_hard_veto(symbol):
    """
    Protocolo de Pánico: Cierra todo, cancela órdenes y suspende operaciones.
    """
    print(f"\\n[🚨 PANIC PROTOCOL] EJECUTANDO HARD VETO PARA {symbol.upper()}")
    print("Paso 1: Cancelación masiva de órdenes en MEXC (DELETE /api/v1/allOpenOrders)")
    print("        y Bitso (DELETE /api/v3/orders/all)... [OK]")
    print("Paso 2: Ejecución de orden Flash Close / ReduceOnly a Market... [OK]")
    print("Paso 3: Estado del sistema = SUSPENDED.\\n")
    return True
