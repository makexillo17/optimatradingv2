import numpy as np

def calculate_obi(df):
    """
    Simula el Order Book Imbalance (OBI) de los primeros 10 niveles (L2) 
    usando la microestructura de la vela actual y reciente (OHLCV).
    Formula real: OBI = (Sum Bids_10 - Sum Asks_10) / (Sum Bids_10 + Sum Asks_10)
    Regla de Oro: OBI > 0.3 = Presion de compra, OBI < -0.3 = Presion de venta.
    """
    if len(df) < 5:
        return 0.0
        
    current = df.iloc[-1]
    
    high_low_range = current['high'] - current['low']
    if high_low_range == 0:
        return 0.0
        
    buy_pressure = (current['close'] - current['low']) / high_low_range
    sell_pressure = (current['high'] - current['close']) / high_low_range
    
    vol = current['volume']
    
    simulated_bids = vol * buy_pressure * 10 
    simulated_asks = vol * sell_pressure * 10
    
    if simulated_bids + simulated_asks == 0:
        return 0.0
        
    obi = (simulated_bids - simulated_asks) / (simulated_bids + simulated_asks)
    return float(obi)

def is_flow_toxic(df, obi_score):
    """
    Verifica si el flujo es toxico comparando el precio con el OBI.
    Logica: Si el precio sube pero el OBI es fuertemente negativo (<-0.3),
    es 'Toxic Flow' (Trampa para minoristas). Retorna True para bloquear BUY.
    (Se implementa solo para el lado de compras o simetrico segun necesidad).
    """
    if len(df) < 2: return False
    
    price_change = df.iloc[-1]['close'] - df.iloc[-2]['close']
    
    # Precio sube, pero OBI muestra venta institucional agresiva
    if price_change > 0 and obi_score < -0.3:
        return True
        
    # Precio baja, pero OBI muestra compra institucional agresiva
    if price_change < 0 and obi_score > 0.3:
        return True
        
    return False
