import ccxt
import pandas as pd
import os
import time

def fetch_data(symbol='BTC/USD', timeframe='1h', limit=365*24):
    """
    Descarga datos históricos de Kraken (u otro exchange) y los guarda en CSV.
    Limit 365*24 = 8760 horas (1 año aprox).
    """
    print(f"--- Iniciando descarga de datos para {symbol} ({timeframe}) ---")
    
    # Usar Kraken que suele tener buen historial público
    exchange = ccxt.kraken()
    
    # Calcular chunks si el límite es grande (Kraken devuelve 720 por request)
    # Pero para simplificar en esta versión, haremos una petición grande o loop simple
    
    since = exchange.milliseconds() - (limit * 3600 * 1000) # Aproximado para 1h
    
    all_ohlcv = []
    current_since = since
    
    # Fetch loop simples
    while len(all_ohlcv) < limit:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            
            print(f"Descargados {len(ohlcv)} velas. Total: {len(all_ohlcv)}")
            
            if len(ohlcv) < 720: # Menos del límite de la API, asumimos fin
                break
                
            time.sleep(1) # Rate limit
            
        except Exception as e:
            print(f"Error descargando datos: {e}")
            break
            
    # Convertir a DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Limpiar y ordenar
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Guardar a CSV
    filename = 'btc_history.csv'
    df.to_csv(filename, index=False)
    print(f"Datos guardados en {filename}. Total filas: {len(df)}")
    
    return df

if __name__ == "__main__":
    fetch_data()
