import ccxt
import pandas as pd
import os
import time

def download_history(symbol='BTC/USD', timeframe='1h', days=365):
    """
    Descarga datos históricos de Kraken (u otro exchange) y los guarda en data/btc_history.csv.
    """
    print(f"--- Iniciando descarga de datos para {symbol} ({timeframe}) - {days} dias ---")
    
    # Asegurar que el directorio data/ existe
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    filepath = os.path.join(data_dir, 'btc_history.csv')
    
    exchange = ccxt.kraken()
    limit_hours = days * 24
    
    since = exchange.milliseconds() - (limit_hours * 3600 * 1000)
    
    all_ohlcv = []
    current_since = since
    
    while len(all_ohlcv) < limit_hours:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            current_since = ohlcv[-1][0] + 1
            
            print(f"Descargados {len(ohlcv)} velas. Total: {len(all_ohlcv)}")
            
            if len(ohlcv) < 720: 
                break
                
            time.sleep(1) 
            
        except Exception as e:
            print(f"Error descargando datos: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Limpiar
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    df.to_csv(filepath, index=False)
    print(f"Datos guardados en {filepath}. Total filas: {len(df)}")
    
    return df

if __name__ == "__main__":
    download_history()
