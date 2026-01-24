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
    
    # Calcular timestamp de inicio
    since = exchange.milliseconds() - (limit_hours * 3600 * 1000)
    
    all_ohlcv = []
    current_since = since
    
    while len(all_ohlcv) < limit_hours:
        try:
            # Kraken devuelve hasta 720 velas por request
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            
            # Actualizar 'since' para la siguiente petición: timestamp de la última vela + 1ms
            last_timestamp = ohlcv[-1][0]
            current_since = last_timestamp + 1
            
            print(f"Descargando lote... Total descargado: {len(all_ohlcv)} velas")
            
            # Si devuelve menos de lo esperado, es probable que hayamos llegado al presente
            if len(ohlcv) < 1: 
                break
                
            time.sleep(1) # Respetar rate limit de API pública
            
        except Exception as e:
            print(f"Error descargando datos: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Limpiar duplicados y ordenar
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    # Recortar si descargamos de más
    if len(df) > limit_hours:
        df = df.iloc[-limit_hours:]
        
    df.to_csv(filepath, index=False)
    print(f"Datos guardados en {filepath}. Total filas: {len(df)}")
    
    return df

if __name__ == "__main__":
    download_history()
