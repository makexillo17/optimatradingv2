import sys
import os
import pandas as pd
import ccxt
import time
from datetime import datetime

# Añadir el directorio raíz al path para poder importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modulos.gap_sniper import GapSniperModule

def run_test():
    print("Iniciando prueba de Gap Sniper...")
    
    # 1. Obtener datos históricos de BTC/USDT (Binance)
    print("Descargando datos históricos de Binance...")
    exchange = ccxt.binance()
    timeframe = '1h'
    symbol = 'BTC/USDT'
    limit = 720  # 30 días * 24 horas = 720 velas
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return

    # Convertir a DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"Datos descargados: {len(df)} velas desde {df['timestamp'].iloc[0]} hasta {df['timestamp'].iloc[-1]}")
    
    # 2. Instanciar el módulo
    sniper = GapSniperModule()
    
    # 3. Iterar sobre el histórico
    print("\nEjecutando simulación paso a paso...")
    signal_count = 0
    bullish_count = 0
    bearish_count = 0
    
    # Necesitamos al menos 5 velas para empezar
    for i in range(5, len(df)):
        # Ventana deslizante: tomamos hasta el índice i (inclusive)
        # El módulo usa iloc[-1] como la vela "actual"
        # Pasamos un slice del DF
        current_data = df.iloc[:i+1].copy()
        
        # Preparar el input como lo espera el módulo
        input_data = {
            'market_data': current_data
        }
        
        # Ejecutar análisis
        result = sniper.analyze(input_data)
        
        # Verificar señal
        if result['recommendation'] != 'neutral':
            signal_count += 1
            timestamp = current_data['timestamp'].iloc[-1]
            price = current_data['close'].iloc[-1]
            signal_type = "GAP ALCISTA" if result['recommendation'] == 'long' else "GAP BAJISTA"
            
            if result['recommendation'] == 'long':
                bullish_count += 1
            else:
                bearish_count += 1
                
            print(f"[{timestamp}] 🎯 {signal_type} detectado a precio ${price:.2f}")
            print(f"   Justificación: {result['justification']}")
            
    # Resumen
    print("\n" + "="*50)
    print("RESUMEN DE RESULTADOS")
    print("="*50)
    print(f"Total de velas analizadas: {len(df) - 5}")
    print(f"Total de señales encontradas: {signal_count}")
    print(f"  - Bullish FVG: {bullish_count}")
    print(f"  - Bearish FVG: {bearish_count}")
    print("="*50)

if __name__ == "__main__":
    run_test()
