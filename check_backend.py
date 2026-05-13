import socket
import sys
import time
import requests
import json
import threading
from urllib.parse import urlparse

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def check_rest_api(url):
    try:
        response = requests.get(f"{url}/engine/status", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

print("🔍 Iniciando Diagnóstico Quirúrgico del Backend 'OptimaTrading V2'")
print("-" * 60)

# 1. Verificar si el puerto local 8000 está en escucha
print("[1] Verificando binding local (0.0.0.0:8000)...")
if check_port("0.0.0.0", 8000) or check_port("127.0.0.1", 8000) or check_port("localhost", 8000):
    print("✅ El proceso FastAPI está escuchando localmente en el puerto 8000.")
else:
    print("❌ ALERTA: No se detectó ningún proceso escuchando en el puerto 8000 local.")
    print("   -> Revisa si uvicorn está corriendo correctamente.")

# 2. Verificar el endpoint REST interno
print("\n[2] Verificando endpoints REST internos...")
is_up, data = check_rest_api("http://127.0.0.1:8000")
if is_up:
    print("✅ REST API responde correctamente (Status 200).")
    print(f"   -> Modo actual: {data.get('mode')}")
else:
    print("❌ ALERTA: La API REST interna no responde o devolvió error.")
    print(f"   -> Detalle: {data}")

# 3. Verificar el WebSocket (Requiere instalar websocket-client: pip install websocket-client)
print("\n[3] Verificando endpoint WebSocket (/ws/telemetry)...")
try:
    import websocket
    def on_message(ws, message):
        print(f"   -> Recibido del WS: {message}")
        ws.close()
    
    def on_error(ws, error):
        print(f"❌ Error WS: {error}")
        
    def on_close(ws, close_status_code, close_msg):
        pass
        
    def on_open(ws):
        print("✅ Conexión WebSocket establecida exitosamente.")
        ws.send(json.dumps({"type": "ping", "timestamp": int(time.time() * 1000)}))

    ws = websocket.WebSocketApp("ws://127.0.0.1:8000/ws/telemetry",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    
    # Run the WS connection in a thread that times out
    wst = threading.Thread(target=ws.run_forever)
    wst.daemon = True
    wst.start()
    wst.join(timeout=3)
    
except ImportError:
    print("⚠️  Módulo 'websocket-client' no instalado. Omitiendo prueba profunda de WS.")
    print("   -> Ejecuta 'pip install websocket-client' si quieres probar el WebSocket internamente.")

print("-" * 60)
print("🏁 Diagnóstico completado.")
print("Si las pruebas locales (1 y 2) pasan pero Vercel no conecta, el problema está en:")
print("a) El mapeo de puertos de Easypanel (Asegúrate que mapee el 8000 del contenedor).")
print("b) El firewall de Hostinger bloqueando conexiones.")
