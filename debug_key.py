"""
debug_key.py — Prueba de aislamiento para ANTHROPIC_API_KEY
"""
import os
from dotenv import load_dotenv

# Ruta absoluta al .env
env_path = r"c:\Users\chump\OneDrive\proyecto personal\.env"

print(f"[debug_key] Cargando: {env_path}")
print(f"[debug_key] Existe: {os.path.exists(env_path)}")

result = load_dotenv(dotenv_path=env_path, override=True)
print(f"[debug_key] load_dotenv retorno: {result}")

key = os.environ.get("ANTHROPIC_API_KEY", "")
print(f"[debug_key] Longitud de la llave: {len(key)}")

if len(key) >= 4:
    print(f"[debug_key] Últimos 4 caracteres: {key[-4:]}")
elif len(key) == 0:
    print("[debug_key] ❌ La llave está VACÍA")
else:
    print(f"[debug_key] Llave demasiado corta: '{key}'")
