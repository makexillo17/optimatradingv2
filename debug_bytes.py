"""
debug_bytes.py - Inspeccion binaria del archivo .env
Detecta BOM, caracteres invisibles, y problemas de codificacion.
"""
import os

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

print(f"=== Inspeccion Binaria de: {env_path} ===\n")

with open(env_path, 'rb') as f:
    data = f.read()

print(f"Bytes totales: {len(data)}")

# Check BOM
if data[:3] == b'\xef\xbb\xbf':
    print("[WARN] BOM UTF-8 detectado (EF BB BF)")
elif data[:2] in (b'\xff\xfe', b'\xfe\xff'):
    print("[WARN] BOM UTF-16 detectado")
else:
    print("[OK] Sin BOM")

# Check for problematic characters
problematic = []
for i, byte in enumerate(data):
    if byte == 0x0D:
        problematic.append((i, '\\r (CR)', hex(byte)))
    elif byte == 0x00:
        problematic.append((i, 'NULL byte', hex(byte)))
    elif byte == 0xC2:
        if i + 1 < len(data) and data[i+1] == 0xA0:
            problematic.append((i, 'Non-breaking space (NBSP)', 'C2 A0'))
    elif byte > 0x7E and byte not in (0x0A,):
        problematic.append((i, f'Non-ASCII byte', hex(byte)))

if problematic:
    print(f"\n[WARN] Caracteres problematicos encontrados ({len(problematic)}):")
    for offset, desc, hexval in problematic[:20]:
        print(f"  Offset {offset}: {desc} ({hexval})")
else:
    print("[OK] Sin caracteres problematicos")

# Hex dump
print("\n=== Hex Dump Completo ===")
for i in range(0, len(data), 16):
    chunk = data[i:i+16]
    hex_part = ' '.join(f'{b:02x}' for b in chunk)
    ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
    print(f"  {i:04d}: {hex_part:<48s}  |{ascii_part}|")

# Show each line
print("\n=== Lineas del archivo ===")
lines = data.decode('utf-8', errors='replace').splitlines(keepends=True)
for idx, line in enumerate(lines, 1):
    print(f"  Linea {idx}: {repr(line)}")

# Check ANTHROPIC_API_KEY value
print("\n=== Valor de ANTHROPIC_API_KEY ===")
for line in lines:
    stripped = line.strip()
    if stripped.startswith('ANTHROPIC_API_KEY'):
        parts = stripped.split('=', 1)
        if len(parts) == 2:
            value = parts[1]
            print(f"  Valor encontrado: '{value}'")
            print(f"  Longitud: {len(value)}")
            if len(value) >= 4:
                print(f"  Ultimos 4 chars: '{value[-4:]}'")
            elif len(value) == 0:
                print("  [FAIL] VACIO - La llave no tiene valor asignado")
            else:
                print(f"  Valor completo: '{value}'")
        break
