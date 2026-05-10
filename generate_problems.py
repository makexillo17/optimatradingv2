import json
import os
from collections import Counter

f = open('sample_errors.json', encoding='utf-8')
d = json.load(f)

c = Counter([diag.get('rule', 'General Syntax/Type Error') for diag in d])

lines = ['# Lista de Problemas Detectados\n']
lines.append('Se han categorizado los problemas de análisis estático encontrados en el código base (totalizando ~370 problemas, de los cuales los 181 principales coinciden con las advertencias de tipado estrictas).\n')
lines.append('## Resumen por Tipo de Error\n')
for rule, count in c.most_common():
    lines.append(f'- **{rule}**: {count} problemas')

lines.append('\n## Archivos más afectados\n')
file_c = Counter([os.path.basename(diag['file']) for diag in d])
for f_name, count in file_c.most_common(20):
    lines.append(f'- `{f_name}`: {count} problemas')

lines.append('\n## Solución Íntegra Propuesta\n')
lines.append('''
Para solucionar la totalidad de los problemas detectados de manera sistémica y **sin afectar la lógica del programa**, implementaremos las siguientes estrategias:

1. **Corrección de Proto gRPC (`anemona_pb2.py` / `anemona_pb2_grpc.py`)**:
   - Generar nuevamente los protos con el flag `--pyi_out` para que Python detecte correctamente los tipos como `MultiplierUpdate` y `SignalRequest`.
   - Alternativamente, agregar declaraciones de tipado estáticas `type: ignore` si los imports dinámicos de Protobuf están confundiendo a Pylance.

2. **Resolución de Variables No Ligadas (`reportPossiblyUnboundVariable`, `reportUnboundVariable`)**:
   - Inicializar variables como `consensus_result` y `entry_fee` con `None` o valores por defecto al inicio de las funciones, antes de bloques condicionales o `try/except`.

3. **Acceso Seguro a Miembros Opcionales (`reportOptionalMemberAccess`)**:
   - Variables que pueden ser `None` (como el retorno de operaciones o diccionarios) serán verificadas explícitamente (`if var is not None: var.upper()`).

4. **Incompatibilidad de Tipos en Argumentos (`reportArgumentType`)**:
   - Utilizar el casting correcto usando `typing.cast` o adecuar los parámetros. Por ejemplo, en pandas/matplotlib, asegurar que los índices son listas limpias o arrays en lugar de tipos opcionales.

5. **Clases y Atributos Desconocidos (`reportAttributeAccessIssue`)**:
   - Limpiar `TextIO` (reconfigure no existe en todos los streams) y validar los tipos base.
''')

open('problems_list.md', 'w', encoding='utf-8').write('\n'.join(lines))
