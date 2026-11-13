"""
Prueba el contrato de datos regla por regla y el fallback cuando el modelo responde mal DOS veces.
NO llama a ningún modelo real (no gasta dinero ni necesita internet): la respuesta inválida se simula.
Uso:  python src/probar_contrato.py
"""
import json                                  # para imprimir resultados prolijos
import os                                    # para poner una clave de mentira (no se usa de verdad)
import sys                                   # para terminar con código de error si alguna prueba falla

os.environ.setdefault("GROQ_API_KEY", "clave-de-prueba")        # el motor exige que exista una clave; acá es falsa
os.environ.setdefault("ANTHROPIC_API_KEY", "clave-de-prueba")

import motor                                 # nuestro motor (src/motor.py)
from pydantic import ValidationError         # error que lanza Pydantic cuando algo no cumple el contrato

# ---- Datos válidos de base: a cada prueba le rompemos UNA sola regla ----
ENTRADA_OK = {"id_empleado": "EMP-101", "area": "Sec. Servicios Generales", "texto_ticket": "Librería El Papel total 18500"}
SALIDA_OK = {"es_comprobante": True, "comercio": "Librería El Papel S.A.", "fecha": "2026-10-14", "monto_total": 18500.0,
             "categoria": "papeleria", "decision": "aprobar", "confianza": 0.97,
             "justificacion": "Factura válida de papelería dentro del tope."}


def probar_entrada(cambios: dict):
    """Intenta crear una Entrada con 'cambios'. Devuelve la lista de campos que fallaron (vacía si es válida)."""
    try:
        motor.Entrada(**{**ENTRADA_OK, **cambios})   # mezcla la entrada válida con el cambio que rompe la regla
        return []
    except ValidationError as e:
        return [x["campo"] for x in motor.resumir_errores(e)]


def probar_salida(cambios: dict):
    """Igual que arriba pero con Salida, a partir de un JSON (como llega del modelo)."""
    try:
        motor.Salida.model_validate_json(json.dumps({**SALIDA_OK, **cambios}))
        return []
    except ValidationError as e:
        return [x["campo"] for x in motor.resumir_errores(e)]


# (descripción, función, cambios, campo que DEBE fallar o None si debe ser válido)
CASOS = [
    ("Entrada válida (EMP-101)", probar_entrada, {}, None),
    ("PATRÓN id_empleado sin prefijo ('101')", probar_entrada, {"id_empleado": "101"}, "id_empleado"),
    ("PATRÓN id_empleado con pocos dígitos ('EMP-1')", probar_entrada, {"id_empleado": "EMP-1"}, "id_empleado"),
    ("PATRÓN id_empleado en minúscula ('emp-101')", probar_entrada, {"id_empleado": "emp-101"}, "id_empleado"),
    ("ENUM rol fuera de la lista ('admin')", probar_entrada, {"rol": "admin"}, "rol"),
    ("VALOR PERMITIDO area inexistente", probar_entrada, {"area": "Área Inventada"}, "area"),
    ("LARGO MÍNIMO texto_ticket ('hola')", probar_entrada, {"texto_ticket": "hola"}, "texto_ticket"),
    ("LARGO MÁXIMO texto_ticket (3001 caracteres)", probar_entrada, {"texto_ticket": "x" * 3001}, "texto_ticket"),
    ("Salida válida", probar_salida, {}, None),
    ("ENUM categoria fuera de la lista ('viajes')", probar_salida, {"categoria": "viajes"}, "categoria"),
    ("ENUM decision fuera de la lista ('aprobado')", probar_salida, {"decision": "aprobado"}, "decision"),
    ("RANGO confianza mayor a 1 (1.4)", probar_salida, {"confianza": 1.4}, "confianza"),
    ("RANGO confianza negativa (-0.1)", probar_salida, {"confianza": -0.1}, "confianza"),
    ("RANGO monto_total negativo (-5)", probar_salida, {"monto_total": -5}, "monto_total"),
    ("RANGO monto_total cero (0)", probar_salida, {"monto_total": 0}, "monto_total"),
    ("RANGO monto_total sobre el máximo (20.000.000)", probar_salida, {"monto_total": 20_000_000}, "monto_total"),
    ("FECHA inexistente ('2026-13-45')", probar_salida, {"fecha": "2026-13-45"}, "fecha"),
    ("LARGO MÍNIMO comercio ('A')", probar_salida, {"comercio": "A"}, "comercio"),
    ("LARGO MÍNIMO justificacion ('x')", probar_salida, {"justificacion": "x"}, "justificacion"),
    ("ENTRE CAMPOS aprobar sin fecha ni monto", probar_salida, {"fecha": None, "monto_total": None}, "(respuesta completa)"),
    ("ENTRE CAMPOS revision_humana sin fecha ni monto (permitido)", probar_salida,
     {"fecha": None, "monto_total": None, "decision": "revision_humana"}, None),
]

fallos = 0
print("=== 1) PRUEBA DEL CONTRATO: una regla rota por vez ===")
for descripcion, funcion, cambios, esperado in CASOS:
    campos = funcion(cambios)                                   # campos que fallaron
    ok = (campos == []) if esperado is None else (esperado in campos)   # ¿falló justo donde debía?
    fallos += 0 if ok else 1
    print(f"[{'OK ' if ok else 'MAL'}] {descripcion:<62} -> campos con error: {campos if campos else 'ninguno'}")

# ---- Falla persistente: el modelo contesta mal en los 2 intentos ----
print("\n=== 2) FALLA PERSISTENTE (respuestas del modelo SIMULADAS, sin llamar a la API) ===")
RESPUESTA_MALA = ('{"es_comprobante": true, "comercio": null, "fecha": "2026-13-45", "monto_total": -5, '
                  '"categoria": "viajes", "decision": "aprobar", "confianza": 1.4, "justificacion": "x"}')
motor.llamar_modelo = lambda cliente, sistema, usuario: (RESPUESTA_MALA, 900, 120, "stop")  # reemplaza la llamada real
resultado = motor.procesar_consulta(ENTRADA_OK)                 # el motor reintenta 1 vez y luego se rinde
print(json.dumps(resultado, indent=2, ensure_ascii=False))
esperado_ok = (resultado["ok"] is False and resultado["error"]["codigo"] == "SALIDA_INVALIDA"
               and resultado["metricas"]["intentos"] == 2 and len(resultado["error"]["campos"]) >= 5)
fallos += 0 if esperado_ok else 1
print(f"[{'OK ' if esperado_ok else 'MAL'}] ok=false, código SALIDA_INVALIDA, 2 intentos y campos afectados informados")

print(f"\nRESUMEN: {len(CASOS) + 1 - fallos}/{len(CASOS) + 1} pruebas correctas")
sys.exit(1 if fallos else 0)    # si alguna falló, la terminal recibe código de error
