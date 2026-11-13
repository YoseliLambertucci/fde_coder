"""
Corre las 10 preguntas de prueba por el RAG completo (búsqueda + modelo) y verifica el comportamiento esperado.
Uso:  python src/correr_preguntas.py
"""
import json
from datetime import datetime

import rag
from motor import Entrada

preguntas = json.loads((rag.RAIZ / "data" / "preguntas_prueba.json").read_text(encoding="utf-8"))
coleccion = rag.abrir_coleccion()
corridas, registro = [], []

# Calentamiento: la primera búsqueda carga (y a veces descarga) el modelo de embeddings. Lo hacemos ANTES de medir
# para que esa demora no se cuente como latencia de la primera pregunta.
print("Calentando el modelo de embeddings (no se cuenta en las mediciones)...")
rag.buscar("calentamiento", rag.Usuario(id_empleado="EMP-101", rol="empleado"), k=1, coleccion=coleccion)


def textos_citados(resultado) -> list:
    """Devuelve el texto de los fragmentos que la respuesta cita como fuente."""
    ids = [f["fragmento_id"] for f in resultado["salida"]["fuentes"]]
    return coleccion.get(ids=ids, include=["documents"])["documents"] if ids else []

# Cada pregunta se corre con el rol que corresponde. Las restringidas se corren con AMBOS roles:
#   empleado  -> debe quedar bloqueada (requiere_humano = true y ninguna fuente restringida)
#   tesoreria -> debe responder citando el documento restringido
for p in preguntas:
    roles = ["empleado", "tesoreria"] if p["tipo"] == "restringida" else ["empleado"]
    for rol in roles:
        entrada = Entrada(id_empleado="EMP-101", area="Sec. Servicios Generales", rol=rol, texto_ticket=p["pregunta"])
        r = rag.responder(entrada, k=6, coleccion=coleccion)
        print(f"\n--- Pregunta {p['id']} [{p['tipo']}] rol={rol} ---\n{p['pregunta']}")
        print(json.dumps(r, indent=2, ensure_ascii=False))
        registro.append({"pregunta": p, "rol": rol, "resultado": r})

        if not r["ok"]:
            correcto = False
        else:
            s = r["salida"]
            docs_citados = {f["documento"] for f in s["fuentes"]}
            # Criterio ESTRICTO: la respuesta solo cuenta como correcta si algún fragmento citado contiene la evidencia clave.
            # (Antes solo se miraba el documento citado, y una respuesta equivocada podía pasar como buena.)
            respaldada = p["evidencia_clave"] is not None and any(rag.contiene(t, p["evidencia_clave"]) for t in textos_citados(r))
            if p["tipo"] == "normal":
                correcto = (not s["requiere_humano"]) and respaldada
            elif p["tipo"] == "sin_respuesta":
                correcto = s["requiere_humano"]
            elif rol == "empleado":     # restringida vista por un empleado
                correcto = s["requiere_humano"] and not any(d.startswith("restringido_") for d in docs_citados)
            else:                       # restringida vista por tesorería
                correcto = (not s["requiere_humano"]) and respaldada
        corridas.append((p, rol, r, correcto))

print("\n\n(¿Esperado? = comportamiento correcto Y, si respondió, la fuente citada contiene la evidencia de la pregunta)")
print("| # | Tipo | Rol | requiere_humano | Fuentes citadas | ¿Esperado? | Tokens (ent/sal) | Costo USD | Latencia ms |")
print("|---|---|---|---|---|---|---|---|---|")
for p, rol, r, ok in corridas:
    if r["ok"]:
        s, m = r["salida"], r["metricas"]
        fuentes = ", ".join(sorted({f"{f['documento']} ({f['seccion']})" for f in s["fuentes"]})) or "-"
        print(f"| {p['id']} | {p['tipo']} | {rol} | {s['requiere_humano']} | {fuentes} | {'Sí' if ok else 'NO'} | "
              f"{m['tokens_entrada']}/{m['tokens_salida']} | {m['costo_usd']:.6f} | {m['latencia_ms']} |")
    else:
        print(f"| {p['id']} | {p['tipo']} | {rol} | ERROR {r['error']['codigo']} | - | NO | - | - | - |")
print(f"\nRESUMEN: {sum(ok for *_, ok in corridas)}/{len(corridas)} corridas con el comportamiento esperado")

carpeta = rag.RAIZ / "logs"; carpeta.mkdir(exist_ok=True)
archivo = carpeta / f"preguntas_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
archivo.write_text(json.dumps(registro, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(f"Registro crudo guardado en: {archivo.relative_to(rag.RAIZ)}")
