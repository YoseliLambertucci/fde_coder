"""
Mide en cuántas de las preguntas el documento correcto aparece entre los primeros k resultados,
con DOS configuraciones de fragmentado (A y B), y muestra cuál ganó.
Uso:  python src/medir_rag.py
"""
import json
from datetime import datetime

import rag

preguntas = json.loads((rag.RAIZ / "data" / "preguntas_prueba.json").read_text(encoding="utf-8"))
evaluables = [p for p in preguntas if p["documentos_esperados"]]       # las 2 "sin respuesta" no tienen documento correcto
KS = (1, 3, 5)
resultados = {}
db_existente = rag.cliente_chroma(rag.RUTA_DB)

for nombre, (tamano, solape) in rag.CONFIGS.items():
    try:                                                               # si la colección ya existe (y es de esta versión) la reutilizamos
        coleccion = db_existente.get_collection(rag.nombre_coleccion(tamano, solape), embedding_function=rag.crear_embedder())
        if (coleccion.metadata or {}).get("version_ingesta") != rag.VERSION_INGESTA:
            raise ValueError("colección desactualizada")
    except Exception:
        print(f"Ingestando configuración {nombre} ({tamano}/{solape})...")
        coleccion = rag.ingestar(tamano=tamano, solape=solape, verbose=False)
    cantidad = coleccion.count()
    filas = []
    for p in evaluables:
        rol = "tesoreria" if p["tipo"] == "restringida" else "empleado"   # las restringidas se miden con el rol que puede leerlas
        usuario = rag.Usuario(id_empleado="EMP-101", rol=rol)
        for modo in ("hibrida", "vectorial"):
            top = rag.buscar(p["pregunta"], usuario, k=max(KS), modo=modo, coleccion=coleccion)
            pos_doc = next((n for n, r in enumerate(top, 1) if r["documento"] in p["documentos_esperados"]), None)
            pos_frag = next((n for n, r in enumerate(top, 1) if rag.contiene(r["texto"], p["evidencia_clave"])), None)
            filas.append({"id": p["id"], "modo": modo, "pos_documento": pos_doc, "pos_fragmento": pos_frag})
    resultados[nombre] = {"tamano": tamano, "solape": solape, "fragmentos": cantidad, "filas": filas}

def aciertos(nombre, modo, k, campo):
    return sum(1 for f in resultados[nombre]["filas"] if f["modo"] == modo and f[campo] is not None and f[campo] <= k)

total = len(evaluables)
print(f"\n=== MEDICIÓN TOP-K ({total} preguntas con documento correcto; las 2 sin respuesta se evalúan en correr_preguntas.py) ===")
print("Documento correcto (criterio de la consigna) / fragmento con la evidencia (criterio más estricto)\n")
print(f"{'Config':<15}{'Fragm.':<8}{'Búsqueda':<11}" + "".join(f"doc@{k:<4}" for k in KS) + "".join(f"frag@{k:<3}" for k in KS))
for nombre, r in resultados.items():
    for modo in ("hibrida", "vectorial"):
        docs = "".join(f"{aciertos(nombre, modo, k, 'pos_documento')}/{total:<5}" for k in KS)
        frags = "".join(f"{aciertos(nombre, modo, k, 'pos_fragmento')}/{total:<5}" for k in KS)
        print(f"{nombre + ' (' + str(r['tamano']) + '/' + str(r['solape']) + ')':<15}{r['fragmentos']:<8}{modo:<11}{docs}{frags}")

# Ganadora: la que más aciertos tiene con búsqueda híbrida en k=3 (desempate: fragmento@3, luego doc@1)
def puntaje(nombre):
    return (aciertos(nombre, "hibrida", 3, "pos_documento"), aciertos(nombre, "hibrida", 3, "pos_fragmento"),
            aciertos(nombre, "hibrida", 1, "pos_documento"))
ganadora = max(resultados, key=puntaje)
empate = puntaje("A") == puntaje("B")
print(f"\nGANADORA (híbrida, k=3): configuración {ganadora}" + ("  [EMPATE: elegir por otro criterio, ver decisiones]" if empate else ""))

carpeta = rag.RAIZ / "logs"; carpeta.mkdir(exist_ok=True)
archivo = carpeta / f"medicion_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
archivo.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Registro crudo guardado en: {archivo.relative_to(rag.RAIZ)}")
