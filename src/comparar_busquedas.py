"""
Compara búsqueda SOLO VECTORIAL contra HÍBRIDA (vectorial + BM25) en consultas con términos exactos
(códigos, nombres, números de artículo). Sirve para encontrar al menos 2 casos donde la híbrida acierta y la vectorial falla.
Uso:  python src/comparar_busquedas.py
"""
import json
from datetime import datetime

import rag

K = 3                                                                     # miramos los primeros 3 resultados
usuario = rag.Usuario(id_empleado="EMP-101", rol="empleado")
coleccion = rag.abrir_coleccion()
consultas = json.loads((rag.RAIZ / "data" / "consultas_exactas.json").read_text(encoding="utf-8"))

print(f"=== VECTORIAL vs HÍBRIDA (top-{K}, rol empleado, fragmentado {rag.CONFIG_DEFECTO}) ===")
registro, casos_clave = [], 0
for c in consultas:
    fila = {"consulta": c["consulta"], "debe_contener": c["debe_contener"]}
    for modo in ("vectorial", "hibrida"):
        resultados = rag.buscar(c["consulta"], usuario, k=K, modo=modo, coleccion=coleccion)
        posicion = next((n for n, r in enumerate(resultados, 1) if rag.contiene(r["texto"], c["debe_contener"])), None)
        fila[modo] = {"acierta": posicion is not None, "posicion": posicion,
                      "top": [f"{r['documento']} | {r['seccion']} | pág {r['pagina']}" for r in resultados]}
    fila["caso_clave"] = fila["hibrida"]["acierta"] and not fila["vectorial"]["acierta"]
    casos_clave += fila["caso_clave"]
    registro.append(fila)

    marca = "  <== HÍBRIDA ACIERTA, VECTORIAL FALLA" if fila["caso_clave"] else ""
    print(f"\nConsulta: {c['consulta']}   (el fragmento correcto contiene: '{c['debe_contener']}'){marca}")
    for modo in ("vectorial", "hibrida"):
        r = fila[modo]
        print(f"  {modo:<9}: {'ACIERTA (posición ' + str(r['posicion']) + ')' if r['acierta'] else 'FALLA'}")
        for linea in r["top"]:
            print(f"             - {linea}")

print(f"\nRESUMEN: híbrida acierta y vectorial falla en {casos_clave} de {len(consultas)} consultas "
      f"| vectorial acierta {sum(f['vectorial']['acierta'] for f in registro)}/{len(consultas)} "
      f"| híbrida acierta {sum(f['hibrida']['acierta'] for f in registro)}/{len(consultas)}")

carpeta = rag.RAIZ / "logs"; carpeta.mkdir(exist_ok=True)
archivo = carpeta / f"comparacion_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json"
archivo.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Registro crudo guardado en: {archivo.relative_to(rag.RAIZ)}")
