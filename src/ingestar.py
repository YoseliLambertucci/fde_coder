"""
UN SOLO COMANDO procesa los documentos y arma la base ChromaDB.
Uso:   python src/ingestar.py              (usa la configuración por defecto)
       python src/ingestar.py --config B   (otra configuración de fragmentado)
"""
import argparse                          # para leer opciones de la línea de comandos
import json                              # para mostrar un ejemplo de metadatos

import rag                               # nuestro módulo RAG (src/rag.py)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=rag.CONFIG_DEFECTO, choices=list(rag.CONFIGS), help="A o B (ver rag.CONFIGS)")
args = parser.parse_args()
tamano, solape = rag.CONFIGS[args.config]

print(f"=== INGESTA: configuración {args.config} (fragmentos de {tamano} caracteres, solape {solape}) ===")
coleccion = rag.ingestar(tamano=tamano, solape=solape)

# Mostramos un ejemplo de fragmento con sus metadatos (evidencia para el PDF)
ejemplo = coleccion.get(where={"$and": [{"documento": "decreto_2980-2000.pdf"}, {"seccion": "Artículo 87"}]}, limit=1,
                        include=["documents", "metadatas"])
print("\n--- Ejemplo de fragmento ingestado ---")
print("id:       ", ejemplo["ids"][0])
print("texto:    ", ejemplo["documents"][0][:220].replace("\n", " "), "...")
print("metadatos:", json.dumps(ejemplo["metadatas"][0], ensure_ascii=False, indent=2))

# Resumen por rol (cuántos fragmentos puede leer cada uno)
for rol in ("empleado", "tesoreria"):
    cantidad = len(coleccion.get(where={f"rol_{rol}": True}, include=[])["ids"])
    print(f"Fragmentos legibles por el rol {rol:<9}: {cantidad}")
