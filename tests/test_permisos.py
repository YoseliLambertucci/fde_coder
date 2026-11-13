"""
Tests de permisos por rol del RAG. NO llaman a ningún modelo real ni descargan nada
(usan un embedder de prueba y un "modelo" simulado que además GRABA el prompt que recibe).
Uso:  python tests/test_permisos.py      (también funciona con pytest)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["EMBEDDER"] = "hash"                                   # embedder de prueba: rápido y sin internet
os.environ.setdefault("GROQ_API_KEY", "clave-de-prueba")          # clave falsa: el modelo está simulado
os.environ.setdefault("ANTHROPIC_API_KEY", "clave-de-prueba")
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))                             # para poder importar rag.py y motor.py

import rag                                                        # módulo RAG
from motor import Entrada                                         # contrato de entrada
from pydantic import ValidationError

# Documentos chicos para los tests (los 2 restringidos tienen una marca única, "canario")
DOCS_TEST = ["decreto_77-2026.txt", "guia_rendicion_empleados.md",
             "restringido_saldos_globales_2026.md", "restringido_criterios_auditoria.md"]
RESTRINGIDOS = ["restringido_saldos_globales_2026.md", "restringido_criterios_auditoria.md"]
CANARIOS = ["CANARIO-SALDOS-8820", "CANARIO-AUDITORIA-4471"]
# Frases que solo existen en los documentos restringidos
FRASES_RESTRINGIDAS = ["87.450", "fraccionamiento", "Kiosco El Atajo", "Contador General", "TOTAL GLOBAL"]
CONSULTAS_SENSIBLES = ["saldo disponible Servicios Generales", "alerta de fraccionamiento comprobantes mismo comercio",
                       "CANARIO-SALDOS-8820", "CANARIO-AUDITORIA-4471", "Kiosco El Atajo", "quién aprueba excepciones"]

_db = tempfile.mkdtemp()                                          # base temporal solo para los tests
_coleccion = rag.ingestar(tamano=500, solape=50, ruta_db=_db, solo=DOCS_TEST, verbose=False)


def usuario(rol):
    return rag.Usuario(id_empleado="EMP-101", rol=rol)


class ModeloSimulado:
    """Reemplaza al modelo real: guarda el prompt que recibe y responde citando F1."""
    def __init__(self):
        self.prompts = []
        self.fragmentos = []

    def __call__(self, cliente, sistema, usuario_msg):
        self.prompts.append(sistema + "\n" + usuario_msg)               # prompt completo
        self.fragmentos.append(usuario_msg.split("Fragmentos recuperados:", 1)[-1])  # solo la parte de fragmentos
        respuesta = {"respuesta": "Respuesta de prueba basada en F1.", "fragmentos_citados": ["F1"],
                     "evidencia_suficiente": True, "confianza": 0.95}
        return json.dumps(respuesta), 800, 60, "stop"


def entrada(rol, pregunta):
    return Entrada(id_empleado="EMP-101", area="Sec. Servicios Generales", rol=rol, texto_ticket=pregunta)


# ---------------------------- ROL EMPLEADO (sin permiso) ----------------------------
def test_empleado_no_recibe_documentos_restringidos_en_ningun_modo():
    for modo in ("hibrida", "vectorial", "bm25"):
        for consulta in CONSULTAS_SENSIBLES:
            for r in rag.buscar(consulta, usuario("empleado"), k=10, modo=modo, coleccion=_coleccion):
                assert r["documento"] not in RESTRINGIDOS, f"{modo}/{consulta}: apareció {r['documento']}"
                assert not any(c in r["texto"] for c in CANARIOS), "apareció un canario en el texto"


def test_empleado_no_manda_nada_restringido_al_modelo():
    """El prompt completo (sistema + usuario) enviado al modelo NO contiene texto ni nombres restringidos."""
    simulado = ModeloSimulado()
    rag.llamar_modelo = simulado                                  # el RAG usará el modelo simulado
    for consulta in CONSULTAS_SENSIBLES:
        r = rag.responder(entrada("empleado", consulta), k=6, coleccion=_coleccion)
        assert r["ok"], r
        assert not any(f["documento"] in RESTRINGIDOS for f in r["salida"]["fuentes"])
    assert simulado.prompts, "el modelo simulado no recibió ningún prompt"
    # 1) Los FRAGMENTOS enviados no pueden contener nada restringido (la pregunta del usuario puede nombrar un tema,
    #    pero el texto de los documentos no).
    for bloque in simulado.fragmentos:
        for prohibido in CANARIOS + FRASES_RESTRINGIDAS + RESTRINGIDOS:
            assert prohibido not in bloque, f"los fragmentos enviados contienen '{prohibido}'"
    # 2) El prompt COMPLETO no puede contener datos o nombres de archivo restringidos que no estén en la pregunta.
    for prompt in simulado.prompts:
        for prohibido in RESTRINGIDOS + ["$ 1.362.650", "TOTAL GLOBAL", "Contador General"]:
            assert prohibido not in prompt, f"el prompt contiene '{prohibido}'"


def test_el_filtro_de_rol_va_dentro_de_la_consulta_a_la_base():
    """Espiamos a ChromaDB: query() y get() deben recibir 'where' con el rol, ANTES de devolver resultados."""
    llamadas = []
    original_query, original_get = _coleccion.query, _coleccion.get
    _coleccion.query = lambda *a, **kw: (llamadas.append(("query", kw.get("where"))), original_query(*a, **kw))[1]
    _coleccion.get = lambda *a, **kw: (llamadas.append(("get", kw.get("where"))), original_get(*a, **kw))[1]
    rag._cache_corpus.clear()                                     # forzamos que se vuelva a consultar la base
    try:
        rag.buscar("saldo disponible", usuario("empleado"), k=3, coleccion=_coleccion)
    finally:
        _coleccion.query, _coleccion.get = original_query, original_get
    tipos = {t for t, _ in llamadas}
    assert {"query", "get"} <= tipos, f"se esperaban query y get, hubo {llamadas}"
    assert all(w == {"rol_empleado": True} for _, w in llamadas), f"consultas sin filtro de rol: {llamadas}"


def test_rol_desconocido_falla_cerrado():
    try:
        rag.Usuario(id_empleado="EMP-101", rol="admin")
        assert False, "se aceptó un rol inexistente"
    except ValidationError:
        pass
    try:
        entrada("admin", "pregunta de prueba larga")
        assert False, "se aceptó un rol inexistente en Entrada"
    except ValidationError:
        pass


# ---------------------------- ROL TESORERÍA (con permiso) ----------------------------
def test_tesoreria_si_recibe_documentos_restringidos():
    """Control positivo: si Tesorería tampoco los viera, el test del empleado no probaría nada."""
    docs = set()
    for consulta in ("CANARIO-SALDOS-8820", "CANARIO-AUDITORIA-4471"):
        for r in rag.buscar(consulta, usuario("tesoreria"), k=5, modo="bm25", coleccion=_coleccion):
            docs.add(r["documento"])
    assert set(RESTRINGIDOS) <= docs, f"Tesorería solo vio {docs}"


def test_tesoreria_manda_el_texto_restringido_al_modelo_y_lo_cita():
    simulado = ModeloSimulado()
    rag.llamar_modelo = simulado
    r = rag.responder(entrada("tesoreria", "CANARIO-SALDOS-8820 saldo disponible Servicios Generales"), k=3, coleccion=_coleccion)
    assert r["ok"] and not r["salida"]["requiere_humano"], r
    assert r["salida"]["fuentes"][0]["documento"] in RESTRINGIDOS
    assert "CANARIO-SALDOS-8820" in simulado.prompts[0]


def test_tesoreria_tambien_ve_los_documentos_publicos():
    r = rag.buscar("rendición una vez al mes", usuario("tesoreria"), k=5, modo="bm25", coleccion=_coleccion)
    assert any(x["documento"] == "decreto_77-2026.txt" for x in r)


# ---------------------------- DENEGADO POR DEFECTO ----------------------------
def test_documento_sin_permisos_declarados_no_se_ingesta():
    carpeta = Path(tempfile.mkdtemp())
    shutil.copy(RAIZ / "docs" / "guia_rendicion_empleados.md", carpeta)
    shutil.copy(RAIZ / "docs" / "restringido_saldos_globales_2026.md", carpeta / "nuevo_sin_permisos.md")
    (carpeta / "permisos.json").write_text(json.dumps({"guia_rendicion_empleados.md": ["empleado"]}), encoding="utf-8")
    col = rag.ingestar(carpeta=carpeta, tamano=500, solape=50, ruta_db=tempfile.mkdtemp(), verbose=False)
    docs = {m["documento"] for m in col.get(include=["metadatas"])["metadatas"]}
    assert docs == {"guia_rendicion_empleados.md"}, docs


def test_cada_fragmento_tiene_metadatos_de_permisos():
    datos = _coleccion.get(include=["metadatas"])
    for m in datos["metadatas"]:
        assert {"documento", "seccion", "roles", "rol_empleado", "rol_tesoreria"} <= set(m)
        if m["documento"] in RESTRINGIDOS:
            assert m["rol_empleado"] is False and m["rol_tesoreria"] is True


# ---------------------------- EJECUTOR ----------------------------
if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    print("=== TESTS DE PERMISOS ===")
    for nombre, funcion in pruebas:
        try:
            funcion()
            print(f"[OK ] {nombre}")
        except AssertionError as e:
            fallos += 1
            print(f"[MAL] {nombre}: {e}")
    print(f"\nRESUMEN: {len(pruebas) - fallos}/{len(pruebas)} tests correctos")
    sys.exit(1 if fallos else 0)
