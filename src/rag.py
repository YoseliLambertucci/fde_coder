"""
RAG con permisos y búsqueda híbrida (PE3).

Tres piezas principales:
  ingestar()  -> lee los documentos, los limpia, los parte en fragmentos y los guarda en ChromaDB con metadatos.
  buscar()    -> busca fragmentos con búsqueda vectorial + BM25 (palabras clave), SIEMPRE filtrando por rol.
  responder() -> arma el prompt solo con lo recuperado, llama al modelo y devuelve respuesta + fuentes.
"""

# ---------------------------------------------------------------------------
# 1) IMPORTS
# ---------------------------------------------------------------------------
import hashlib                       # para el "embedder de prueba" (hash de palabras)
import json                          # para leer permisos.json
import logging                       # mensajes en la terminal
import os                            # variables de entorno
import re                            # expresiones regulares (buscar patrones en texto)
import time                          # medir tiempos
import unicodedata                   # quitar tildes para comparar palabras
from pathlib import Path             # rutas de archivos
from typing import Optional

import chromadb                      # base de datos vectorial
from chromadb.config import Settings
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader          # lee PDFs
from rank_bm25 import BM25Okapi      # búsqueda por palabras clave (BM25)

# Reutilizamos piezas del motor de la PE2 (contrato de entrada, llamada al modelo, medición, errores).
from motor import (CONFIG, RAIZ, Entrada, ErrorProveedor, Rol, armar_metricas, cargar_plantilla, crear_cliente,
                   error_estructurado, limpiar_json, llamar_modelo, resumir_errores)

log = logging.getLogger("rag")

# ---------------------------------------------------------------------------
# 2) CONFIGURACIÓN
# ---------------------------------------------------------------------------
CARPETA_DOCS = RAIZ / "docs"                          # acá están los documentos del caso
RUTA_DB = RAIZ / "chroma_db"                          # acá ChromaDB guarda la base en disco
VERSION_INGESTA = "pe3-v2-tablas-stemming"             # cambia cuando cambia la forma de ingestar: detecta bases viejas
CANDIDATOS = 20                                       # cuántos candidatos trae cada buscador antes de fusionar
RRF_K = 60                                            # constante estándar de la fusión RRF
UMBRAL_CONFIANZA_RAG = 0.90                           # igual que el umbral de la PE1: por debajo, revisión humana

# Dos configuraciones de fragmentado para comparar en la medición: (tamaño en caracteres, solape en caracteres)
CONFIGS = {"A": (500, 50), "B": (1000, 150)}
CONFIG_DEFECTO = os.getenv("FRAGMENTADO", "A").upper()  # cambiar a la que gane en la medición


# ---------------------------------------------------------------------------
# 3) EMBEDDINGS (convierten texto en números para buscar por significado)
# ---------------------------------------------------------------------------
def tokenizar(texto: str) -> list:
    """Parte un texto en palabras en minúscula y sin tildes. Lo usa BM25 y el embedder de prueba."""
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", sin_tildes.lower())


# Palabras muy comunes que no ayudan a encontrar nada ("el", "que", "cual"...). BM25 las ignora para no distraerse.
STOPWORDS = set("""a al algo algun alguna algunas alguno algunos ante antes como con contra cual cuales cuando cuanto cuantos
cada de del desde dice dispone donde durante e el ella ellas ellos en entre era es esa esas ese eso esos esta estan estar este esto
establece indica
estos fue ha han hay la las le les lo los mas me mi mis mucho muy no nos o otra otro para pero poco por porque puede
pueden puedo que quien quienes se ser si sin sobre son su sus tambien tanto te tiene tienen todo todos tu un una uno unos
y ya yo""".split())


def raiz(palabra: str) -> str:
    """Stemming casero: se quita la 's' del plural y las palabras largas se recortan a sus primeras 5 letras
    (caja/cajas -> 'caja'; rendir/rendición/rendiciones -> 'rendi'). Los números y códigos (88, 2026, r05) se dejan enteros."""
    if any(c.isdigit() for c in palabra):
        return palabra
    if len(palabra) > 3 and palabra.endswith("s"):
        palabra = palabra[:-1]
    return palabra[:5]


def tokenizar_bm25(texto: str) -> list:
    """Igual que tokenizar() pero sin palabras vacías y con raíces: es lo que usa BM25."""
    return [raiz(t) for t in tokenizar(texto) if t not in STOPWORDS]


class EmbedderHash:
    """
    Embedder SIMPLE y offline (cuenta palabras con 'hash'). Solo sirve para TESTS rápidos sin descargar modelos.
    La búsqueda real usa el modelo de ChromaDB (EMBEDDER=onnx, por defecto).
    """
    def __init__(self):
        pass

    @staticmethod
    def name() -> str:
        return "embedder_hash_pruebas"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict):
        return EmbedderHash()

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list:
        return ["cosine", "l2", "ip"]

    def embed_query(self, input):
        return self.__call__(input)

    def __call__(self, input):
        vectores = []
        for texto in input:
            v = [0.0] * 256                                              # vector de 256 posiciones
            for palabra in tokenizar(texto):
                posicion = int(hashlib.md5(palabra.encode()).hexdigest(), 16) % 256  # palabra -> posición
                v[posicion] += 1.0
            norma = sum(x * x for x in v) ** 0.5 or 1.0                  # normalizamos el largo del vector
            vectores.append([x / norma for x in v])
        return vectores


def crear_embedder():
    """Elige el embedder: 'onnx' (modelo real de ChromaDB, por defecto) o 'hash' (solo pruebas)."""
    if os.getenv("EMBEDDER", "onnx").lower() == "hash":
        return EmbedderHash()
    from chromadb.utils import embedding_functions
    return embedding_functions.DefaultEmbeddingFunction()  # modelo all-MiniLM-L6-v2 (se descarga una vez)


def cliente_chroma(ruta_db):
    """Abre (o crea) la base de datos ChromaDB guardada en disco."""
    return chromadb.PersistentClient(path=str(ruta_db), settings=Settings(anonymized_telemetry=False))


def nombre_coleccion(tamano: int, solape: int) -> str:
    """Cada configuración de fragmentado tiene su propia colección."""
    return f"caja_chica_{tamano}_{solape}"


# ---------------------------------------------------------------------------
# 4) INGESTA: leer -> limpiar -> dividir en secciones -> dividir en fragmentos -> guardar
# ---------------------------------------------------------------------------
_cache_lectura = {}  # evita volver a leer el PDF de 576 páginas si se ingesta dos veces en la misma ejecución


def leer_documento(ruta: Path) -> list:
    """Devuelve una lista de (número_de_página, texto). Los .txt y .md se leen como una sola 'página'."""
    clave = (str(ruta), ruta.stat().st_mtime)
    if clave in _cache_lectura:
        return _cache_lectura[clave]
    if ruta.suffix.lower() == ".pdf":
        lector = PdfReader(str(ruta))
        paginas = [(n + 1, p.extract_text() or "") for n, p in enumerate(lector.pages)]
    else:
        paginas = [(None, ruta.read_text(encoding="utf-8"))]
    _cache_lectura[clave] = paginas
    return paginas


def limpiar(texto: str) -> str:
    """Limpieza básica: quita números de página sueltos, une palabras cortadas y ordena espacios."""
    texto = texto.replace("\x0c", " ")                          # salto de página de los PDF
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)              # palabra cortada con guion al final de línea
    texto = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", texto)           # líneas que son solo un número (nº de página)
    texto = re.sub(r"[ \t]+", " ", texto)                       # espacios repetidos -> uno solo
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)                  # muchas líneas en blanco -> una
    return texto.strip()


# Un artículo empieza en una línea con ARTICULO 87 / ARTICULO 1° ...; un anexo con ANEXO N° 58
ENCABEZADO_NORMA = re.compile(r"(?m)^\s*(ARTICULO|ANEXO)\s+N?[°º]?\s*(\d+)")
ENCABEZADO_MD = re.compile(r"(?m)^#{1,3}\s+(.+)$")


def dividir_en_secciones(paginas: list, nombre_archivo: str) -> list:
    """
    Parte el documento en SECCIONES (artículos, anexos o títulos). Devuelve una lista de
    {"seccion": "Artículo 87", "pagina": 47, "texto": "..."}. La sección se guarda como metadato.
    """
    secciones = []
    actual = {"seccion": "Encabezado", "pagina": paginas[0][0], "texto": ""}  # lo que viene antes del primer título
    es_md = nombre_archivo.lower().endswith(".md")
    patron = ENCABEZADO_MD if es_md else ENCABEZADO_NORMA
    for pagina, crudo in paginas:
        texto = limpiar(crudo)
        posicion = 0
        for m in patron.finditer(texto):
            actual["texto"] += texto[posicion:m.start()]             # el texto anterior pertenece a la sección actual
            if actual["texto"].strip():
                secciones.append(actual)                             # cerramos la sección anterior
            if es_md:
                nombre = m.group(1).strip()
            else:
                nombre = ("Artículo " if m.group(1) == "ARTICULO" else "Anexo ") + m.group(2)
            actual = {"seccion": nombre, "pagina": pagina, "texto": ""}  # empieza una sección nueva
            posicion = m.start()
        actual["texto"] += texto[posicion:] + "\n"
    if actual["texto"].strip():
        secciones.append(actual)
    return secciones


def dividir(texto: str, tamano: int, solape: int) -> list:
    """Parte un texto en fragmentos de ~'tamano' caracteres, repitiendo ~'solape' caracteres entre uno y otro."""
    texto = texto.strip()
    if len(texto) <= tamano:
        return [texto] if texto else []
    fragmentos, inicio = [], 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))
        if fin < len(texto):                                         # si no es el final, buscamos un buen lugar para cortar
            corte = max(texto.rfind(". ", inicio, fin), texto.rfind("\n", inicio, fin))   # 1º: fin de oración o de línea
            if corte <= inicio + tamano * 0.6:
                corte = texto.rfind(" ", inicio, fin)                # 2º: al menos un espacio (no cortar palabras)
            if corte > inicio + tamano * 0.6:                        # solo si el corte no deja un fragmento muy chico
                fin = corte + 1
        pedazo = texto[inicio:fin].strip()
        if pedazo:
            fragmentos.append(pedazo)
        if fin >= len(texto):
            break
        inicio = max(fin - solape, inicio + 1)                       # retrocedemos ~'solape' caracteres (y siempre avanzamos)
        siguiente = texto.find(" ", inicio, fin)                     # el nuevo fragmento arranca en una palabra entera
        if siguiente != -1:
            inicio = siguiente + 1
    return fragmentos


def contiene(texto: str, clave: str) -> bool:
    """¿El texto contiene la frase clave? Ignora mayúsculas y espacios/saltos de línea repetidos (conserva las tildes)."""
    normalizar = lambda t: re.sub(r"\s+", " ", t.lower())
    return normalizar(clave) in normalizar(texto)


def es_fila_de_tabla(linea: str) -> bool:
    """¿La línea es una fila de tabla (celdas separadas por '|')? Las líneas separadoras '|---|' no cuentan."""
    return linea.count("|") >= 2 and not set(linea.strip()) <= set("|-: ")


def expandir_tablas(texto: str) -> str:
    """
    Convierte cada fila de una tabla en una oración autosuficiente que repite los encabezados:
      'Sec. de Obras Públicas | $ 150.000 | $ 100.000 | Diana Markovic'
      -> 'Cajas chicas por área - ÁREA: Sec. de Obras Públicas; MONTO: $ 150.000; VALOR M.: $ 100.000; RESPONSABLE: Diana Markovic.'
    Sin esto, un fragmento con solo la fila no dice qué significa cada número.
    """
    lineas, salida, i = texto.split("\n"), [], 0
    while i < len(lineas):
        if not es_fila_de_tabla(lineas[i]):
            salida.append(lineas[i])
            i += 1
            continue
        bloque = []
        while i < len(lineas) and (es_fila_de_tabla(lineas[i]) or set(lineas[i].strip()) <= set("|-: ") and lineas[i].strip()):
            if es_fila_de_tabla(lineas[i]):
                bloque.append(lineas[i])
            i += 1
        if len(bloque) < 2:                                       # una sola línea: no es una tabla
            salida.extend(bloque)
            continue
        previas = [l.strip().lstrip("#").strip() for l in salida if l.strip()]
        titulo = previas[-1][:110] if previas else ""             # la última línea con texto antes de la tabla
        celdas = lambda fila: [c.strip() for c in fila.strip().strip("|").split("|")]
        encabezados = celdas(bloque[0])
        for fila in bloque[1:]:
            pares = "; ".join(f"{h}: {c}" for h, c in zip(encabezados, celdas(fila)))
            salida.append(f"{titulo} - {pares}.")
    return "\n".join(salida)


def cargar_permisos(carpeta: Path) -> dict:
    """Lee docs/permisos.json: qué roles pueden leer cada documento."""
    return json.loads((carpeta / "permisos.json").read_text(encoding="utf-8"))


def ingestar(carpeta=CARPETA_DOCS, tamano: int = None, solape: int = None, ruta_db=RUTA_DB,
             solo: list = None, verbose: bool = True):
    """
    Procesa los documentos de 'carpeta' y devuelve la colección de ChromaDB.
    Cada fragmento guarda como metadatos: documento, sección, página y los roles que pueden leerlo.
    Regla de seguridad: un documento que NO figura en permisos.json NO se ingesta (se deniega por defecto).
    """
    carpeta = Path(carpeta)
    if tamano is None or solape is None:
        tamano, solape = CONFIGS[CONFIG_DEFECTO]
    permisos = cargar_permisos(carpeta)
    inicio = time.perf_counter()

    ids, textos, metadatos = [], [], []
    for ruta in sorted(carpeta.iterdir()):
        if ruta.suffix.lower() not in (".pdf", ".txt", ".md"):       # ignoramos permisos.json y otros archivos
            continue
        if solo and ruta.name not in solo:
            continue
        if ruta.name not in permisos:                                # SIN permisos declarados -> no se ingesta
            log.warning("'%s' no figura en permisos.json: NO se ingesta (denegado por defecto).", ruta.name)
            continue
        roles = permisos[ruta.name]
        cantidad = 0
        for sec in dividir_en_secciones(leer_documento(ruta), ruta.name):
            for pedazo in dividir(expandir_tablas(sec["texto"]), tamano, solape):
                numero = len(ids)
                ids.append(f"{ruta.name}::{numero}")                 # identificador único del fragmento
                # Anteponemos la sección al texto: así los números de artículo y los títulos también se buscan.
                textos.append(f"({sec['seccion']}) {pedazo}")
                metadatos.append({
                    "documento": ruta.name,
                    "seccion": sec["seccion"],
                    "pagina": sec["pagina"] if sec["pagina"] is not None else 0,
                    "roles": ",".join(roles),                        # para mostrar (ej: "empleado,tesoreria")
                    "rol_empleado": "empleado" in roles,             # un True/False por rol: es lo que filtra Chroma
                    "rol_tesoreria": "tesoreria" in roles,
                    "fragmento_n": numero,
                    "tamano": tamano,
                    "solape": solape,
                })
                cantidad += 1
        if verbose:
            print(f"  {ruta.name:<42} -> {cantidad:>5} fragmentos  (roles: {', '.join(roles)})")

    cliente = cliente_chroma(ruta_db)
    nombre = nombre_coleccion(tamano, solape)
    try:
        cliente.delete_collection(nombre)                            # si ya existía, la recreamos desde cero
    except Exception:
        pass
    coleccion = cliente.create_collection(name=nombre, embedding_function=crear_embedder(),
                                          metadata={"version_ingesta": VERSION_INGESTA})
    for i in range(0, len(ids), 128):                                # guardamos de a 128 fragmentos
        coleccion.add(ids=ids[i:i + 128], documents=textos[i:i + 128], metadatas=metadatos[i:i + 128])
        if verbose and len(ids) > 400:
            print(f"    ...guardados {min(i + 128, len(ids))}/{len(ids)}", end="\r")
    if verbose:
        print(f"\n  Colección '{nombre}': {coleccion.count()} fragmentos (tamaño {tamano}, solape {solape}) "
              f"en {time.perf_counter() - inicio:.1f} s")
    return coleccion


def abrir_coleccion(tamano: int = None, solape: int = None, ruta_db=RUTA_DB):
    """Abre una colección ya ingestada (si no existe, explica qué comando correr)."""
    if tamano is None or solape is None:
        tamano, solape = CONFIGS[CONFIG_DEFECTO]
    try:
        coleccion = cliente_chroma(ruta_db).get_collection(nombre_coleccion(tamano, solape), embedding_function=crear_embedder())
    except Exception as e:
        raise SystemExit(f"No encuentro la colección ({e}). Primero corré:  python src/ingestar.py")
    if (coleccion.metadata or {}).get("version_ingesta") != VERSION_INGESTA:   # base armada con una versión vieja
        raise SystemExit("La base está desactualizada (se armó con otra versión de la ingesta). Corré de nuevo:  python src/ingestar.py")
    return coleccion


# ---------------------------------------------------------------------------
# 5) BÚSQUEDA: vectorial + BM25 + fusión RRF, con permisos aplicados ANTES de buscar
# ---------------------------------------------------------------------------
class Usuario(BaseModel):
    """Quién hace la consulta. El rol es un Enum: cualquier valor desconocido es un error (falla cerrado)."""
    id_empleado: str = Field(pattern=r"^EMP-\d{3}$")
    rol: Rol


def filtro_permisos(usuario: Usuario) -> dict:
    """Filtro de ChromaDB: solo fragmentos marcados como legibles por el rol del usuario."""
    return {f"rol_{usuario.rol.value}": True}


_cache_corpus = {}  # índice BM25 ya armado, por (colección, rol)


def _corpus_permitido(coleccion, usuario: Usuario) -> dict:
    """
    Trae de la base SOLO los fragmentos que el rol puede leer (el filtro va en la consulta a la base)
    y arma con ellos el índice BM25. Así el texto restringido nunca entra al ranking de un usuario sin permiso.
    """
    clave = (str(coleccion.id), usuario.rol.value)
    if clave not in _cache_corpus:
        datos = coleccion.get(where=filtro_permisos(usuario), include=["documents", "metadatas"])
        tokens = [tokenizar_bm25(t) for t in datos["documents"]]
        _cache_corpus[clave] = {
            "ids": datos["ids"], "textos": datos["documents"], "metadatos": datos["metadatas"],
            "bm25": BM25Okapi(tokens) if tokens else None,
        }
    return _cache_corpus[clave]


def fusionar(ranking_vec: list, ranking_bm25: list, k: int, garantizar: bool = True) -> list:
    """
    Fusiona los dos rankings (listas de ids, el mejor primero) y devuelve hasta k pares (id, puntaje).

    1) RRF: cada fragmento suma 1/(60 + posición) por cada ranking en el que aparece.
    2) Garantía: el MEJOR resultado de cada buscador entra siempre al top-k. Sin esto, varios fragmentos "de consenso"
       (que aparecen en los dos rankings por palabras genéricas) pueden desplazar al único fragmento que contiene
       un nombre o código exacto, que solo BM25 encontró.
    """
    fusion = {}
    for ranking in (ranking_vec, ranking_bm25):
        for posicion, fid in enumerate(ranking, start=1):
            fusion[fid] = fusion.get(fid, 0.0) + 1.0 / (RRF_K + posicion)
    ordenados = sorted(fusion.items(), key=lambda x: x[1], reverse=True)        # orden RRF puro
    if not garantizar or k < 2:
        return ordenados[:k]
    garantizados = []
    for ranking in (ranking_bm25, ranking_vec):                                  # el mejor de cada buscador
        if ranking and ranking[0] not in garantizados:
            garantizados.append(ranking[0])
    resto = [x for x in ordenados if x[0] not in garantizados]                   # el resto, en orden RRF
    elegidos = [(fid, fusion[fid]) for fid in garantizados] + resto[: max(k - len(garantizados), 0)]
    return sorted(elegidos, key=lambda x: x[1], reverse=True)[:k]


def buscar(consulta: str, usuario: Usuario, k: int = 5, modo: str = "hibrida", coleccion=None) -> list:
    """
    Devuelve hasta k fragmentos: [{documento, seccion, pagina, fragmento_id, texto, puntaje, ...}].
    modo: "hibrida" (vectorial + BM25 fusionados), "vectorial" o "bm25".
    """
    coleccion = coleccion or abrir_coleccion()
    corpus = _corpus_permitido(coleccion, usuario)           # solo lo que este rol puede leer
    if not corpus["ids"]:
        return []
    por_id = {i: n for n, i in enumerate(corpus["ids"])}    # id -> posición en el corpus permitido

    # --- Búsqueda vectorial (por significado). El filtro de permisos va DENTRO de la consulta a Chroma. ---
    ranking_vec, distancias = [], {}
    if modo in ("hibrida", "vectorial"):
        r = coleccion.query(query_texts=[consulta], n_results=min(CANDIDATOS, len(corpus["ids"])),
                            where=filtro_permisos(usuario), include=["distances"])
        ranking_vec = r["ids"][0]
        distancias = dict(zip(r["ids"][0], r["distances"][0]))

    # --- Búsqueda por palabras clave (BM25) sobre el corpus ya filtrado ---
    ranking_bm25, puntajes_bm25 = [], {}
    if modo in ("hibrida", "bm25") and corpus["bm25"] is not None:
        puntajes = corpus["bm25"].get_scores(tokenizar_bm25(consulta))
        orden = sorted(range(len(puntajes)), key=lambda n: puntajes[n], reverse=True)[:CANDIDATOS]
        for n in orden:
            if puntajes[n] > 0:                              # puntaje 0 = ninguna palabra coincide
                ranking_bm25.append(corpus["ids"][n])
                puntajes_bm25[corpus["ids"][n]] = float(puntajes[n])

    # --- Fusión de los dos rankings (RRF + garantía del mejor de cada buscador) ---
    if modo == "vectorial":
        ordenados = [(fid, 1.0 / (1.0 + distancias[fid])) for fid in ranking_vec]
    elif modo == "bm25":
        ordenados = [(fid, puntajes_bm25[fid]) for fid in ranking_bm25]
    else:
        ordenados = fusionar(ranking_vec, ranking_bm25, k)

    resultados = []
    for fid, puntaje in ordenados[:k]:
        n = por_id[fid]
        m = corpus["metadatos"][n]
        resultados.append({
            "documento": m["documento"], "seccion": m["seccion"], "pagina": m["pagina"],
            "fragmento_id": fid, "texto": corpus["textos"][n], "puntaje": round(puntaje, 5),
            "pos_vectorial": ranking_vec.index(fid) + 1 if fid in ranking_vec else None,
            "pos_bm25": ranking_bm25.index(fid) + 1 if fid in ranking_bm25 else None,
        })
    return resultados


# ---------------------------------------------------------------------------
# 6) RESPUESTA CON FUENTES
# ---------------------------------------------------------------------------
class Fuente(BaseModel):
    """Un fragmento citado en la respuesta."""
    documento: str
    seccion: str
    fragmento_id: str
    puntaje: float


class RespuestaLLM(BaseModel):
    """Lo que tiene que devolver el modelo (se valida igual que en la PE2)."""
    respuesta: str = Field(min_length=5, max_length=1500)
    fragmentos_citados: list[str]                      # ej: ["F1", "F3"]
    evidencia_suficiente: bool                         # False = los fragmentos no alcanzan para responder
    confianza: float = Field(ge=0, le=1)


class SalidaRAG(BaseModel):
    """Salida final del sistema: la respuesta, sus fuentes y si hace falta una persona."""
    respuesta: str
    requiere_humano: bool
    motivo_humano: Optional[str] = None
    confianza: float = Field(ge=0, le=1)
    fuentes: list[Fuente]


def _formatear_fragmentos(fragmentos: list) -> str:
    """Arma el bloque de texto con los fragmentos F1, F2, ... que se envía al modelo."""
    return "\n\n".join(
        f"[F{n}] documento: {f['documento']} | sección: {f['seccion']}\n{f['texto']}"
        for n, f in enumerate(fragmentos, start=1))


def _humano(respuesta: str, motivo: str, metricas: dict) -> dict:
    """Resultado estándar cuando el sistema NO responde por falta de evidencia: requiere_humano = true."""
    salida = SalidaRAG(respuesta=respuesta, requiere_humano=True, motivo_humano=motivo, confianza=0.0, fuentes=[])
    return {"ok": True, "salida": salida.model_dump(), "metricas": metricas}


def responder(entrada: Entrada, k: int = 6, coleccion=None) -> dict:
    """
    Recibe la Entrada de la PE2 (texto de la consulta + id_empleado + área + rol) y devuelve la salida con fuentes.
    Si no hay evidencia suficiente: requiere_humano = true (el sistema NO inventa).
    """
    inicio = time.perf_counter()
    usuario = Usuario(id_empleado=entrada.id_empleado, rol=entrada.rol)
    fragmentos = buscar(entrada.texto_ticket, usuario, k=k, coleccion=coleccion)

    # Sin fragmentos para este rol: no hace falta ni llamar al modelo.
    if not fragmentos:
        return _humano("No encuentro esa información en los documentos disponibles para tu rol.",
                       "SIN_FRAGMENTOS", armar_metricas(inicio, 0, 0, 0))

    # Prompt armado SOLO con lo recuperado (y ya filtrado por rol).
    sistema = cargar_plantilla("rag_sistema.txt")
    usuario_base = cargar_plantilla("rag_usuario.txt").format(
        id_empleado=entrada.id_empleado, rol=entrada.rol.value, area=entrada.area,
        pregunta=entrada.texto_ticket, fragmentos=_formatear_fragmentos(fragmentos))

    if CONFIG["var_clave"] and not os.getenv(CONFIG["var_clave"]):   # Ollama no necesita clave
        return error_estructurado("CLAVE_FALTANTE", f"Falta {CONFIG['var_clave']} en el archivo .env.", [],
                                  armar_metricas(inicio, 0, 0, 0))

    cliente = crear_cliente()
    mensaje, resp, campos, ultimo = usuario_base, None, [], ""
    tokens_in = tokens_out = 0
    for intento in (1, 2):                                           # 1 llamada + 1 reintento, como en la PE2
        try:
            texto, t_in, t_out, _ = llamar_modelo(cliente, sistema, mensaje)
        except ErrorProveedor as e:
            return error_estructurado(e.codigo, str(e), [], armar_metricas(inicio, tokens_in, tokens_out, intento))
        tokens_in += t_in
        tokens_out += t_out
        ultimo = texto
        try:
            resp = RespuestaLLM.model_validate_json(limpiar_json(texto))
            break
        except ValidationError as e:
            campos = resumir_errores(e)
            log.warning("RAG intento %d: respuesta NO validó. Campos: %s", intento, campos)
            mensaje = (usuario_base + "\n\nTu respuesta anterior no fue válida. Problemas: "
                       + "; ".join(f"{c['campo']}: {c['problema']}" for c in campos) + ". Respondé SOLO con el JSON corregido.")
    metricas = armar_metricas(inicio, tokens_in, tokens_out, intento)
    if resp is None:
        return error_estructurado("SALIDA_INVALIDA", "El modelo no devolvió una respuesta válida tras 2 intentos.",
                                  campos, metricas, respuesta_cruda=ultimo)

    # Solo aceptamos citas a fragmentos que realmente se enviaron (F1..Fk); cualquier otra se descarta.
    citados = []
    for etiqueta in resp.fragmentos_citados:
        m = re.fullmatch(r"F(\d+)", etiqueta.strip().upper())
        if m and 1 <= int(m.group(1)) <= len(fragmentos):
            f = fragmentos[int(m.group(1)) - 1]
            if f["fragmento_id"] not in [c["fragmento_id"] for c in citados]:
                citados.append(f)

    # Reglas de seguridad sobre la respuesta (las decide Python, no el modelo).
    motivo = None
    if not resp.evidencia_suficiente:
        motivo = "EVIDENCIA_INSUFICIENTE"
    elif not citados:
        motivo = "SIN_CITAS_VALIDAS"
    elif resp.confianza < UMBRAL_CONFIANZA_RAG:
        motivo = "CONFIANZA_BAJA"
    fuentes = [] if motivo == "EVIDENCIA_INSUFICIENTE" else [
        Fuente(documento=f["documento"], seccion=f["seccion"], fragmento_id=f["fragmento_id"], puntaje=f["puntaje"])
        for f in citados]
    salida = SalidaRAG(respuesta=resp.respuesta, requiere_humano=motivo is not None, motivo_humano=motivo,
                       confianza=resp.confianza, fuentes=fuentes)
    return {"ok": True, "salida": salida.model_dump(), "fragmentos_recuperados": [f["fragmento_id"] for f in fragmentos],
            "metricas": metricas}
