# fde-caja-chica — Asistente de rendición de caja chica

> **PE2:** motor LLM con validación (`src/motor.py`). **PE3:** RAG con permisos y búsqueda híbrida (`src/rag.py`, ver sección *PE3* más abajo).

Motor LLM (PE2): recibe el texto de un ticket/factura y el área del empleado, consulta a un modelo de lenguaje y devuelve una
decisión (`aprobar` / `revision_humana` / `rechazar`) validada con Pydantic, junto con tokens, costo en USD y latencia en ms.

- **Proveedor y modelo usados en las mediciones de la PE2:** Groq, modelo `openai/gpt-oss-120b` (modelo de OpenAI servido por una API compatible con OpenAI, endpoint `https://api.groq.com/openai/v1`).
- **Normativa:** Decreto municipal 77/26 (cajas chicas 2026, General Juan Madariaga) y Decreto provincial 2980/00 (arts. 87–88).

## Estructura

```
fde-caja-chica/
├── README.md
├── requirements.txt
├── .env.example                 # nombres de variables (sin valores)
├── .gitignore                   # incluye .env
├── prompts/
│   ├── sistema.txt              # prompt de sistema (plantilla separada del código)
│   └── usuario.txt              # mensaje por consulta, con {huecos}
├── data/                        # consultas_prueba.json (PE2), preguntas_prueba.json y consultas_exactas.json (PE3)
├── docs/                        # documentos del caso + permisos.json (quién lee qué)
├── tests/test_permisos.py       # tests de permisos por rol (PE3)
├── logs/                        # registro crudo de cada corrida (JSON)
└── src/
    ├── motor.py                 # contrato Pydantic + llamada al modelo + medición + errores
    ├── correr_pruebas.py        # corre las 5 consultas, arma la tabla y guarda el log
    ├── probar_contrato.py       # prueba cada regla del contrato y la falla persistente (sin API)
    ├── rag.py                   # PE3: ingestar() + buscar() + responder()
    ├── ingestar.py              # PE3: UN comando que procesa los documentos
    ├── comparar_busquedas.py    # PE3: híbrida vs solo vectorial en consultas con términos exactos
    ├── medir_rag.py             # PE3: top-k con dos configuraciones de fragmentado
    └── correr_preguntas.py      # PE3: las 10 preguntas de prueba por el RAG completo
```

## Cómo correrlo desde un clon limpio

```bash
git clone https://github.com/YoseliLambertucci/fde_coder.git && cd fde_coder
python -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # En Windows: copy .env.example .env
# Abrí .env y pegá tu clave en GROQ_API_KEY= (gratis en console.groq.com)
python src/probar_contrato.py      # no usa API: prueba cada regla del contrato y el fallback
python src/motor.py                # demo: una ejecución exitosa y una con error
python src/correr_pruebas.py       # las 5 consultas + tabla de medición + log en logs/
```

## Proveedores del modelo

| PROVEEDOR | Clave | Modelo por defecto | Precio USD por millón de tokens (entrada / salida) |
|---|---|---|---|
| `groq` (por defecto) | `GROQ_API_KEY` | `openai/gpt-oss-120b` | 0,15 / 0,60 |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | 0,10 / 0,50 |
| `ollama` | ninguna | `llama3.1:8b` (local) | 0 / 0 |

Cambiar de proveedor no requiere tocar el código, solo `PROVEEDOR` en `.env`. Los precios se pueden sobrescribir con
`PRECIO_ENTRADA_USD_MTOK` y `PRECIO_SALIDA_USD_MTOK` (conviene verificarlos en la página oficial del proveedor).

## Contrato de datos

- `Entrada`: `id_empleado` (patrón `^EMP-\d{3}$`, ej. `EMP-101`), `area` (debe figurar en el Decreto 77/26), `texto_ticket` (10 a 3000 caracteres).
- `Salida`: `es_comprobante`, `comercio` (mín. 2), `fecha` (fecha real), `monto_total` (0 < x ≤ 10.000.000), `categoria` (Enum), `decision` (Enum), `confianza` (0 a 1), `justificacion` (10 a 500) y una regla entre campos: no se puede `aprobar` un comprobante al que le faltan comercio, fecha o monto.

## Qué hace cuando falla la validación

Si la respuesta del modelo no cumple el contrato `Salida`, registra **qué campo falló y por qué**, reintenta una vez
informándole al modelo el problema y, si vuelve a fallar, devuelve un error estructurado
(`{"ok": false, "error": {"codigo": "SALIDA_INVALIDA", "campos": [...], "respuesta_cruda": "..."}}`).
`python src/probar_contrato.py` muestra esta falla persistente con respuestas simuladas.
Los errores de la API (clave inválida, límite de uso, sin conexión, tiempo agotado, modelo inexistente) también se devuelven con código propio.

## Decisiones de diseño

- El modelo **extrae y clasifica**; la comparación contra el tope por pago y el umbral de confianza (90 %) las hace Python, porque son reglas exactas.
- El texto del ticket se trata como dato no confiable (defensa básica contra instrucciones escondidas en el ticket).
- Proveedor intercambiable desde `.env` (extra opcional de la PE2).
- Cada corrida de `correr_pruebas.py` guarda su registro crudo en `logs/` para poder auditar y reproducir.

---

# PE3 — RAG con permisos y búsqueda híbrida

## Documentos (`docs/`) y permisos (`docs/permisos.json`)

| Documento | Roles que pueden leerlo | Origen |
|---|---|---|
| `decreto_77-2026.txt` | empleado, tesoreria | Decreto 77/26 (la página 2 del PDF es una imagen y se transcribió a mano) |
| `decreto_2980-2000.pdf` | empleado, tesoreria | Decreto provincial 2980/00 (576 páginas) |
| `guia_rendicion_empleados.md` | empleado, tesoreria | Documento **ficticio** de apoyo |
| `restringido_saldos_globales_2026.md` | **solo tesoreria** | Documento **ficticio** restringido |
| `restringido_criterios_auditoria.md` | **solo tesoreria** | Documento **ficticio** restringido |

Un documento que no figura en `permisos.json` **no se ingesta** (denegado por defecto).

## Cómo correr la PE3

```bash
pip install -r requirements.txt
python tests/test_permisos.py      # tests de permisos (sin API, sin descargas)
python tests/test_fusion.py        # tests de la fusión de rankings (sin API, sin descargas)
python src/ingestar.py             # UN comando: lee, limpia, fragmenta y guarda en ChromaDB
python src/comparar_busquedas.py   # híbrida vs solo vectorial (términos exactos)
python src/medir_rag.py            # top-k con configuraciones A (500/50) y B (1000/150)
python src/correr_preguntas.py     # 10 preguntas por el RAG completo (usa el modelo del .env)
```

La primera vez, ChromaDB descarga un modelo de embeddings (~80 MB) y la ingesta puede tardar unos minutos.
Requiere Python 3.10 a 3.13 (versiones muy nuevas pueden no tener paquetes para `chromadb`).

## Piezas

- `ingestar(carpeta)` -> colección ChromaDB. Cada fragmento guarda `documento`, `seccion`, `pagina`, `roles`, `rol_empleado`, `rol_tesoreria`.
- `buscar(consulta, usuario, k)` -> fragmentos con `documento`, `fragmento_id`, `texto`, `puntaje`. Fusiona búsqueda vectorial y BM25 con RRF y una *garantía*: el mejor resultado de cada buscador siempre entra al top-k (así un nombre o código exacto hallado solo por BM25 no queda desplazado por fragmentos "de consenso").
  **El filtro por rol va en la consulta a la base** (`where`), tanto para la búsqueda vectorial como para armar el índice BM25.
- `responder(entrada)` -> `Salida` con `respuesta`, `fuentes`, `confianza` y `requiere_humano`. Si no hay evidencia suficiente: `requiere_humano = true`.

## Mejoras de la ingesta (iteración tras la primera corrida real)

La primera corrida de `correr_preguntas.py` dio 7/12: el sistema no inventó nada, pero pidió revisión humana en preguntas cuya respuesta sí estaba en los documentos.
Causas y correcciones: (1) las filas de tabla se ingestaban sueltas, sin los nombres de las columnas -> `expandir_tablas()` convierte cada fila en una oración autosuficiente;
(2) BM25 no relacionaba "rendir" con "rendición" -> stemming casero (`raiz()`: quita la "s" del plural y recorta a 5 letras);
(3) se envían 6 fragmentos al modelo en vez de 4; (4) el umbral de confianza del RAG sube a 0,90 (el de la PE1) y `correr_preguntas.py` verifica que la fuente citada contenga la evidencia (una respuesta equivocada con el documento correcto ya no pasa como buena). La colección guarda una marca de versión (`VERSION_INGESTA`): si la base es de una versión vieja, el sistema pide volver a ingestar.

## Tests de permisos

`tests/test_permisos.py` verifica, para el rol **empleado** y el rol **tesoreria**, que: (1) un empleado no recibe nada restringido en los resultados (vectorial, BM25 e híbrida),
(2) el prompt enviado al modelo no contiene texto restringido, (3) el filtro viaja dentro de la consulta a la base, (4) Tesorería sí lee lo restringido (control positivo),
(5) un rol desconocido falla cerrado y (6) un documento sin permisos declarados no se ingesta.
