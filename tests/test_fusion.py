"""
Tests de la fusión de rankings (funciones puras: sin base de datos, sin modelo, sin internet).
Uso:  python tests/test_fusion.py
"""
import os, sys
from pathlib import Path

os.environ["EMBEDDER"] = "hash"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import rag


def ids(resultado):
    return [fid for fid, _ in resultado]


def test_un_fragmento_en_ambos_rankings_gana_en_rrf_puro():
    r = rag.fusionar(["a", "b", "c"], ["c", "x", "y"], k=3, garantizar=False)
    assert ids(r)[0] == "c", r                      # "c" aparece en los dos rankings


def test_sin_garantia_el_resultado_exacto_de_bm25_se_pierde():
    """Reproduce el problema real: 3 fragmentos de consenso desplazan al resultado exacto (solo en BM25)."""
    vec = ["ruido1", "a", "b", "c"]
    bm25 = ["EXACTO", "a", "b", "c"]
    assert "EXACTO" not in ids(rag.fusionar(vec, bm25, k=3, garantizar=False))


def test_con_garantia_el_mejor_de_bm25_siempre_entra():
    vec = ["ruido1", "a", "b", "c"]
    bm25 = ["EXACTO", "a", "b", "c"]
    assert "EXACTO" in ids(rag.fusionar(vec, bm25, k=3, garantizar=True))


def test_con_garantia_el_mejor_vectorial_tambien_entra():
    vec = ["SEMANTICO", "a", "b", "c"]
    bm25 = ["ruido", "a", "b", "c"]
    r = ids(rag.fusionar(vec, bm25, k=3, garantizar=True))
    assert "SEMANTICO" in r and "ruido" in r, r


def test_cuando_los_buscadores_coinciden_no_cambia_nada():
    r1 = ids(rag.fusionar(["a", "b", "c", "d"], ["a", "b", "c", "d"], k=3, garantizar=True))
    r2 = ids(rag.fusionar(["a", "b", "c", "d"], ["a", "b", "c", "d"], k=3, garantizar=False))
    assert r1 == r2 == ["a", "b", "c"]


def test_k_uno_y_listas_vacias():
    assert len(rag.fusionar(["a"], ["b"], k=1)) == 1
    assert rag.fusionar([], [], k=3) == []
    assert ids(rag.fusionar([], ["b", "c"], k=3)) == ["b", "c"]      # si un buscador no trae nada, sirve el otro


def test_nunca_devuelve_mas_de_k_ni_repetidos():
    r = ids(rag.fusionar(["a", "b", "c", "d", "e"], ["e", "d", "c", "b", "a"], k=3))
    assert len(r) == 3 and len(set(r)) == 3, r


if __name__ == "__main__":
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    print("=== TESTS DE FUSIÓN DE RANKINGS ===")
    for nombre, funcion in pruebas:
        try:
            funcion(); print(f"[OK ] {nombre}")
        except AssertionError as e:
            fallos += 1; print(f"[MAL] {nombre}: {e}")
    print(f"\nRESUMEN: {len(pruebas) - fallos}/{len(pruebas)} tests correctos")
    sys.exit(1 if fallos else 0)
