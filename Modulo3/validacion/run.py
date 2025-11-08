"""
run.py
Comparación simple entre:
AG.py (original, sin modificar)
AGF.py (original corregido)
mi_version.py (implementación propia OOP + NumPy + Pandas)

Métrica:
Si corre o falla
Distancia encontrada
Tiempo de ejecución

Casos:
10, 50, 100, 1000 ciudades
Semilla fija: 777
"""

import time
import random
import numpy as np
import pandas as pd

SEED = 777
TAM_CIUDADES = [10, 50, 100, 1000]


def generar_coordenadas(n):
    random.seed(SEED)
    np.random.seed(SEED)
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]


def correr_AG(n):
    try:
        import AG
    except Exception as e:
        return False, None, 0.0, f"Import error: {e}"

    try:
        coords = generar_coordenadas(n)
        municipios = [AG.municipio(x, y) for x, y in coords]
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.perf_counter()
        mejor = AG.algoritmoGenetico(
            poblacion=municipios,
            tamanoPoblacion=100,
            indivSelecionados=20,
            razonMutacion=0.01,
            generaciones=200
        )
        t1 = time.perf_counter()
        dist = sum(
            mejor[i].distancia(mejor[(i + 1) % len(mejor)])
            for i in range(len(mejor))
        )
        return True, dist, t1 - t0, None
    except Exception as e:
        return False, None, 0.0, f"Run error: {e}"


def correr_AGF(n):
    try:
        import AGF
    except Exception as e:
        return False, None, 0.0, f"Import error: {e}"

    try:
        coords = generar_coordenadas(n)
        municipios = [AGF.municipio(x, y) for x, y in coords]
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.perf_counter()
        mejor = AGF.algoritmoGenetico(
            poblacion=municipios,
            tamanoPoblacion=100,
            indivSelecionados=20,
            razonMutacion=0.01,
            generaciones=200
        )
        t1 = time.perf_counter()
        dist = sum(
            mejor[i].distancia(mejor[(i + 1) % len(mejor)])
            for i in range(len(mejor))
        )
        return True, dist, t1 - t0, None
    except Exception as e:
        return False, None, 0.0, f"Run error: {e}"


def correr_mi_version(n):
    try:
        import mi_version as mv
    except Exception as e:
        return False, None, 0.0, f"Import error: {e}"

    try:
        coords = generar_coordenadas(n)
        municipios = [mv.Municipio(x, y, nombre=f"Ciudad_{i+1}")
                      for i, (x, y) in enumerate(coords)]
        ga = mv.AlgoritmoGenetico(
            municipios=municipios,
            tam_poblacion=100,
            n_elite=20,
            tasa_mutacion=0.01
        )
        random.seed(SEED)
        np.random.seed(SEED)
        t0 = time.perf_counter()
        mejor = ga.run(generaciones=200, verbose=False)
        t1 = time.perf_counter()
        dist = mejor.distancia_total()
        return True, dist, t1 - t0, None
    except Exception as e:
        return False, None, 0.0, f"Run error: {e}"


if __name__ == "__main__":
    resultados = []
    ag_ok = True
    agf_ok = True
    mi_ok = True

    print("\nComparación de implementaciones AG para TSP")
    print(f"Semilla usada: {SEED}\n")

    for n in TAM_CIUDADES:
        print(f"=== {n} ciudades ===")

        if ag_ok:
            ok, dist, t, err = correr_AG(n)
            ag_ok = ok
            print(f"AG.py        -> {'OK' if ok else 'ERROR'}  dist={dist}  t={t:.6f}s", ("" if not err else f"  ({err})"))
            resultados.append({"Impl": "AG.py", "Ciudades": n,"Estado": "OK" if ok else "ERROR","Distancia": dist, "Tiempo (s)": t})
        else:
            resultados.append({"Impl": "AG.py", "Ciudades": n,"Estado": "DESCARTADA", "Distancia": None, "Tiempo (s)": 0.0})

        if agf_ok:
            ok, dist, t, err = correr_AGF(n)
            agf_ok = ok
            print(f"AGF.py       -> {'OK' if ok else 'ERROR'}  dist={dist}  t={t:.6f}s", ("" if not err else f"  ({err})"))
            resultados.append({"Impl": "AGF.py", "Ciudades": n,"Estado": "OK" if ok else "ERROR","Distancia": dist, "Tiempo (s)": t})
        else:
            resultados.append({"Impl": "AGF.py", "Ciudades": n,"Estado": "DESCARTADA", "Distancia": None, "Tiempo (s)": 0.0})

        if mi_ok:
            ok, dist, t, err = correr_mi_version(n)
            mi_ok = ok
            print(f"mi_version.py-> {'OK' if ok else 'ERROR'}  dist={dist}  t={t:.6f}s", ("" if not err else f"  ({err})"))
            resultados.append({"Impl": "mi_version.py", "Ciudades": n,"Estado": "OK" if ok else "ERROR","Distancia": dist, "Tiempo (s)": t})
        else:
            resultados.append({"Impl": "mi_version.py", "Ciudades": n,"Estado": "DESCARTADA", "Distancia": None, "Tiempo (s)": 0.0})

        print()

    print("\n Tabla comparativa:\n")
    df = pd.DataFrame(resultados)
    print(df.to_string(index=False))
