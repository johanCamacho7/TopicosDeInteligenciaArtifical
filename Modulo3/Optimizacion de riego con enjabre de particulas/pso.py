#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Entradas / salidas
ruta_matriz_idoneidad = "idoneidad_total.npy"  # también acepta .csv
ruta_csv_salida = "sensores_pso.csv"

# Problema
n_sensores = 20
dist_minima_celdas = 5        # F4: distancia mínima entre sensores (en celdas)
peso_penalizacion_dist = 2.0  # peso de la penalización por distancia
penalizacion_nan = 10.0       # penalizo caer en NaN

# PSO (doble fase)
n_particulas = 140
iteraciones_totales = 200
porc_fase_independiente = 0.25
porc_fase_social = 0.75

# Fase 1: más individual
w_ind, c1_ind, c2_ind = 0.65, 2.0, 0.3
# Fase 2: más social
w_soc, c1_soc, c2_soc = 0.55, 0.2, 2.2

# Visual
mostrar_grafica = True
tamanio_fig = (7, 6)
punto_tamanio = 60

# Semilla
semilla = 42

def cargar_matriz(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encuentro el archivo: {ruta}")
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".npy":
        M = np.load(ruta)
    elif ext == ".csv":
        M = pd.read_csv(ruta, header=None).values
    else:
        raise ValueError("Usa .npy o .csv")
    if M.ndim != 2:
        raise ValueError("La matriz debe ser 2D")
    return M.astype(float)

def recortar_redondear(pos, n_filas, n_cols):
    p = pos.copy()
    p[0::2] = np.clip(np.round(p[0::2]), 0, n_cols - 1)  # x/col
    p[1::2] = np.clip(np.round(p[1::2]), 0, n_filas - 1) # y/fila
    return p.astype(int)

def pares_idx(pos_enteras):
    return np.column_stack([pos_enteras[0::2], pos_enteras[1::2]])  # [[col,fila], ...]

def penalizacion_distancias(indices, dmin):
    p = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            dx = float(indices[i,0] - indices[j,0])
            dy = float(indices[i,1] - indices[j,1])
            d = (dx*dx + dy*dy) ** 0.5
            if d < dmin:
                p += (dmin - d) / max(dmin, 1e-9)
    return p

def resolver_duplicados(indices, M):
    ocupadas = set()
    for k in range(indices.shape[0]):
        col, fil = int(indices[k,0]), int(indices[k,1])
        if (col, fil) not in ocupadas:
            ocupadas.add((col, fil))
            continue
        mejor = (col, fil)
        mejor_val = -np.inf
        for radio in (1, 2):
            for dc in range(-radio, radio + 1):
                for df in range(-radio, radio + 1):
                    if dc == 0 and df == 0:
                        continue
                    c2, f2 = col + dc, fil + df
                    if c2 < 0 or f2 < 0 or f2 >= M.shape[0] or c2 >= M.shape[1]:
                        continue
                    if (c2, f2) in ocupadas:
                        continue
                    v = M[f2, c2]
                    if np.isnan(v):
                        continue
                    if v > mejor_val:
                        mejor_val = v
                        mejor = (c2, f2)
        indices[k,0], indices[k,1] = mejor
        ocupadas.add(mejor)
    return indices

class ProblemaSensores:
    # Sumo idoneidad y aplico F4 (distancia mínima)
    def __init__(self, M, n_sens, dmin, peso_dist, penal_nan):
        self.M = M
        self.n_sens = n_sens
        self.n_filas, self.n_cols = M.shape
        self.dmin = dmin
        self.peso_dist = peso_dist
        self.penal_nan = penal_nan

    def costo_lote(self, X):
        costos = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            pos = recortar_redondear(X[i,:], self.n_filas, self.n_cols)
            pares = pares_idx(pos)
            suma_idon = 0.0
            penal_nan_total = 0.0
            for (c, f) in pares:
                v = self.M[f, c]
                if np.isnan(v):
                    penal_nan_total += self.penal_nan
                else:
                    suma_idon += float(v)
            pen_dist = penalizacion_distancias(pares, self.dmin) * self.peso_dist
            costos[i] = -(suma_idon) + pen_dist + penal_nan_total
        return costos

def main():
    np.random.seed(semilla)

    metodo = "pso_dos_fases"  # <<<<<< nombre del método para imprimir/CSV

    M = cargar_matriz(ruta_matriz_idoneidad)
    n_filas, n_cols = M.shape
    dim = 2 * n_sensores

    # Cada partícula propone (col,fila) para cada sensor
    low = np.zeros(dim, dtype=float)
    high = np.zeros(dim, dtype=float)
    low[0::2] = 0.0
    high[0::2] = n_cols - 1.0  # x/col
    low[1::2] = 0.0
    high[1::2] = n_filas - 1.0 # y/fila
    bounds = (low, high)

    problema = ProblemaSensores(M, n_sensores, dist_minima_celdas,
                                peso_penalizacion_dist, penalizacion_nan)

    from pyswarms.single.global_best import GlobalBestPSO

    # Iteraciones por fase
    it_ind = int(round(iteraciones_totales * porc_fase_independiente))
    it_soc = max(iteraciones_totales - it_ind, 0)

    # Fase 1
    opt = GlobalBestPSO(n_particles=n_particulas, dimensions=dim,
                        options={"c1": c1_ind, "c2": c2_ind, "w": w_ind},
                        bounds=bounds)
    print("\n=== Fase 1: Independiente ===")
    mejor_costo_1, mejor_pos_1 = opt.optimize(problema.costo_lote, iters=it_ind, verbose=True)

    # Fase 2 (continúo desde estado anterior)
    opt.options = {"c1": c1_soc, "c2": c2_soc, "w": w_soc}
    print("\n=== Fase 2: Social ===")
    mejor_costo_2, mejor_pos_2 = opt.optimize(problema.costo_lote, iters=it_soc, verbose=True)

    # Resultado final
    mejor_costo = float(mejor_costo_2 if it_soc > 0 else mejor_costo_1)
    mejor_pos = mejor_pos_2 if it_soc > 0 else mejor_pos_1

    # A celdas enteras y evito duplicados
    mejor_entera = recortar_redondear(mejor_pos, n_filas, n_cols)
    indices = pares_idx(mejor_entera)
    indices = resolver_duplicados(indices, M)

    # Métricas
    idon = []
    penal_nan_total = 0.0
    for (c, f) in indices:
        v = M[f, c]
        if np.isnan(v):
            penal_nan_total += penalizacion_nan
            idon.append(np.nan)
        else:
            idon.append(float(v))
    suma_idon = float(np.nansum(idon))
    pen_dist = penalizacion_distancias(indices, dist_minima_celdas) * peso_penalizacion_dist
    # Costo con la misma definición que el objetivo:
    costo_total = -(suma_idon) + pen_dist + penal_nan_total

    # ---- Salida estilo "comparativas" ----
    print("\n===== Resultado (comparativa) =====")
    print(f"Método: {metodo}")
    print(f"Iteraciones: indep={it_ind}, social={it_soc}, total={it_ind+it_soc}")
    print(f"Suma de idoneidad: {suma_idon:.6f}")
    print(f"Costo: {costo_total:.6f}")  # <<<<<< imprime costo total
    print("Sensores (col=x, fila=y):")
    for i, (c, f) in enumerate(indices, 1):
        print(f"  {i:02d}: col={c}, fila={f}, idoneidad={idon[i-1]}")

    if ruta_csv_salida:
        df_out = pd.DataFrame({
            "metodo": metodo,                              # <<<<<< columna método
            "sensor": np.arange(1, len(indices) + 1, dtype=int),
            "col": indices[:,0].astype(int),
            "fila": indices[:,1].astype(int),
            "idoneidad": idon
        })
        # Para tener el costo visible del mismo modo que en consola,
        # lo agrego como columna constante (una copia por fila).
        df_out["costo_total"] = costo_total             # <<<<<< costo por método
        df_out.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
        print(f"\nCSV: {ruta_csv_salida}")

    if mostrar_grafica:
        plt.figure(figsize=tamanio_fig)
        extent = [0, n_cols - 1, 0, n_filas - 1]
        plt.imshow(M, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Idoneidad")
        xs = indices[:,0].astype(float)
        ys = indices[:,1].astype(float)
        plt.scatter(xs, ys, s=punto_tamanio, edgecolor="k", facecolor="none", label="Sensores (PSO)")
        for i, (x, y) in enumerate(zip(xs, ys), 1):
            plt.text(x + 0.5, y + 0.5, str(i), fontsize=9, color="k")
        plt.title(
            f"{metodo} — {len(indices)} sensores | suma={suma_idon:.3f} | costo={costo_total:.3f}"
        )

        plt.xlabel("Columnas (x)")
        plt.ylabel("Filas (y)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
