#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# comparativas.py
# Comparo dos formas de ubicar sensores usando la misma matriz de idoneidad:
# 1) Aleatorio
# 2) Patrón tipo malla (grid)
# Reporta la suma de idoneidad y costo (con F4: distancia mínima + NaN)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Entradas / salidas
ruta_matriz_idoneidad = "idoneidad_total.npy"  # también acepta .csv
ruta_csv_salida = "comparativas.csv"

# Configuración
n_sensores = 20
dist_minima_celdas = 5        # F4: distancia mínima en celdas
peso_penalizacion_dist = 2.0  # peso de la penalización por distancia
penalizacion_nan = 10.0       # penalizo caer en NaN

# Visual
mostrar_graficas = True
tamanio_fig = (7, 6)
punto = 60

# Semilla
semilla = 123

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
    # muevo duplicados a una vecina con buen valor (rápido y simple)
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

def evaluar(M, indices):
    # suma de idoneidad y costo con F4 + NaN
    vals = []
    penal_nan_total = 0.0
    for (c, f) in indices:
        v = M[f, c]
        if np.isnan(v):
            penal_nan_total += penalizacion_nan
            vals.append(np.nan)
        else:
            vals.append(float(v))
    suma_idon = float(np.nansum(vals))
    pen_dist = penalizacion_distancias(indices, dist_minima_celdas) * peso_penalizacion_dist
    costo = -(suma_idon) + pen_dist + penal_nan_total
    return suma_idon, costo, vals

def ubicacion_aleatoria(n_filas, n_cols, n_sens, rng):
    # col,fila uniformes
    cols = rng.integers(0, n_cols, size=n_sens, endpoint=False)
    fils = rng.integers(0, n_filas, size=n_sens, endpoint=False)
    return np.column_stack([cols, fils]).astype(int)

def ubicacion_malla(n_filas, n_cols, n_sens):
    # reparto en una malla ~cuadrada y tomo los primeros n_sens
    lado = int(np.ceil(np.sqrt(n_sens)))
    xs = np.linspace(0, n_cols - 1, num=lado)
    ys = np.linspace(0, n_filas - 1, num=lado)
    XI, YI = np.meshgrid(xs, ys)
    pares = np.column_stack([XI.ravel(), YI.ravel()]).astype(int)
    pares = pares[:n_sens]
    # por si cae fuera por redondeo
    pares[:,0] = np.clip(pares[:,0], 0, n_cols - 1)
    pares[:,1] = np.clip(pares[:,1], 0, n_filas - 1)
    return pares

def plot_ubicanos(M, indices, titulo):
    plt.figure(figsize=tamanio_fig)
    extent = [0, M.shape[1] - 1, 0, M.shape[0] - 1]
    plt.imshow(M, origin="lower", extent=extent, aspect="auto")
    plt.colorbar(label="Idoneidad")
    xs = indices[:,0].astype(float)
    ys = indices[:,1].astype(float)
    plt.scatter(xs, ys, s=punto, edgecolor="k", facecolor="none", label="Sensores")
    for i, (x, y) in enumerate(zip(xs, ys), 1):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=9, color="k")
    plt.title(titulo)
    plt.xlabel("Columnas (x)")
    plt.ylabel("Filas (y)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    np.random.seed(semilla)
    rng = np.random.default_rng(semilla)

    M = cargar_matriz(ruta_matriz_idoneidad)
    n_filas, n_cols = M.shape

    # ALEATORIO
    idx_rand = ubicacion_aleatoria(n_filas, n_cols, n_sensores, rng)
    idx_rand = resolver_duplicados(idx_rand, M)
    suma_rand, costo_rand, vals_rand = evaluar(M, idx_rand)

    # MALLA
    idx_grid = ubicacion_malla(n_filas, n_cols, n_sensores)
    idx_grid = resolver_duplicados(idx_grid, M)
    suma_grid, costo_grid, vals_grid = evaluar(M, idx_grid)

    # Reporte
    print("\n===== Comparativas =====")
    print(f"Sensores: {n_sensores}")
    print(f"Aleatorio -> Suma idoneidad: {suma_rand:.6f} | Costo: {costo_rand:.6f}")
    print(f"Malla     -> Suma idoneidad: {suma_grid:.6f} | Costo: {costo_grid:.6f}")
    print("\nNota: la 'suma de idoneidad' es sin penalizaciones; el 'costo' sí incluye F4 y NaN.")

    # CSV
    if ruta_csv_salida:
        df_a = pd.DataFrame({
            "metodo": "aleatorio",
            "sensor": np.arange(1, len(idx_rand) + 1, dtype=int),
            "col": idx_rand[:,0].astype(int),
            "fila": idx_rand[:,1].astype(int),
            "idoneidad": vals_rand
        })
        df_g = pd.DataFrame({
            "metodo": "malla",
            "sensor": np.arange(1, len(idx_grid) + 1, dtype=int),
            "col": idx_grid[:,0].astype(int),
            "fila": idx_grid[:,1].astype(int),
            "idoneidad": vals_grid
        })
        df_out = pd.concat([df_a, df_g], ignore_index=True)
        df_out.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
        print(f"\nCSV: {ruta_csv_salida}")

    # Plots
    if mostrar_graficas:
        plot_ubicanos(M, idx_rand, f"Aleatorio — suma={suma_rand:.3f} | costo={costo_rand:.3f}")
        plot_ubicanos(M, idx_grid, f"Malla — suma={suma_grid:.3f} | costo={costo_grid:.3f}")

if __name__ == "__main__":
    main()
