#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pso.py — PSO en dos fases (independiente y social) para ubicar sensores sobre una matriz de idoneidad.

Requisitos:
    pip install numpy matplotlib pyswarms pandas

Idea:
    - Fase 1 (independiente): c1 > c2  -> exploración individual (memoria propia)
    - Fase 2 (social):       c2 > c1  -> explotación colectiva (mejor global)

Ambas fases suman el total de iteraciones. Se continúa desde el estado de la fase anterior.
"""

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

# =========================
# Parámetros configurables
# =========================

# Rutas de entrada/salida
ruta_matriz_idoneidad = "idoneidad_total.npy"  # acepta .npy o .csv
ruta_XI_opcional = None                        # ejemplo: "XI.npy" (longitudes por celda)
ruta_YI_opcional = None                        # ejemplo: "YI.npy" (latitudes por celda)
ruta_csv_salida = "sensores_pso.csv"           # None para no guardar CSV

# Hiperparámetros del problema
n_sensores = 20              # cantidad de sensores a ubicar
dist_minima_celdas = 5       # distancia mínima entre sensores (en celdas)
peso_penalizacion_dist = 2.0 # penalización por violar la distancia mínima
penalizacion_nan = 10.0      # penalización por caer en celda NaN

# Hiperparámetros del PSO (globales)
n_particulas = 140
iteraciones_totales = 200

# % de iteraciones por fase (deben sumar 1.0) — 25% indep, 75% social
porc_fase_independiente = 0.25
porc_fase_social = 0.75

# Fase 1: inclinada a exploración individual (c1 >> c2, ambos > 0)
inercia_indep   = 0.65  # w
cognitivo_indep = 2.0   # c1 alto
social_indep    = 0.3   # c2 pequeño (NO cero)

# Fase 2: inclinada a explotación social (c2 >> c1, ambos > 0)
inercia_social   = 0.55 # w
cognitivo_social = 0.2  # c1 pequeño (NO cero)
social_social    = 2.2  # c2 alto

# Control de velocidad: fracción del rango por dimensión (None para dejar por defecto)
fraccion_velocidad_max = 0.25

# Reproducibilidad
semilla = 42  # usamos np.random.seed(semilla)

# Visualización
mostrar_grafica = True
tamanio_fig = (7, 6)
punto_tamanio = 60

# =========================
# Carga de datos
# =========================

def cargar_matriz_idoneidad(ruta: str) -> np.ndarray:
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    ext = os.path.splitext(ruta)[1].lower()
    if ext == ".npy":
        M = np.load(ruta)
    elif ext == ".csv":
        M = pd.read_csv(ruta, header=None).values
    else:
        raise ValueError("Formato no soportado. Usa .npy o .csv")
    if M.ndim != 2:
        raise ValueError("La matriz de idoneidad debe ser 2D")
    return M.astype(float)

def cargar_malla_opcional(ruta: Optional[str]) -> Optional[np.ndarray]:
    if ruta is None:
        return None
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo de malla: {ruta}")
    return np.load(ruta)

# =========================
# Utilidades del problema
# =========================

def recortar_roundear_posiciones(pos: np.ndarray, n_filas: int, n_cols: int) -> np.ndarray:
    pos = pos.copy()
    pos[0::2] = np.clip(np.round(pos[0::2]), 0, n_cols - 1)  # x/col
    pos[1::2] = np.clip(np.round(pos[1::2]), 0, n_filas - 1) # y/fila
    return pos.astype(int)

def pares_idx(pos_enteras: np.ndarray) -> np.ndarray:
    return np.column_stack([pos_enteras[0::2], pos_enteras[1::2]])

def distancia_euclidea_int(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def penalizacion_distancias(indices: np.ndarray, dmin: float) -> float:
    p = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            d = distancia_euclidea_int(indices[i], indices[j])
            if d < dmin:
                p += (dmin - d) / max(dmin, 1e-9)
    return p

def resolver_conflictos_por_duplicado(indices: np.ndarray, M: np.ndarray) -> np.ndarray:
    ocupadas = set()
    for k in range(indices.shape[0]):
        col, fil = int(indices[k, 0]), int(indices[k, 1])
        if (col, fil) not in ocupadas:
            ocupadas.add((col, fil))
            continue
        mejor = (col, fil)
        mejor_val = -np.inf
        encontrado = False
        for radio in (1, 2, 3):
            for dc in range(-radio, radio + 1):
                for df in range(-radio, radio + 1):
                    if dc == 0 and df == 0:
                        continue
                    c2 = col + dc
                    f2 = fil + df
                    if c2 < 0 or f2 < 0 or f2 >= M.shape[0] or c2 >= M.shape[1]:
                        continue
                    if (c2, f2) in ocupadas:
                        continue
                    val = M[f2, c2]
                    if np.isnan(val):
                        continue
                    if val > mejor_val:
                        mejor_val = val
                        mejor = (c2, f2)
                        encontrado = True
            if encontrado:
                break
        indices[k, 0], indices[k, 1] = mejor
        ocupadas.add(mejor)
    return indices

# =========================
# Función objetivo
# =========================

class ProblemaSensores:
    def __init__(self, M: np.ndarray, n_sensores: int,
                 dmin: float, peso_penal_dist: float, penal_nan: float):
        self.M = M
        self.n_sensores = n_sensores
        self.n_filas, self.n_cols = M.shape
        self.dmin = dmin
        self.peso_penal_dist = peso_penal_dist
        self.penal_nan = penal_nan

    def costo_lote(self, X: np.ndarray) -> np.ndarray:
        costos = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            pos = recortar_roundear_posiciones(X[i, :], self.n_filas, self.n_cols)
            pares = pares_idx(pos)  # [[col,fila], ...]
            suma_idon = 0.0
            penal_nan_total = 0.0
            for (c, f) in pares:
                val = self.M[f, c]
                if np.isnan(val):
                    penal_nan_total += self.penal_nan
                else:
                    suma_idon += float(val)
            pen_dist = penalizacion_distancias(pares, self.dmin) * self.peso_penal_dist
            costos[i] = -(suma_idon) + pen_dist + penal_nan_total
        return costos

# =========================
# Ejecución con dos fases
# =========================

def main():
    # Reproducibilidad
    np.random.seed(semilla)

    # Cargar insumos
    M = cargar_matriz_idoneidad(ruta_matriz_idoneidad)
    XI = cargar_malla_opcional(ruta_XI_opcional)
    YI = cargar_malla_opcional(ruta_YI_opcional)

    n_filas, n_cols = M.shape
    dim = 2 * n_sensores

    # Límites por dimensión
    low = np.zeros(dim, dtype=float)
    high = np.zeros(dim, dtype=float)
    low[0::2] = 0.0
    high[0::2] = n_cols - 1.0  # x/col
    low[1::2] = 0.0
    high[1::2] = n_filas - 1.0 # y/fila
    bounds = (low, high)

    # Velocidad máxima (opcional)
    if fraccion_velocidad_max is not None:
        rango_max = np.max(high - low)
        vmax = fraccion_velocidad_max * rango_max
        v_clamp = (-vmax, vmax)
    else:
        v_clamp = None

    problema = ProblemaSensores(
        M=M,
        n_sensores=n_sensores,
        dmin=dist_minima_celdas,
        peso_penal_dist=peso_penalizacion_dist,
        penal_nan=penalizacion_nan
    )

    from pyswarms.single.global_best import GlobalBestPSO

    # Inicializamos con la FASE INDEPENDIENTE (c1>c2, ambos > 0)
    opciones = {"c1": cognitivo_indep, "c2": social_indep, "w": inercia_indep}
    optimizador = GlobalBestPSO(
        n_particles=n_particulas,
        dimensions=dim,
        options=opciones,
        bounds=bounds,
        velocity_clamp=v_clamp,
        ftol=1e-6,
        ftol_iter=25,
        bh_strategy="periodic",  # rebote periódico en límites
        init_pos=None
    )

    # Partición de iteraciones
    it_indep = int(round(iteraciones_totales * porc_fase_independiente))
    it_social = max(iteraciones_totales - it_indep, 0)

    # -------- Fase 1: Independiente --------
    print("\n===== Fase 1: Independiente (cognitivo>social) =====")
    mejor_costo_1, mejor_pos_1 = optimizador.optimize(
        problema.costo_lote,
        iters=it_indep,
        n_processes=None,
        verbose=True
    )

    # -------- Fase 2: Social --------
    optimizador.options = {"c1": cognitivo_social, "c2": social_social, "w": inercia_social}
    print("\n===== Fase 2: Social (social>cognitivo) =====")
    mejor_costo_2, mejor_pos_2 = optimizador.optimize(
        problema.costo_lote,
        iters=it_social,
        n_processes=None,
        verbose=True
    )

    # Mejor final
    mejor_costo_final = float(mejor_costo_2 if it_social > 0 else mejor_costo_1)
    mejor_pos_final = mejor_pos_2 if it_social > 0 else mejor_pos_1

    # A índices discretos y resolver duplicados
    mejor_pos_entera = recortar_roundear_posiciones(mejor_pos_final, n_filas, n_cols)
    indices = pares_idx(mejor_pos_entera)
    indices = resolver_conflictos_por_duplicado(indices, M)

    # Métrica final
    idoneidades = []
    for (c, f) in indices:
        val = M[f, c]
        idoneidades.append(np.nan if np.isnan(val) else float(val))
    suma_idoneidad = float(np.nansum(idoneidades))

    print("\n====== Resultado PSO (dos fases) ======")
    print(f"Iteraciones: indep={it_indep}, social={it_social}, total={it_indep+it_social}")
    print(f"Mejor costo final (menor es mejor): {mejor_costo_final:.6f}")
    print(f"Suma de idoneidad (más es mejor): {suma_idoneidad:.6f}")
    print("Sensores (columna=x, fila=y):")
    for k, (c, f) in enumerate(indices, 1):
        print(f"  Sensor {k}: col={c}, fila={f}, idoneidad={idoneidades[k-1]}")

    # Guardar CSV
    if ruta_csv_salida:
        df_out = pd.DataFrame({
            "sensor": np.arange(1, len(indices) + 1, dtype=int),
            "col": indices[:, 0].astype(int),
            "fila": indices[:, 1].astype(int),
            "idoneidad": idoneidades
        })
        if XI is not None and YI is not None:
            longitudes = [float(XI[f, c]) for (c, f) in indices]
            latitudes  = [float(YI[f, c]) for (c, f) in indices]
            df_out["lon"] = longitudes
            df_out["lat"] = latitudes
        df_out.to_csv(ruta_csv_salida, index=False, encoding="utf-8")
        print(f"\nCSV guardado en: {ruta_csv_salida}")

    # Visualización
    if mostrar_grafica:
        plt.figure(figsize=tamanio_fig)
        extent = [0, n_cols - 1, 0, n_filas - 1]
        plt.imshow(M, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="Idoneidad")
        xs = indices[:, 0].astype(float)
        ys = indices[:, 1].astype(float)
        plt.scatter(xs, ys, s=punto_tamanio, edgecolor="k", facecolor="none", label="Sensores (PSO)")
        for i, (x, y) in enumerate(zip(xs, ys), 1):
            plt.text(x + 0.5, y + 0.5, str(i), fontsize=9, color="k")
        plt.title(f"Ubicación de {len(indices)} sensores — PSO en dos fases (25%/75%)")
        plt.xlabel("Columnas (x)")
        plt.ylabel("Filas (y)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
