# pso_polen_simple.py
# PSO mínimo para ubicar 10 "abejas" (sensores) y maximizar el polen cubierto en un grid 2D.
# - Campo = matriz 2D "pollen" con calidad por celda.
# - Cada sensor cubre su celda y las 8 vecinas (bloque 3×3). No se cuenta doble.
# - PSO global-best con evaluación discreta (redondeo a celdas).

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Crear un mapa simple de "polen"
# -----------------------------
np.random.seed(7)
H, W = 60, 60          # tamaño del campo (alto, ancho)
K = 10                 # número de sensores a colocar
num_bumps = 3          # "manchas" de polen

# Generar un mapa con 3 gaussianas aleatorias (zonas ricas en polen)
y, x = np.mgrid[0:H, 0:W]
pollen = np.zeros((H, W), dtype=float)

centers = np.random.rand(num_bumps, 2) * np.array([H, W])
sigmas = np.random.uniform(6, 12, size=num_bumps)   # dispersión de cada mancha
amps   = np.random.uniform(0.6, 1.0, size=num_bumps)  # amplitud de cada mancha

for (cy, cx), s, a in zip(centers, sigmas, amps):
    pollen += a * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2)))

# Normalizar a [0,1] y añadir ruido suave
pollen -= pollen.min()
pollen /= (pollen.max() + 1e-9)
pollen = 0.9 * pollen + 0.1 * np.random.rand(H, W)

# -----------------------------
# 2) Fitness: cobertura 3x3 sin doble conteo
# -----------------------------
def fitness(position_flat):
    """
    position_flat: array shape (2*K,) con posiciones (x1, y1, ..., xK, yK) en continuo.
    La evaluación hace 'snap' a la celda más cercana y suma el polen cubierto por bloques 3×3.
    Penaliza sensores superpuestos (misma celda).
    """
    pts = position_flat.reshape(K, 2).copy()
    # redondear a celdas válidas
    pts[:, 0] = np.clip(np.rint(pts[:, 0]), 0, W - 1)  # x
    pts[:, 1] = np.clip(np.rint(pts[:, 1]), 0, H - 1)  # y
    pts = pts.astype(int)

    covered = np.zeros((H, W), dtype=bool)
    for x0, y0 in pts:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = x0 + dx, y0 + dy
                if 0 <= xx < W and 0 <= yy < H:
                    covered[yy, xx] = True

    total_pollen = float(pollen[covered].sum())

    # Penalizar sensores en la misma celda (superposición extrema)
    unique_cells = set((int(x), int(y)) for x, y in pts)
    duplicates = K - len(unique_cells)
    penalty = 0.25 * duplicates  # penalización simple

    return total_pollen - penalty

# -----------------------------
# 3) PSO minimalista (global-best)
# -----------------------------
def pso_optimize(num_particles=30, iters=60, w=0.72, c1=1.5, c2=1.5, v_clamp=2.0):
    D = 2 * K  # dimensión del vector de decisión
    # Inicialización uniforme en el campo
    X = np.empty((num_particles, D), dtype=float)
    for p in range(num_particles):
        xs = np.random.uniform(0, W - 1, size=K)
        ys = np.random.uniform(0, H - 1, size=K)
        X[p] = np.column_stack([xs, ys]).reshape(-1)

    V = np.zeros_like(X)
    pbest_pos = X.copy()
    pbest_val = np.array([fitness(x) for x in X], dtype=float)
    g_idx = int(np.argmax(pbest_val))
    gbest_pos = pbest_pos[g_idx].copy()
    gbest_val = float(pbest_val[g_idx])

    history = [gbest_val]

    for _ in range(iters):
        for p in range(num_particles):
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            V[p] = (w * V[p]
                    + c1 * r1 * (pbest_pos[p] - X[p])
                    + c2 * r2 * (gbest_pos - X[p]))
            V[p] = np.clip(V[p], -v_clamp, v_clamp)
            X[p] = X[p] + V[p]

            # límites del campo continuo (el fitness hará el redondeo a celdas)
            X[p, 0::2] = np.clip(X[p, 0::2], 0, W - 1)  # x
            X[p, 1::2] = np.clip(X[p, 1::2], 0, H - 1)  # y

            f = fitness(X[p])
            if f > pbest_val[p]:
                pbest_val[p] = f
                pbest_pos[p] = X[p].copy()

        g_idx = int(np.argmax(pbest_val))
        if pbest_val[g_idx] > gbest_val:
            gbest_val = float(pbest_val[g_idx])
            gbest_pos = pbest_pos[g_idx].copy()

        history.append(gbest_val)

    return gbest_pos, gbest_val, np.array(history)

# -----------------------------
# 4) Ejecutar y visualizar
# -----------------------------
if __name__ == "__main__":
    best_pos, best_val, hist = pso_optimize()

    # Discretizar para dibujar
    best_pts = best_pos.reshape(K, 2).copy()
    best_pts[:, 0] = np.clip(np.rint(best_pts[:, 0]), 0, W - 1)
    best_pts[:, 1] = np.clip(np.rint(best_pts[:, 1]), 0, H - 1)
    best_pts = best_pts.astype(int)

    # Gráfico
    plt.figure(figsize=(6, 6))
    plt.imshow(pollen, origin="lower")
    plt.scatter(best_pts[:, 0], best_pts[:, 1], marker="x", s=80)
    plt.title(f"Mapa de polen y 10 sensores óptimos (fitness={best_val:.2f})")
    plt.xlabel("x (celdas)")
    plt.ylabel("y (celdas)")
    plt.tight_layout()
    plt.show()

    # Consola
    print("Fitness inicial → final:", hist[0], "→", hist[-1])
    print("Mejor fitness:", best_val)
    print("Coordenadas de sensores (x, y):")
    print(best_pts)
