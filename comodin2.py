# pso_polen_pyswarms.py
# PSO con PySwarms para ubicar 10 "sensores" maximizando el polen cubierto en un grid 2D.
# - Campo = matriz 2D "pollen" con calidad por celda.
# - Cada sensor cubre su celda y las 8 vecinas (bloque 3×3). No se cuenta doble.
# - Usamos pyswarms.single.GlobalBestPSO (maximizamos devolviendo -cost para minimizar).

import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

# -----------------------------
# 1) Crear un mapa simple de "polen"
# -----------------------------
np.random.seed(7)
H, W = 60, 60          # tamaño del campo (alto, ancho)
K = 10                 # número de sensores a colocar
num_bumps = 3          # "manchas" de polen

y, x = np.mgrid[0:H, 0:W]
pollen = np.zeros((H, W), dtype=float)

centers = np.random.rand(num_bumps, 2) * np.array([H, W])
sigmas = np.random.uniform(6, 12, size=num_bumps)
amps   = np.random.uniform(0.6, 1.0, size=num_bumps)

for (cy, cx), s, a in zip(centers, sigmas, amps):
    pollen += a * np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * s ** 2)))

pollen -= pollen.min()
pollen /= (pollen.max() + 1e-9)
pollen = 0.9 * pollen + 0.1 * np.random.rand(H, W)

# -----------------------------
# 2) Fitness: cobertura 3x3 sin doble conteo
#    PySwarms minimiza, así que devolvemos el negativo para maximizar.
# -----------------------------
def fitness_single(position_flat: np.ndarray) -> float:
    """Evalúa una sola solución (vector 2*K con x1,y1,...xK,yK).
       'Snap' a celdas, suma polen cubierto en 3×3, penaliza sensores en misma celda."""
    pts = position_flat.reshape(K, 2).copy()
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
    unique_cells = set((int(x), int(y)) for x, y in pts)
    duplicates = K - len(unique_cells)
    penalty = 0.25 * duplicates
    return total_pollen - penalty

def fitness_batch(X: np.ndarray) -> np.ndarray:
    """PySwarms envía un batch: shape (n_particles, 2*K).
       Debe devolver costos a minimizar → usamos -fitness."""
    vals = np.array([fitness_single(x) for x in X], dtype=float)
    return -vals  # minimizar el negativo = maximizar fitness

# -----------------------------
# 3) Configurar y correr PySwarms
# -----------------------------
D = 2 * K
# límites del espacio continuo (el fitness luego redondea a celdas)
lower_bounds = np.array([0, 0] * K, dtype=float)
upper_bounds = np.array([W - 1, H - 1] * K, dtype=float)
bounds = (lower_bounds, upper_bounds)

options = {
    "c1": 1.5,     # componente cognitiva
    "c2": 1.5,     # componente social
    "w": 0.72,     # inercia
    "k": None,     # usar topología global por defecto
    "p": 2,
}

optimizer = ps.single.GlobalBestPSO(
    n_particles=30,
    dimensions=D,
    options=options,
    bounds=bounds,
    velocity_clamp=(-2.0, 2.0),
    init_pos=None,
)

cost, pos = optimizer.optimize(fitness_batch, iters=60, verbose=True)
# cost es el mínimo de -fitness → el fitness máximo es -cost
best_fitness = -float(cost)
best_pos = pos.copy()

# -----------------------------
# 4) Visualizar resultado
# -----------------------------
best_pts = best_pos.reshape(K, 2).copy()
best_pts[:, 0] = np.clip(np.rint(best_pts[:, 0]), 0, W - 1)
best_pts[:, 1] = np.clip(np.rint(best_pts[:, 1]), 0, H - 1)
best_pts = best_pts.astype(int)

plt.figure(figsize=(6, 6))
plt.imshow(pollen, origin="lower")
plt.scatter(best_pts[:, 0], best_pts[:, 1], marker="x", s=80)
plt.title(f"PySwarms: 10 sensores óptimos (fitness={best_fitness:.2f})")
plt.xlabel("x (celdas)")
plt.ylabel("y (celdas)")
plt.tight_layout()
plt.show()

print("Mejor fitness (max):", best_fitness)
print("Coordenadas de sensores (x, y):\n", best_pts)
