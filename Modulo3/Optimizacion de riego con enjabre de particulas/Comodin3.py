import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Parámetros AJUSTABLES
"""
excel_path = "Datos/data.xlsx"
sheet_name = "Hoja1"

n_grid = 220
margin = 0.005
eps = 1e-12
"""
Vecindad
"""
kernel_type = "idw"              # "idw" o "box"
idw_power = 4.2                  # 3.5–5.0 => menos suavizado
max_neighbors = 8                # 6–12 => más local si es menor
box_halfsize_deg = 0.0015        # tamaño de parcela para 'box' (grados)

# Pesos ( 3 factores)
W1, W2, W3 = 0.50, 0.30, 0.20    # F1 + F2 - F3

# Carga y columnas fijas
df = pd.read_excel(excel_path, sheet_name=sheet_name)

# Renombre columnas para comodidad
df = df.rename(columns={
    "Humedad (%)": "humedad",
    "Cultivo": "cultivo",
    "Elevación (m)": "elevacion",
    "Salinidad (dS/m)": "salinidad",
    "Temperatura (°C)": "temperatura",
    "Latitud": "lat",
    "Longitud": "lon",
})

# Grid
min_lat, max_lat = df["lat"].min() - margin, df["lat"].max() + margin
min_lon, max_lon = df["lon"].min() - margin, df["lon"].max() + margin

grid_lats = np.linspace(min_lat, max_lat, n_grid)
grid_lons = np.linspace(min_lon, max_lon, n_grid)
XI, YI = np.meshgrid(grid_lons, grid_lats)  # XI: lon, YI: lat

xs = df["lon"].values
ys = df["lat"].values

# Utilidades compactas
def topk_weights(w, k):
    if k is None or k <= 0 or k >= w.size:
        return w
    idx = np.argpartition(w, -k)[-k:]
    w2 = np.zeros_like(w)
    w2[idx] = w[idx]
    return w2

def local_weights(x0, y0, xs, ys, kernel="idw", power=2.0, k=None, box_h=0.002, eps=1e-12):
    if kernel == "box":
        mask = (np.abs(x0 - xs) <= box_h) & (np.abs(y0 - ys) <= box_h)
        w = np.where(mask, 1.0, 0.0).astype(float)
        if w.sum() == 0.0:  # fallback
            dist2 = (x0 - xs)**2 + (y0 - ys)**2 + eps
            w = 1.0 / (dist2 ** (power / 2.0))
            if k is not None and 0 < k < w.size:
                w = topk_weights(w, k)
        return w
    else:
        dist2 = (x0 - xs)**2 + (y0 - ys)**2 + eps
        w = 1.0 / (dist2 ** (power / 2.0))
        if k is not None and 0 < k < w.size:
            w = topk_weights(w, k)
        return w

def idw_interpolate(xs, ys, zs, XI, YI, power=2.0, k=None, eps=1e-12, chunk=6000):
    xyi = np.column_stack([XI.ravel(), YI.ravel()])
    out = np.full(xyi.shape[0], np.nan, dtype=float)
    pts = np.column_stack([xs, ys])
    zs = np.asarray(zs, dtype=float)

    for start in range(0, xyi.shape[0], chunk):
        end = min(start + chunk, xyi.shape[0])
        block = xyi[start:end]
        dx = block[:, [0]] - pts[:, 0]
        dy = block[:, [1]] - pts[:, 1]
        dist2 = dx*dx + dy*dy + eps
        w = 1.0 / (dist2 ** (power / 2.0))
        if k is not None and k > 0 and k < w.shape[1]:
            for r in range(w.shape[0]):
                w[r, :] = topk_weights(w[r, :], k)

        num = (w * zs).sum(axis=1)
        den = w.sum(axis=1)
        block_vals = num / den

        near_mask = (dist2 <= (eps * 2)).any(axis=1)
        if near_mask.any():
            idxs = np.where(near_mask)[0]
            for bi in idxs:
                j = np.argmin(dist2[bi])
                block_vals[bi] = zs[j]

        out[start:end] = block_vals

    return out.reshape(XI.shape)

def normalized_entropy(weights_by_class):
    w = np.array(weights_by_class, dtype=float)
    tot = w.sum()
    if tot <= 0: return 0.0
    p = w / tot
    p = p[p > 0]
    if p.size == 0: return 0.0
    H = -(p * np.log(p)).sum()
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0

def safe_range(a):
    a = np.asarray(a, dtype=float)
    r = np.nanmax(a) - np.nanmin(a)
    return r if r > 0 else 1.0

def local_microbiome_homogeneity(weights, v_elev, v_sal, v_temp, rng_e, rng_s, rng_t):
    w = np.asarray(weights, dtype=float)
    wn = w / w.sum() if w.sum() > 0 else w
    if wn.sum() == 0: return 0.0
    def wstd(x, wn):
        m = (wn * x).sum(); var = (wn * (x - m)**2).sum(); return np.sqrt(var)
    s_elev = wstd(v_elev, wn) / rng_e
    s_sal  = wstd(v_sal,  wn) / rng_s
    s_temp = wstd(v_temp, wn) / rng_t
    return float(np.clip(1.0 - np.mean([s_elev, s_sal, s_temp]), 0.0, 1.0))

def plot_heatmap(Z, XI, YI, title, points_df):
    plt.figure(figsize=(7, 6))
    extent = [XI.min(), XI.max(), YI.min(), YI.max()]
    plt.imshow(Z, origin="lower", extent=extent, aspect="auto")
    plt.scatter(points_df["lon"], points_df["lat"], s=10, alpha=0.6)
    plt.xlabel("Longitud"); plt.ylabel("Latitud")
    plt.title(title); plt.colorbar(label=title)
    plt.tight_layout(); plt.show()


# Interpolaciones (3 mapas)
ZI_elev = idw_interpolate(xs, ys, df["elevacion"].values, XI, YI,
                        power=idw_power, k=max_neighbors, eps=eps)
ZI_sal  = idw_interpolate(xs, ys, df["salinidad"].values,  XI, YI,
                        power=idw_power, k=max_neighbors, eps=eps)
ZI_temp = idw_interpolate(xs, ys, df["temperatura"].values, XI, YI,
                        power=idw_power, k=max_neighbors, eps=eps)

# Factores
cultivos = df["cultivo"].astype(str).values
cultivos_unique, cult_idx = np.unique(cultivos, return_inverse=True)

vals_elev = df["elevacion"].values
vals_sal  = df["salinidad"].values
vals_temp = df["temperatura"].values

range_elev = safe_range(vals_elev)
range_sal  = safe_range(vals_sal)
range_temp = safe_range(vals_temp)

F1 = np.zeros_like(XI, dtype=float)  # premio: mismo cultivo
F2 = np.zeros_like(XI, dtype=float)  # premio: homogeneidad
F3 = np.zeros_like(XI, dtype=float)  # penalización: multicultivo

for i in range(XI.shape[0]):
    for j in range(XI.shape[1]):
        x0, y0 = XI[i, j], YI[i, j]
        w = local_weights(x0, y0, xs, ys,
                        kernel=kernel_type, power=idw_power,
                        k=max_neighbors, box_h=box_halfsize_deg, eps=eps)

        by_class = np.zeros(len(cultivos_unique), dtype=float)
        np.add.at(by_class, cult_idx, w)
        tot_w = by_class.sum()

        F1[i, j] = 0.0 if tot_w <= 0 else float(by_class.max() / tot_w)
        F3[i, j] = normalized_entropy(by_class)
        F2[i, j] = local_microbiome_homogeneity(
            w, vals_elev, vals_sal, vals_temp,
            range_elev, range_sal, range_temp
        )

# Mapa maestro (0..1)
S = np.clip(W1 * F1 + W2 * F2 - W3 * F3, 0.0, 1.0)

# Plots (4 mapas)
plot_heatmap(ZI_elev, XI, YI, f"Elevación (IDW p={idw_power}, k={max_neighbors})", df)
plot_heatmap(ZI_sal,  XI, YI, f"Salinidad (IDW p={idw_power}, k={max_neighbors})", df)
plot_heatmap(ZI_temp, XI, YI, f"Temperatura (IDW p={idw_power}, k={max_neighbors})", df)
plot_heatmap(S,       XI, YI, f"Idoneidad total (kernel={kernel_type})", df)
