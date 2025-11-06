import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datos de entrada
ruta_excel = "Datos/data.xlsx"
nombre_hoja = "Hoja1"

# Resolución de la malla y detalles numéricos
n_celdas = 220
margen = 0.005
epsilon = 1e-12

# IDW simple
potencia_idw = 4.0     # puedo moverlo un poco si hace falta
max_vecinos = 8        # me quedo con los vecinos más cercanos

# Pesos de la fórmula final
W_MONO = 0.70
W_UNIF = 0.30
P_MULTI = 0.30

# Radio para factores (en grados) para juntar parches cercanos
semilado_factores_deg = 0.004

# Cargo y dejo las columnas con nombres simples
df = pd.read_excel(ruta_excel, sheet_name=nombre_hoja)
df = df.rename(columns={
    "Humedad (%)": "humedad",
    "Cultivo": "cultivo",
    "Elevación (m)": "elevacion",
    "Salinidad (dS/m)": "salinidad",
    "Temperatura (°C)": "temperatura",
    "Latitud": "lat",
    "Longitud": "lon",
})

# Reviso columnas necesarias
cols_req = ["lat", "lon", "cultivo", "elevacion", "salinidad", "temperatura"]
faltantes = [c for c in cols_req if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas en el Excel: {faltantes}")

# A numérico y limpio filas con NA
for c in ["lat", "lon", "elevacion", "salinidad", "temperatura"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["lat", "lon", "elevacion", "salinidad", "temperatura", "cultivo"]).reset_index(drop=True)

# Malla de cálculo
min_lat, max_lat = df["lat"].min() - margen, df["lat"].max() + margen
min_lon, max_lon = df["lon"].min() - margen, df["lon"].max() + margen
latitudes_malla  = np.linspace(min_lat, max_lat, n_celdas)
longitudes_malla = np.linspace(min_lon, max_lon, n_celdas)
XI, YI = np.meshgrid(longitudes_malla, latitudes_malla)  # XI: lon, YI: lat

x_obs = df["lon"].to_numpy()
y_obs = df["lat"].to_numpy()

def k_mas_cercanos(pesos: np.ndarray, k: int) -> np.ndarray:
    # Me quedo con los k más grandes (forma simple)
    if k is None or k <= 0 or k >= pesos.size:
        return pesos
    idx = np.argsort(pesos)[-k:]
    out = np.zeros_like(pesos)
    out[idx] = pesos[idx]
    return out

def idw_simple(xs, ys, zs, XI, YI, power=2.0, k=None, eps=1e-12, chunk=6000):
    # Interpolo con IDW en bloques para no quedarme sin memoria
    xyi = np.column_stack([XI.ravel(), YI.ravel()])
    out = np.full(xyi.shape[0], np.nan, dtype=float)
    pts = np.column_stack([xs, ys])
    zs = np.asarray(zs, dtype=float)

    for inicio in range(0, xyi.shape[0], chunk):
        fin = min(inicio + chunk, xyi.shape[0])
        bloque = xyi[inicio:fin]
        dx = bloque[:, [0]] - pts[:, 0]
        dy = bloque[:, [1]] - pts[:, 1]
        dist2 = dx*dx + dy*dy + eps
        w = 1.0 / (dist2 ** (power / 2.0))

        # Me quedo con k vecinos si quiero que sea más local
        if k is not None and k > 0 and k < w.shape[1]:
            for r in range(w.shape[0]):
                w[r, :] = k_mas_cercanos(w[r, :], k)

        num = (w * zs).sum(axis=1)
        den = w.sum(axis=1)
        valores = num / den

        # Si hay un punto encima, uso su valor directamente
        cerca = (dist2 <= (eps * 2)).any(axis=1)
        if cerca.any():
            idxs = np.where(cerca)[0]
            for r in idxs:
                j = np.argmin(dist2[r])
                valores[r] = zs[j]

        out[inicio:fin] = valores

    return out.reshape(XI.shape)

def rango_seguro(a):
    r = np.nanmax(a) - np.nanmin(a)
    return r if r > 0 else 1.0

def homogeneidad_simple(pesos, v_elev, v_sal, v_temp, re, rs, rt):
    # Mido cuánta variación hay cerca usando los mismos pesos (versión directa)
    w = np.asarray(pesos, dtype=float)
    s = w.sum()
    if s <= 0:
        return 0.0
    wn = w / s

    def desvest_pond(x, wnorm):
        m = (wnorm * x).sum()
        var = (wnorm * (x - m) ** 2).sum()
        return float(np.sqrt(var))

    se = desvest_pond(v_elev, wn) / re
    ss = desvest_pond(v_sal,  wn) / rs
    st = desvest_pond(v_temp, wn) / rt

    # Entre más baja la desviación, más homogéneo (me acerco a 1)
    return float(np.clip(1.0 - np.mean([se, ss, st]), 0.0, 1.0))

def pesos_locales_idw(x0, y0, xs, ys, power, k, eps):
    # Pesos IDW en un punto (forma simple)
    dist2 = (x0 - xs)**2 + (y0 - ys)**2 + eps
    w = 1.0 / (dist2 ** (power / 2.0))
    if k is not None and 0 < k < w.size:
        w = k_mas_cercanos(w, k)
    return w

def graficar(Z, XI, YI, titulo, puntos_df):
    plt.figure(figsize=(7, 6))
    extent = [XI.min(), XI.max(), YI.min(), YI.max()]
    plt.imshow(Z, origin="lower", extent=extent, aspect="auto")

    for cultivo in puntos_df["cultivo"].astype(str).unique():
        m = (puntos_df["cultivo"].astype(str) == cultivo)
        plt.scatter(puntos_df.loc[m, "lon"], puntos_df.loc[m, "lat"], s=10, alpha=0.6, label=str(cultivo))

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.colorbar(label=titulo)
    plt.legend(title="Cultivo", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

# Interpolo las variables ambientales con IDW
Z_elev = idw_simple(x_obs, y_obs, df["elevacion"].values, XI, YI,
                    power=potencia_idw, k=max_vecinos, eps=epsilon)
Z_sal  = idw_simple(x_obs, y_obs, df["salinidad"].values, XI, YI,
                    power=potencia_idw, k=max_vecinos, eps=epsilon)
Z_temp = idw_simple(x_obs, y_obs, df["temperatura"].values, XI, YI,
                    power=potencia_idw, k=max_vecinos, eps=epsilon)

# Factores F1/F2/F3
cultivos = df["cultivo"].astype(str).values
clases, idx_clase = np.unique(cultivos, return_inverse=True)

val_elev = df["elevacion"].values
val_sal  = df["salinidad"].values
val_temp = df["temperatura"].values

re = rango_seguro(val_elev)
rs = rango_seguro(val_sal)
rt = rango_seguro(val_temp)

F1 = np.zeros_like(XI, dtype=float)  # fracción del cultivo dominante
F2 = np.zeros_like(XI, dtype=float)  # homogeneidad ambiental (0..1)
F3 = np.zeros_like(XI, dtype=float)  # mezcla (penalizo si hay mucha mezcla)

# Para juntar parches cercanos, uso una "caja" simple hecha con una máscara booleana
def pesos_caja_simple(x0, y0, xs, ys, semilado):
    m = (np.abs(x0 - xs) <= semilado) & (np.abs(y0 - ys) <= semilado)
    w = np.where(m, 1.0, 0.0)
    return w

for i in range(XI.shape[0]):
    for j in range(XI.shape[1]):
        x0, y0 = XI[i, j], YI[i, j]

        # Para F1/F3 uso una caja simple (peso 1 si cae dentro)
        w_fact = pesos_caja_simple(x0, y0, x_obs, y_obs, semilado_factores_deg)
        if w_fact.sum() == 0:
            # Si la caja queda vacía, uso IDW normal
            w_fact = pesos_locales_idw(x0, y0, x_obs, y_obs, potencia_idw, max_vecinos, epsilon)

        # Sumo pesos por cultivo (versión directa con bucle para que sea fácil de leer)
        pesos_por_clase = np.zeros(len(clases), dtype=float)
        for c in range(len(clases)):
            pesos_por_clase[c] = w_fact[idx_clase == c].sum()

        total = pesos_por_clase.sum()
        if total > 0:
            frac_max = float(pesos_por_clase.max() / total)  # F1
        else:
            frac_max = 0.0

        F1[i, j] = frac_max
        F3[i, j] = 1.0 - frac_max   # Mezcla simple: si domina uno, penalizo poco

        # Para F2 uso IDW normal (misma vecindad que antes)
        w_amb = pesos_locales_idw(x0, y0, x_obs, y_obs, potencia_idw, max_vecinos, epsilon)
        F2[i, j] = homogeneidad_simple(w_amb, val_elev, val_sal, val_temp, re, rs, rt)

# Idoneidad final (0..1)
idoneidad_total = np.clip(W_MONO * F1 + W_UNIF * F2 - P_MULTI * F3, 0.0, 1.0)

# Mapas
graficar(Z_elev, XI, YI, f"Elevación (IDW p={potencia_idw}, k={max_vecinos})", df)
graficar(Z_sal,  XI, YI, f"Salinidad (IDW p={potencia_idw}, k={max_vecinos})", df)
graficar(Z_temp, XI, YI, f"Temperatura (IDW p={potencia_idw}, k={max_vecinos})", df)
graficar(idoneidad_total, XI, YI,
         "Idoneidad: 0.70 monocultivo + 0.30 uniformidad - penalización por mezcla",
         df)

# Guardo salidas para pso.py
np.save("idoneidad_total.npy", idoneidad_total)
np.save("XI.npy", XI)
np.save("YI.npy", YI)


