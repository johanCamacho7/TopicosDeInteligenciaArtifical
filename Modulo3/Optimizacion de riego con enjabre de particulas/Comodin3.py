import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Parámetros de la ejecución
"""
ruta_excel = "Datos/data.xlsx"
nombre_hoja = "Hoja1"

n_celdas = 220
margen = 0.005
epsilon = 1e-12

"""
Configuración de vecindad/IDW
- kernel "idw": pondera por distancia (1/d^p)
"""
tipo_kernel = "idw"               # "idw"
potencia_idw = 4.2                # 3.5–5.0 => menos suavizado (más local)
max_vecinos = 8                   # 6–12 => más local si es menor
semilado_caja_deg = 0.0015        # semilado de la "parcela" para 'box' (en grados)
"""
Pesos de los 3 factores (F1 + F2 - F3)
f1 = cercania de cultivos similares
f2 = homogeneidad ambiental local
f3 = penalización por mezcla de cultivos
"""
W1, W2, W3 = 0.50, 0.30, 0.20

# Carga de datos y columnas estándar
df = pd.read_excel(ruta_excel, sheet_name=nombre_hoja)

# Renombrar columnas a nombres cortos y en minúsculas
df = df.rename(columns={
    "Humedad (%)": "humedad",
    "Cultivo": "cultivo",
    "Elevación (m)": "elevacion",
    "Salinidad (dS/m)": "salinidad",
    "Temperatura (°C)": "temperatura",
    "Latitud": "lat",
    "Longitud": "lon",
})

# Construcción de la malla
min_lat, max_lat = df["lat"].min() - margen, df["lat"].max() + margen
min_lon, max_lon = df["lon"].min() - margen, df["lon"].max() + margen

latitudes_malla = np.linspace(min_lat, max_lat, n_celdas)
longitudes_malla = np.linspace(min_lon, max_lon, n_celdas)
XI, YI = np.meshgrid(longitudes_malla, latitudes_malla)  # XI: lon, YI: lat

# Puntos observados (coordenadas de las muestras)
x_obs = df["lon"].values
y_obs = df["lat"].values

# Utilidades y funciones base
def mantener_k_mayores(w: np.ndarray, k: int) -> np.ndarray:
    """
    Mantiene únicamente las k entradas más grandes del vector de pesos `w`
    y pone a 0 el resto. Si k es None, <= 0 o >= tamaño de w, devuelve w.
    """
    if k is None or k <= 0 or k >= w.size:
        return w
    idx = np.argpartition(w, -k)[-k:]
    w2 = np.zeros_like(w)
    w2[idx] = w[idx]
    return w2


def pesos_locales(x0: float, y0: float,
                xs: np.ndarray, ys: np.ndarray,
                kernel: str = "idw",
                power: float = 2.0,
                k: int | None = None,
                box_h: float = 0.002,
                eps: float = 1e-12) -> np.ndarray:
    """
    Calcula los pesos locales en el punto (x0, y0) respecto a los puntos (xs, ys).

    Parámetros
    ----------
    kernel : "idw" o "box"
        - "idw": w = 1 / dist^power
        - "box": w = 1 dentro de una caja de semilado 'box_h'
    power : float
        Exponente usado por IDW (p). Valores altos => más local.
    k : int | None
        Si se especifica, se conservan solo los k vecinos más influyentes.
    box_h : float
        Semilado de la caja (en grados) para el kernel "box".
    eps : float
        Pequeño valor para evitar división entre cero.

    Retorna
    -------
    np.ndarray
        Vector de pesos (mismo tamaño que xs/ys).
    """
    if kernel == "box":
        mascara = (np.abs(x0 - xs) <= box_h) & (np.abs(y0 - ys) <= box_h)
        w = np.where(mascara, 1.0, 0.0).astype(float)
        # Si no hay puntos en la caja, hacer fallback a IDW:
        if w.sum() == 0.0:
            dist2 = (x0 - xs)**2 + (y0 - ys)**2 + eps
            w = 1.0 / (dist2 ** (power / 2.0))
            if k is not None and 0 < k < w.size:
                w = mantener_k_mayores(w, k)
        return w
    else:
        dist2 = (x0 - xs)**2 + (y0 - ys)**2 + eps
        w = 1.0 / (dist2 ** (power / 2.0))
        if k is not None and 0 < k < w.size:
            w = mantener_k_mayores(w, k)
        return w


def interpolar_idw(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
                XI: np.ndarray, YI: np.ndarray,
                power: float = 2.0, k: int | None = None,
                eps: float = 1e-12, chunk: int = 6000) -> np.ndarray:
    """
    Interpolación IDW (Inverse Distance Weighting) en la malla (XI, YI)
    dados puntos (xs, ys) con valores `zs`.

    Notas de rendimiento:
    - Se procesa por bloques (chunk) para controlar memoria.
    - Si se fija k, para cada celda se conservan solo los k vecinos con mayor peso.
    """
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

        # Mantener solo k vecinos por fila si aplica
        if k is not None and k > 0 and k < w.shape[1]:
            for r in range(w.shape[0]):
                w[r, :] = mantener_k_mayores(w[r, :], k)

        numerador = (w * zs).sum(axis=1)
        denominador = w.sum(axis=1)
        valores_bloque = numerador / denominador

        # Si alguna celda coincide casi exactamente con un punto observado, copiar su valor
        mascara_cercana = (dist2 <= (eps * 2)).any(axis=1)
        if mascara_cercana.any():
            idxs = np.where(mascara_cercana)[0]
            for bi in idxs:
                j = np.argmin(dist2[bi])
                valores_bloque[bi] = zs[j]

        out[inicio:fin] = valores_bloque

    return out.reshape(XI.shape)


def entropia_normalizada(pesos_por_clase: np.ndarray) -> float:
    """
    Entropía de Shannon normalizada (0..1) de un vector de pesos por clase.
    0 => una sola clase domina; 1 => máxima mezcla uniforme.
    """
    w = np.array(pesos_por_clase, dtype=float)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    H = -(p * np.log(p)).sum()
    Hmax = np.log(len(p))
    return float(H / Hmax) if Hmax > 0 else 0.0


def rango_seguro(a: np.ndarray) -> float:
    """
    Retorna el rango (max-min). Si el rango es 0 o NaN, devuelve 1.0 para
    evitar divisiones por cero más adelante.
    """
    a = np.asarray(a, dtype=float)
    r = np.nanmax(a) - np.nanmin(a)
    return r if r > 0 else 1.0


def homogeneidad_ambiental_local(pesos: np.ndarray,
                                v_elev: np.ndarray,
                                v_sal: np.ndarray,
                                v_temp: np.ndarray,
                                rango_elev: float,
                                rango_sal: float,
                                rango_temp: float) -> float:
    """
    Mide homogeneidad (0..1) del entorno local ponderado por `pesos`,
    usando la dispersión relativa de elevación, salinidad y temperatura.
    1 => muy homogéneo; 0 => muy heterogéneo.
    """
    w = np.asarray(pesos, dtype=float)
    wn = w / w.sum() if w.sum() > 0 else w
    if wn.sum() == 0:
        return 0.0

    def desviacion_std_ponderada(x: np.ndarray, wnorm: np.ndarray) -> float:
        m = (wnorm * x).sum()
        var = (wnorm * (x - m) ** 2).sum()
        return np.sqrt(var)

    s_elev = desviacion_std_ponderada(v_elev, wn) / rango_elev
    s_sal  = desviacion_std_ponderada(v_sal,  wn) / rango_sal
    s_temp = desviacion_std_ponderada(v_temp, wn) / rango_temp

    # Convertir dispersión (alto = heterogéneo) a homogeneidad (alto = homogéneo)
    return float(np.clip(1.0 - np.mean([s_elev, s_sal, s_temp]), 0.0, 1.0))


def graficar_mapa_calor(Z: np.ndarray, XI: np.ndarray, YI: np.ndarray,
                        titulo: str, puntos_df: pd.DataFrame) -> None:
    """
    Dibuja un mapa de calor para la matriz Z en la extensión de la malla (XI, YI),
    y superpone los puntos de muestreo coloreados por tipo de cultivo.
    """
    plt.figure(figsize=(7, 6))
    extent = [XI.min(), XI.max(), YI.min(), YI.max()]
    plt.imshow(Z, origin="lower", extent=extent, aspect="auto")

    # Cada tipo de cultivo con un color distinto
    cultivos_unicos = puntos_df["cultivo"].unique()
    for cultivo in cultivos_unicos:
        mascara = puntos_df["cultivo"] == cultivo
        plt.scatter(
            puntos_df.loc[mascara, "lon"],
            puntos_df.loc[mascara, "lat"],
            s=10, alpha=0.6, label=cultivo
        )

    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(titulo)
    plt.colorbar(label=titulo)
    plt.legend(title="Tipos de Cultivo", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()


# Interpolaciones (3 capas)
Z_elevacion = interpolar_idw(
    x_obs, y_obs, df["elevacion"].values, XI, YI,
    power=potencia_idw, k=max_vecinos, eps=epsilon
)
Z_salinidad = interpolar_idw(
    x_obs, y_obs, df["salinidad"].values, XI, YI,
    power=potencia_idw, k=max_vecinos, eps=epsilon
)
Z_temperatura = interpolar_idw(
    x_obs, y_obs, df["temperatura"].values, XI, YI,
    power=potencia_idw, k=max_vecinos, eps=epsilon
)

# Factores
cultivos = df["cultivo"].astype(str).values
cultivos_unicos, idx_cultivo = np.unique(cultivos, return_inverse=True)

valores_elevacion = df["elevacion"].values
valores_salinidad = df["salinidad"].values
valores_temperatura = df["temperatura"].values

rango_elevacion = rango_seguro(valores_elevacion)
rango_salinidad = rango_seguro(valores_salinidad)
rango_temperatura = rango_seguro(valores_temperatura)

# F1: premio por predominio de un mismo cultivo (0..1)
# F2: premio por homogeneidad ambiental local (0..1)
# F3: penalización por mezcla de cultivos (entropía normalizada 0..1)
factor_cultivo = np.zeros_like(XI, dtype=float)
factor_homogeneidad = np.zeros_like(XI, dtype=float)
factor_multicultivo = np.zeros_like(XI, dtype=float)

for i in range(XI.shape[0]):
    for j in range(XI.shape[1]):
        x0, y0 = XI[i, j], YI[i, j]
        w = pesos_locales(
            x0, y0, x_obs, y_obs,
            kernel=tipo_kernel, power=potencia_idw,
            k=max_vecinos, box_h=semilado_caja_deg, eps=epsilon
        )

        # Sumar pesos por clase de cultivo
        pesos_por_clase = np.zeros(len(cultivos_unicos), dtype=float)
        np.add.at(pesos_por_clase, idx_cultivo, w)
        total_peso = pesos_por_clase.sum()

        # F1: fracción del cultivo dominante en la vecindad
        factor_cultivo[i, j] = 0.0 if total_peso <= 0 else float(pesos_por_clase.max() / total_peso)

        # F3: entropía (mezcla) — más mezcla => mayor penalización
        factor_multicultivo[i, j] = entropia_normalizada(pesos_por_clase)

        # F2: homogeneidad ambiental local (elev/sal/temp)
        factor_homogeneidad[i, j] = homogeneidad_ambiental_local(
            w, valores_elevacion, valores_salinidad, valores_temperatura,
            rango_elevacion, rango_salinidad, rango_temperatura
        )

# Mapa maestro de idoneidad
idoneidad_total = np.clip(
    W1 * factor_cultivo + W2 * factor_homogeneidad - W3 * factor_multicultivo,
    0.0, 1.0
)


# Visualización
graficar_mapa_calor(
    Z_elevacion, XI, YI,
    f"Elevación (IDW p={potencia_idw}, k={max_vecinos})", df
)
graficar_mapa_calor(
    Z_salinidad, XI, YI,
    f"Salinidad (IDW p={potencia_idw}, k={max_vecinos})", df
)
graficar_mapa_calor(
    Z_temperatura, XI, YI,
    f"Temperatura (IDW p={potencia_idw}, k={max_vecinos})", df
)
graficar_mapa_calor(
    idoneidad_total, XI, YI,
    f"Idoneidad total (kernel={tipo_kernel})", df
)

# En tu script de idoneidad:
np.save("idoneidad_total.npy", idoneidad_total)
# (Opcional) si ya tienes XI, YI:
np.save("XI.npy", XI)
np.save("YI.npy", YI)

