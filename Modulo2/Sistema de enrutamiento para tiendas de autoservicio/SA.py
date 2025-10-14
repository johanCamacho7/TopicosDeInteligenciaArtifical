import math
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

# --- Constantes del Problema ---
NUM_CEDIS = 10
NUM_SUCURSALES = 90
MAX_TIENDAS_POR_CEDIS = 15
MIN_TIENDAS_POR_CEDIS = 1
RUTA_DATA = "data/matrizCompuesta.xlsx"
PENALIZACION_COSTO_FUERA_LIMITE = 10000000.0  # Penalización alta para asegurar el cumplimiento de 1-15


def cargar_matriz_distancia(ruta: str) -> np.ndarray:
    """
    Carga la matriz de costos
    """
    try:
        df = pd.read_excel(ruta, header=0, index_col=0)
        # Convertir a matriz NumPy de tipo float
        matriz = df.values.astype(float)
        N = NUM_CEDIS + NUM_SUCURSALES
        if matriz.shape != (N, N):
            print(f"Advertencia: El tamaño de la matriz cargada ({matriz.shape}) no es el esperado ({N}, {N}).")
        print("Matriz de costos cargada exitosamente, usando la estructura de tabla estándar.")
        return matriz
    except FileNotFoundError:
        print(f"ERROR: Archivo '{ruta}' no encontrado. ¡Verifica la ruta!")
        return np.zeros((NUM_CEDIS + NUM_SUCURSALES, NUM_CEDIS + NUM_SUCURSALES))
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el Excel: {e}")
        return np.zeros((NUM_CEDIS + NUM_SUCURSALES, NUM_CEDIS + NUM_SUCURSALES))

# Estado/Solución: Lista de 10 sub-rutas. Cada sub-ruta es una lista de índices de sucursales.

def generar_solucion_inicial() -> List[List[int]]:
    # Los índices de las sucursales van de 10 a 99
    sucursales = list(range(NUM_CEDIS, NUM_CEDIS + NUM_SUCURSALES))
    random.shuffle(sucursales)
    solucion_inicial = [[] for _ in range(NUM_CEDIS)]
    # Asegurar al menos 1 tienda por CEDIS
    for cedi_idx in range(NUM_CEDIS):
        tienda_idx = sucursales.pop(0)
        solucion_inicial[cedi_idx].append(tienda_idx)
    # Asignar el resto aleatoriamente, respetando el MAX
    while sucursales:
        tienda_idx = sucursales.pop(0)
        # Candidatos: CEDIS que tienen menos del MAX permitido
        cedis_candidatos = [i for i in range(NUM_CEDIS) if len(solucion_inicial[i]) < MAX_TIENDAS_POR_CEDIS]
        if not cedis_candidatos:
            # exeption
            break
        cedis_elegido = random.choice(cedis_candidatos)
        # Insertar en posición aleatoria para aleatorizar la ruta inicial
        pos_destino = random.randrange(len(solucion_inicial[cedis_elegido]) + 1)
        solucion_inicial[cedis_elegido].insert(pos_destino, tienda_idx)
    return solucion_inicial

def calcularCostoRutas(solucion: List[List[int]]) -> float:
    """
    Calcula el costo total de todas las 10 rutas.
    Ruta: CEDIS -> Tienda_1 -> ... -> Tienda_N -> CEDIS.
    """
    costo_global = 0.0
    for cedi_idx, ruta in enumerate(solucion):
        # cedi_idx es el índice del CEDIS (0 a 9)
        num_tiendas = len(ruta)
        # anadir costo por incumplimiento de límites (MIN/MAX)
        if num_tiendas < MIN_TIENDAS_POR_CEDIS or num_tiendas > MAX_TIENDAS_POR_CEDIS:
            costo_global += PENALIZACION_COSTO_FUERA_LIMITE
            continue
        if not ruta:
            # Si se cumple el MIN_TIENDAS_POR_CEDIS >= 1, esta condición solo se daría con MIN=0
            continue
        # CEDIS -> Primer Punto
        primer_punto = ruta[0]
        costo_global += DISTANCIA_MATRIZ[cedi_idx, primer_punto]
        # Puntos Intermedios (Tienda_i a Tienda_i+1)
        for i in range(len(ruta) - 1):
            punto_actual = ruta[i]
            punto_siguiente = ruta[i + 1]
            costo_global += DISTANCIA_MATRIZ[punto_actual, punto_siguiente]
        # Último Punto -> CEDIS
        ultimo_punto = ruta[-1]
        costo_global += DISTANCIA_MATRIZ[ultimo_punto, cedi_idx]
    return costo_global

def generar_vecino(solucion_actual: List[List[int]]) -> Tuple[List[List[int]], str]:
    """
    Genera una solución vecina aplicando un operador de movimiento aleatorio.
    Se prefieren los movimientos Inter-Ruta para explorar asignaciones.
    """
    vecino = [list(r) for r in solucion_actual]

    # 70% Inter-Ruta (Cambio de ASIGNACIÓN), 30% Intra-Ruta (Cambio de ORDEN)
    if random.random() < 0.7:
        # Mover una tienda de un CEDIS a otro (Inter-Ruta)
        return generar_vecino_inter_ruta(vecino)
    else:
        # Intercambiar dos tiendas dentro de la misma ruta (Intra-Ruta)
        return generar_vecino_intra_ruta(vecino)


def generar_vecino_inter_ruta(vecino: List[List[int]]) -> Tuple[List[List[int]], str]:
    """Mueve una tienda de un CEDIS a otro (Asignación).
    """

    #Selecciona un CEDIS de origen que pueda ceder una tienda (ruta.length > MIN)
    cedis_origen_candidatos = [i for i, r in enumerate(vecino) if len(r) > MIN_TIENDAS_POR_CEDIS]

    if not cedis_origen_candidatos:
        return vecino, "Inter-Ruta: No es posible mover (todos al mínimo)"

    cedis_origen = random.choice(cedis_origen_candidatos)
    tienda_idx_en_ruta = random.randrange(len(vecino[cedis_origen]))
    tienda_movida = vecino[cedis_origen].pop(tienda_idx_en_ruta)

    # Selecciona unos CEDIS de destino que pueda aceptar una tienda (ruta.length < MAX)
    cedis_destino_candidatos = [i for i, r in enumerate(vecino) if len(r) < MAX_TIENDAS_POR_CEDIS]

    if not cedis_destino_candidatos:
        # Revertir la operación
        vecino[cedis_origen].insert(tienda_idx_en_ruta, tienda_movida)
        return vecino, "Inter-Ruta: No es posible mover (todos al máximo)"

    cedis_destino = random.choice(cedis_destino_candidatos)

    # 3. Insertar la tienda en una posición aleatoria en la ruta destino
    pos_destino = random.randrange(len(vecino[cedis_destino]) + 1)
    vecino[cedis_destino].insert(pos_destino, tienda_movida)

    movimiento_info = f"Mover: T{tienda_movida} de C{cedis_origen + 1} a C{cedis_destino + 1}"
    return vecino, movimiento_info


def generar_vecino_intra_ruta(vecino: List[List[int]]) -> Tuple[List[List[int]], str]:
    """Intercambia el orden de dos tiendas dentro de la misma ruta (Orden).
    """

    # Seleccionar un CEDIS (ruta) con al menos 2 tiendas
    rutas_validas = [i for i, r in enumerate(vecino) if len(r) >= 2]

    if not rutas_validas:
        # Si no hay rutas de tamaño >= 2, intentar un movimiento Inter-Ruta
        return generar_vecino_inter_ruta(vecino)

    cedi_idx = random.choice(rutas_validas)
    ruta = vecino[cedi_idx]

    # Elegir dos posiciones para el swap
    i, j = random.sample(range(len(ruta)), 2)
    ruta[i], ruta[j] = ruta[j], ruta[i]

    movimiento_info = f"Swap: Orden de T{ruta[i]} y T{ruta[j]} en C{cedi_idx + 1}"
    return vecino, movimiento_info


def recocidoSimulado(problema_inicial: List[List[int]]):
    """Implementación del recocido simulado para rutas y asignación.
    """

    solucion_actual = [list(r) for r in problema_inicial]
    costo_actual = calcularCostoRutas(solucion_actual)
    mejor_solucion = [list(r) for r in solucion_actual]
    mejor_costo = costo_actual

    print(f"Costo Inicial (Distancia Total): {costo_actual:.2f}\n")

    # Parámetros ajustados
    n_puntos = NUM_SUCURSALES
    temp_de_arranque = 10.0  #
    temp_maxima = 1000.0
    temp_minima = 1e-4
    iteraciones_por_nivel = n_puntos * 5
    factor_calentamiento = 1.05
    iteraciones_en_pico = n_puntos * 5
    factor_enfriamiento = 0.98
    #factor_enfriamiento = 1.0 - (1.0 / (n_puntos * 30.0))
    # Control de fases
    estado_actual = "CALENTAMIENTO"
    temp_actual = temp_de_arranque
    contador_pico = 0
    iteracion = 1
    print("----- Iniciando Proceso de Recocido Simulado -----")
    while True:
        swap_elegido_info = ""
        costo_anterior = costo_actual
        # Exploración del vecindario
        for _ in range(iteraciones_por_nivel):
            vecino, swap_elegido_info = generar_vecino(solucion_actual)
            costo_vecino = calcularCostoRutas(vecino)
            diferencia_costo = costo_vecino - costo_actual
            # Decisión: Aceptamos si es mejor o si la probabilidad lo permite
            if diferencia_costo < 0 or random.random() < math.exp(-diferencia_costo / temp_actual):
                solucion_actual = vecino
                costo_actual = costo_vecino
                if costo_actual < mejor_costo:
                    mejor_solucion = [list(r) for r in solucion_actual]
                    mejor_costo = costo_actual
        # Lógica de Cambio de Fase
        if estado_actual == "CALENTAMIENTO":
            temp_actual *= factor_calentamiento
            if temp_actual >= temp_maxima:
                temp_actual = temp_maxima
                estado_actual = "PICO_TEMPERATURA"
        elif estado_actual == "PICO_TEMPERATURA":
            contador_pico += 1
            if contador_pico >= iteraciones_en_pico:
                estado_actual = "ENFRIAMIENTO"
        elif estado_actual == "ENFRIAMIENTO":
            temp_actual *= factor_enfriamiento
            if temp_actual < temp_minima:
                break
        # Reporte de la iteración
        if iteracion % 10 == 0 or costo_actual < costo_anterior:  # Imprime si hubo mejora
            print(
                f"Iteración {iteracion:3}: Fase={estado_actual:<18} Temp={temp_actual:9.3f}, Costo={costo_actual:12.2f}, Mejor Costo={mejor_costo:12.2f} | {swap_elegido_info}")
        iteracion += 1
    return mejor_solucion, mejor_costo

if __name__ == "__main__":
    DISTANCIA_MATRIZ = cargar_matriz_distancia(RUTA_DATA)
    # Validación de la matriz cargada
    if np.all(DISTANCIA_MATRIZ == 0) and DISTANCIA_MATRIZ.size > 0:
        print("\n--- ¡ATENCIÓN! El algoritmo NO puede ejecutarse sin la matriz de costos. ---\n")
    else:
        print(
            f"--- Problema: Asignación de {NUM_SUCURSALES} Sucursales a {NUM_CEDIS} CEDIS y Optimización de Rutas ---")
        print(f"Límites por CEDIS: {MIN_TIENDAS_POR_CEDIS} a {MAX_TIENDAS_POR_CEDIS} tiendas.")

        # Generar el problema inicial
        problema_inicial_rutas = generar_solucion_inicial()

        recorrido_optimo, costo_optimo = recocidoSimulado(problema_inicial_rutas)

        print("\n--- Resultados Finales ---")
        print(f"Mejor Costo Global (Distancia Total): {costo_optimo:.2f}")
        print("Mejor Asignación y Rutas (Índice de Tiendas: 10 a 99):")

        # Imprimir la mejor solución
        for i, ruta in enumerate(recorrido_optimo):
            num_tiendas = len(ruta)
            costo_ruta = calcularCostoRutas([[ruta]])  # Calculamos el costo individual (sin penalización)

            if ruta:
                # La ruta es CEDIS i+1 (index i) -> Tiendas -> CEDIS i+1
                print(
                    f"  CEDIS {i + 1} (Tiendas: {num_tiendas}, Costo: {costo_ruta:.2f}): [C{i + 1}] -> {' -> '.join(map(str, ruta))} -> [C{i + 1}]")
            else:
                # Esto solo ocurrirá si el costo óptimo final es penalizado
                print(f"  CEDIS {i + 1} (Tiendas: {num_tiendas}): Sin asignaciones")