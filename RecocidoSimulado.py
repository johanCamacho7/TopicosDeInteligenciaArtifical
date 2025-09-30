import math
import random
import numpy as np


def calcularDistanciaTotal(recorrido, matriz_dist):
    """Calcula la distancia total de un recorrido (costo)."""
    distancia = 0
    for i in range(len(recorrido) - 1):
        distancia += matriz_dist[recorrido[i]][recorrido[i + 1]]
    return distancia


def vecinoMasCercano(matriz_dist):
    """
    heurístico del vecino más cercano.
    """
    n_ciudades = len(matriz_dist)
    ciudad_actual = 0
    recorrido = [ciudad_actual]
    no_visitadas = set(range(1, n_ciudades))

    while no_visitadas:
        ciudad_mas_cercana = min(no_visitadas, key=lambda ciudad: matriz_dist[ciudad_actual][ciudad])
        recorrido.append(ciudad_mas_cercana)
        no_visitadas.remove(ciudad_mas_cercana)
        ciudad_actual = ciudad_mas_cercana

    recorrido.append(recorrido[0])
    return recorrido


def recocidoSimulado(matriz_dist):
    """
    implementa el recocido simulado:
    1. calentamiento gradual.
    2. mantenimiento en temperatura máxima.
    3. enfriamiento gradual.
    """
    n = len(matriz_dist)
    #solución Inicial
    solucion_actual = vecinoMasCercano(matriz_dist)
    costo_actual = calcularDistanciaTotal(solucion_actual, matriz_dist)
    mejor_solucion = list(solucion_actual)
    mejor_costo = costo_actual

    print(f"Solución Inicial (Vecino Más Cercano): {solucion_actual}")
    print(f"Costo Inicial: {costo_actual:.2f}\n")

    #Parámetros de la simulacion
    temp_de_arranque = float(n)  # empieza bajo
    temp_maxima = float(n * 30)  # pico de temperatura
    temp_minima = 1e-5  # criterio de paro
    iteraciones_por_nivel = n * 10
    factor_calentamiento = 1.02  # sube un en cada paso
    iteraciones_en_pico = n * 40  # iteraciones  en temp máxima
    factor_enfriamiento = 1.0 - (1.0 / (n * 10.0))  # enfria lentamente

    #control de fases
    estado_actual = "CALENTAMIENTO"  # Estados: CALENTAMIENTO, PICO_TEMPERATURA, ENFRIAMIENTO
    temp_actual = temp_de_arranque
    contador_pico = 0
    iteracion = 1
    print("----- Iniciando Proceso de Recocido Simulado -----")
    while True:
        #exploración del vecindario
        for _ in range(iteraciones_por_nivel):
            vecino = list(solucion_actual)
            i, j = random.sample(range(1, n), 2)
            vecino[i], vecino[j] = vecino[j], vecino[i]
            costo_vecino = calcularDistanciaTotal(vecino, matriz_dist)
            diferencia_costo = costo_vecino - costo_actual
            if diferencia_costo < 0 or random.random() < math.exp(-diferencia_costo / temp_actual):
                solucion_actual = list(vecino)
                costo_actual = costo_vecino
                if costo_actual < mejor_costo:
                    mejor_solucion = list(solucion_actual)
                    mejor_costo = costo_actual

        #logica de transición de fases y temperatura
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
                break  #termina el algoritmo

        print(f"Iteración {iteracion:3}: Fase={estado_actual:<18} Temp={temp_actual:9.3f}, Costo Actual={costo_actual:8.2f}, Mejor Costo={mejor_costo:8.2f}")
        iteracion += 1

    return mejor_solucion, mejor_costo


if __name__ == "__main__":
    NUM_CIUDADES = 15
    # generación aleatoria del problema
    coordenadas = np.random.rand(NUM_CIUDADES, 2) * 100
    distancias = np.zeros((NUM_CIUDADES, NUM_CIUDADES))
    for i in range(NUM_CIUDADES):
        for j in range(NUM_CIUDADES):
            distancias[i][j] = np.linalg.norm(coordenadas[i] - coordenadas[j])
    print(f"--- Problema del  Viajero con {NUM_CIUDADES} ciudades ---")
    recorrido_optimo, costo_optimo = recocidoSimulado(distancias)
    print("\n--- Resultados Finales ---")
    print(f"Mejor recorrido encontrado: {recorrido_optimo}")
    print(f"Costo del mejor recorrido: {costo_optimo:.2f}")