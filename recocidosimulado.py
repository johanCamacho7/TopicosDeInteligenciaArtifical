import math
import random
import numpy as np


def calcularCostoDesorden(arreglo):
    """Calcula el costo (desorden) de un arreglo. Un arreglo ordenado tiene costo 0.
     Este se calcula Por Pares invertidos suman costo cada par invertido.
    """
    costo = 0
    n = len(arreglo)
    for i in range(n):
        for j in range(i + 1, n):
            if arreglo[i] > arreglo[j]:
                costo += 1
    return costo


def recocidoSimulado(problema_inicial):
    """
    implementa el recocido simulado:
    1. calentamiento gradual.
    2. mantenimiento en temperatura máxima.
    3. enfriamiento gradual.
    """
    n = len(problema_inicial)
    solucion_actual = list(problema_inicial)
    costo_actual = calcularCostoDesorden(solucion_actual)
    mejor_solucion = list(solucion_actual)
    mejor_costo = costo_actual

    print(f"Solución Inicial (Arreglo Desordenado): {solucion_actual}")
    print(f"Costo Inicial (Desorden): {costo_actual:.2f}\n")

    # parámetros de la simulación
    temp_de_arranque = float(n) / 4
    temp_maxima = float(n * 10)
    temp_minima = 1e-5
    iteraciones_por_nivel = n * 10
    factor_calentamiento = 1.02 #
    iteraciones_en_pico = n * 10 # cuando ya está caliente
    factor_enfriamiento = 1.0 - (1.0 / (n * 10.0))

    # control de fases
    estado_actual = "CALENTAMIENTO"
    temp_actual = temp_de_arranque
    contador_pico = 0
    iteracion = 1
    print("----- Iniciando Proceso de Recocido Simulado  -----")
    while True:
        swap_elegido_info = ""

        # exploración del vecindario
        for _ in range(iteraciones_por_nivel):
            vecino = list(solucion_actual)
            i, j = random.sample(range(n), 2)

            valor_i = vecino[i]
            valor_j = vecino[j]

            # Realizamos el swap
            vecino[i], vecino[j] = vecino[j], vecino[i]

            costo_vecino = calcularCostoDesorden(vecino)
            diferencia_costo = costo_vecino - costo_actual

            #  aceptamos el nuevo vecino
            if diferencia_costo < 0 or random.random() < math.exp(-diferencia_costo / temp_actual):
                solucion_actual = list(vecino)
                costo_actual = costo_vecino

                # aceptamos el swap
                swap_elegido_info = f" | Swap elegido posisiones que cambian: {valor_i} <-> {valor_j}"

                if costo_actual < mejor_costo:
                    mejor_solucion = list(solucion_actual)
                    mejor_costo = costo_actual
                    if mejor_costo == 0:
                        break

        if mejor_costo == 0: # a sido ordenado
            print(f"Iteración {iteracion:3}: Solución óptima (costo 0) encontrada{swap_elegido_info}")
            print(f"           Arreglo final:   {solucion_actual}\n")
            break

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
        # impresion
        print(
            f"Iteración {iteracion:3}: Fase={estado_actual:<18} Temp={temp_actual:9.3f}, Costo actual={costo_actual:8.2f}, Mejor Costo={mejor_costo:8.2f}{swap_elegido_info}")
        print(f"           Arreglo actual:  {solucion_actual}\n")
        iteracion += 1

    return mejor_solucion, mejor_costo


# problema de ordenamiento
if __name__ == "__main__":
    NUM_ELEMENTOS = 15

    arreglo_objetivo = list(range(NUM_ELEMENTOS))
    arreglo_desordenado = list(arreglo_objetivo)
    random.shuffle(arreglo_desordenado)
    #arreglo_desordenado = [1,3,2,4,7,5,9,8,6] #10
    #arreglo_desordenado = [1,3,2,4,100,5,9,8,1500] #9
    print(f"--- Problema: Ordenar un arreglo de {NUM_ELEMENTOS} elementos usando recocido simulado   ---")

    recorrido_optimo, costo_optimo = recocidoSimulado(arreglo_desordenado)

    print("\n--- Resultados Finales ---")
    print(f" Arreglo ordenado: {recorrido_optimo}")