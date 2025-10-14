import random
import math
import copy


def costo_inversiones(array):
    """Calcula el número de inversiones (cuántos pares están desordenados)."""
    costo = 0
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if array[i] > array[j]:
                costo += 1
    return costo


def recocido_simulado(array_inicial, T_inicial=1000, T_final=1, factor_enfriamiento=0.99):
    solucion_actual = array_inicial
    mejor_solucion = copy.deepcopy(solucion_actual)
    T = T_inicial

    print(f"Costo inicial (Inversiones): {costo_inversiones(solucion_actual)}")

    while T > T_final:
        # 1. Generar un vecino (movimiento: intercambio aleatorio)
        n = len(solucion_actual)
        idx1, idx2 = random.sample(range(n), 2)

        solucion_vecina = copy.deepcopy(solucion_actual)
        solucion_vecina[idx1], solucion_vecina[idx2] = solucion_vecina[idx2], solucion_vecina[idx1]

        costo_actual = costo_inversiones(solucion_actual)
        costo_vecino = costo_inversiones(solucion_vecina)

        delta_E = costo_vecino - costo_actual

        # 2. Decisión de Movimiento (Criterio de Metropolis)
        if delta_E < 0:
            # Es una mejora: aceptar siempre.
            solucion_actual = solucion_vecina
        else:
            # Es un empeoramiento: aceptar con probabilidad.
            probabilidad = math.exp(-delta_E / T)
            if random.random() < probabilidad:
                solucion_actual = solucion_vecina

        # 3. Actualizar el mejor global
        if costo_inversiones(solucion_actual) < costo_inversiones(mejor_solucion):
            mejor_solucion = copy.deepcopy(solucion_actual)

        # 4. Enfriamiento (Disminuir la temperatura)
        T *= factor_enfriamiento

        # Opcional: Salir si se alcanza el óptimo
        if costo_inversiones(mejor_solucion) == 0:
            break

    print(f"\nRecocido Simulado Finalizado. Costo: {costo_inversiones(mejor_solucion)}")
    return mejor_solucion


# --- Ejecución ---
array = [5, 2, 8, 1, 6, 4, 7, 3]
resultado_sa = recocido_simulado(array)
print(f"Resultado SA: {resultado_sa}")