import random
import copy
from collections import deque


def costo_inversiones(array):
    """Calcula el número de inversiones (misma función que en SA)."""
    costo = 0
    n = len(array)
    for i in range(n):
        for j in range(i + 1, n):
            if array[i] > array[j]:
                costo += 1
    return costo


def busqueda_tabu(array_inicial, max_iteraciones=2000, tamano_tabu=7):
    solucion_actual = array_inicial
    mejor_solucion = copy.deepcopy(solucion_actual)

    # La lista tabú almacena los movimientos (pares de índices) que no se pueden revertir.
    lista_tabu = deque(maxlen=tamano_tabu)

    print(f"Costo inicial (Inversiones): {costo_inversiones(solucion_actual)}")

    for k in range(max_iteraciones):
        n = len(solucion_actual)
        mejor_movimiento = None
        mejor_costo_vecino = float('inf')

        # 1. Explorar el vecindario completo
        for i in range(n):
            for j in range(i + 1, n):
                # El movimiento es intercambiar (i, j)
                movimiento = tuple(sorted((i, j)))

                # Criterio de Aspiración: El movimiento es aceptado si mejora la mejor solución histórica (independiente de la lista tabú)
                es_aspiracion = False

                # Generar el vecino temporalmente
                vecino = copy.deepcopy(solucion_actual)
                vecino[i], vecino[j] = vecino[j], vecino[i]
                costo_vecino = costo_inversiones(vecino)

                # Criterio de Aspiración
                if costo_vecino < costo_inversiones(mejor_solucion):
                    es_aspiracion = True

                # 2. Decidir si el movimiento es el mejor ACEPTABLE (no tabú o aspiración)
                if (movimiento not in lista_tabu) or es_aspiracion:
                    if costo_vecino < mejor_costo_vecino:
                        mejor_costo_vecino = costo_vecino
                        mejor_movimiento = (i, j)

        # 3. Mover a la mejor solución no tabú (o aspiracional)
        if mejor_movimiento:
            i, j = mejor_movimiento
            solucion_actual[i], solucion_actual[j] = solucion_actual[j], solucion_actual[i]

            # 4. Actualizar la Lista Tabú (Prohibir el movimiento inverso)
            movimiento_inverso = tuple(sorted((i, j)))
            lista_tabu.append(movimiento_inverso)

            # 5. Actualizar el mejor global
            if mejor_costo_vecino < costo_inversiones(mejor_solucion):
                mejor_solucion = copy.deepcopy(solucion_actual)

        # Opcional: Salir si se alcanza el óptimo
        if costo_inversiones(mejor_solucion) == 0:
            break

    print(f"\nBúsqueda Tabú Finalizada. Costo: {costo_inversiones(mejor_solucion)}")
    return mejor_solucion


# --- Ejecución ---
array = [5, 2, 8, 1, 6, 4, 7, 3]
resultado_ts = busqueda_tabu(array)
print(f"Resultado TS: {resultado_ts}")