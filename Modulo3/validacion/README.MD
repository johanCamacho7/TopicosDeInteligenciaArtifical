# Algoritmo Genético para el Problema del Viajante de Comercio (TSP)

Este proyecto implementa un algoritmo genético para resolver el clásico problema del viajante de comercio (TSP), donde el objetivo es encontrar la ruta más corta que recorra una serie de ciudades y regrese al punto de partida.

## Descripción del algoritmo

Un **algoritmo genético (AG)** es una técnica de optimización inspirada en la evolución biológica.
En este caso, cada posible recorrido entre los municipios se representa como un individuo dentro de una población.
A lo largo de varias generaciones, los individuos más aptos (con menor distancia total) se seleccionan y combinan para generar nuevas rutas.
El proceso incluye:

1. **Inicialización:** Se crea una población de rutas aleatorias.
2. **Evaluación:** Se calcula la distancia total y la aptitud de cada ruta (fitness = 1 / distancia).
3. **Selección:** Se eligen los mejores individuos (elitismo) y otros mediante ruleta proporcional.
4. **Cruce (crossover):** Se combinan pares de rutas para generar nuevas soluciones.
5. **Mutación:** Se intercambian aleatoriamente posiciones en las rutas para mantener diversidad.
6. **Evolución:** Se repite el ciclo durante varias generaciones hasta encontrar una ruta óptima o cercana al óptimo.

---

## Principales cambios respecto a la versión original

1. **Reestructuración completa en clases**

    * Se crearon las clases `Municipio`, `Ruta` y `AlgoritmoGenetico`.
    * Antes, la mayoría de la lógica estaba distribuida en funciones sueltas y una clase auxiliar de aptitud.
    * Ahora, cada entidad se encarga de su propia funcionalidad (OOP más limpia y mantenible).

2. **Uso extensivo de NumPy y Pandas**

    * Las coordenadas se almacenan como arreglos de NumPy.
    * Se calcula una **matriz de distancias** vectorizada, mucho más eficiente.
    * Pandas se usa para clasificar rutas, gestionar probabilidades y facilitar la selección por ruleta.

3. **Mejoras de rendimiento y legibilidad**

    * Eliminación de bucles redundantes y cálculos repetidos.
    * Métodos más cortos, con nombres descriptivos y `docstrings` en español.
    * Se cachean los valores de distancia y aptitud para evitar recomputaciones.

4. **Corrección de errores y validación**

    * Se corrigió la creación incorrecta de la lista de ciudades.
    * Se arreglaron variables sobrescritas y errores lógicos en la selección.
---

## 📁 Archivos del proyecto

| Archivo           | Descripción                                                                                                                                                                             |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **AG.py**         | Código **original**, base del proyecto. Contiene la primera versión del algoritmo genético antes de realizar correcciones y mejoras.                                                    |
| **AGF.py**        | Versión del algoritmo original **con los errores corregidos** que impedían su ejecución.                                                                                                |
| **mi_version.py** | Implementación **reestructurada y optimizada**. Usa orientación a objetos, NumPy y Pandas de forma completa. Incluye docstrings, métodos claros y mejoras en rendimiento y legibilidad. |
| **run.py**        | Script de **pruebas y comparación** entre las distintas versiones (`ag`, `agf`, `mi_version`). Permite ejecutar los algoritmos y observar diferencias de desempeño o resultados.        |


## Cómo ejecutar el código

1. Clonar o descargar este proyecto.

2. Asegúrate de tener Python 3.8 o superior.

3. Instala las dependencias necesarias:

   ```bash
   pip install numpy pandas
   ```

4. Ejecuta el script principal desde la terminal o dentro de un IDE de Python como PyCharm:

   ```bash
   python run.py
   ```

5. El programa mostrará en consola una comporacion de los siguientes datos entre los dos codigos:
    * La distancia final después de las generaciones.
    * La mejor ruta encontrada.
    * tiempo de ejecucion
    * distancia total de la mejor ruta
---

## Dependencias

* **Python 3.8+**
* **NumPy** → manejo eficiente de arreglos y operaciones vectorizadas.
* **Pandas** → manejo de datos y cálculo de probabilidades para selección.
