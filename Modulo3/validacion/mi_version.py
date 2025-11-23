"""
mi implementacion de algoritmos geneticos para la solucion del problema del viajante de comercio
    mi primer cambio fue la identificacion de clases definidas
        las clases ruta/el algoritmo genetico estaban definidas como funciones las converti en clases con sus respectivos metodos
        asi como la clase aptitud se convirtio en un metodo de la clase ruta
    mi segundo cambio fue integrar a fondo las librerias Numpy y Pandas
        estas estaban importadas, pero siendo usadas de manera limitada
    mi tercer cambio fue la correccion de errores de sintaxis y logica en los metodos
"""
# imports
import numpy as np
import pandas as pd

class Municipio:
    """
    clase municipio con sus coordenadas x e y
    estos objetos seran los elementos de la poblacion
    usaremos numpy para almacenar las coordenadas
    no solamente para hacer el cálculo de distancias
    """
    def __init__(self, x, y, nombre=None):
        """
        nuevo objeto municipio
        :param x: coordenada X
        :param y: coordenada Y
        :param nombre: nombre opcional del municipio
        """
        self.coord = np.array([x, y], dtype=float)
        self.nombre = nombre #solo para mostrar

    def distancia(self, otro):
        """
        Distancia entre este municipio y otro
        Ya no es usada por el AG, pero
        se queda porque es natural que un objeto municipio sepa su distancia de otros
        :param otro:
        :return:
        """
        diff = self.coord - otro.coord
        return np.sqrt(np.dot(diff, diff))

class Ruta:
    """
    representacion de una ruta posible es decir una solucion
    un recorrido por todos los municipios
    calula la distancia total de la ruta y su aptitud
    aprovecha de mejor manera numpy para evaluar
    """
    def __init__(self, indices, dist_matrix, municipios):
        """
        nueva ruta
        :param indices: índices de los municipios
        :param dist_matrix: matriz de distancias
        :param municipios: lista de municipios
        """
        self.indices = np.array(indices, dtype=int)
        self.dist_matrix = dist_matrix  # matriz NxN de distancias (NumPy)
        self.municipios = municipios    # referencia solo para imprimir
        self._distancia = None
        self._aptitud = None

    def distancia_total(self):
        """
        Calcula la distancia total del ciclo usando operaciones vectorizadas.
        Porque es más eficiente y se adapta mejor a problemas más grandes
        Mismo razonamiento aprovechar a full las librerias
        """
        if self._distancia is None:
            idx_from = self.indices
            idx_to = np.roll(self.indices, -1)  # desplaza para cerrar el ciclo
            self._distancia = self.dist_matrix[idx_from, idx_to].sum()
        return self._distancia

    def aptitud(self):
        """
        antes una clase ahora un metodo de ruta
        convirtiendolo en una evaluacion propia
        calculado de esta manera
        Fitness = 1 / distancia_total.
        """
        if self._aptitud is None:
            d = self.distancia_total()
            self._aptitud = 1.0 / d if d > 0 else 0.0
        return self._aptitud

    # CAMBIO: representación legible para depuración y análisis de resultados.
    # Ayuda a evaluar visualmente si la búsqueda converge a rutas razonables.
    def __repr__(self):
        nombres = [self.municipios[i].nombre or str(i) for i in self.indices]
        return f"Ruta({nombres}) Dist={self.distancia_total():.4f}"

class AlgoritmoGenetico:
    """
    algoritmo genetico para optimizar rutas en el problema TSP simple
    es el conjunto de los metodos de la implementacion original
    a sí mismo seguimos usando a fondo las librerias
    Matriz de distancias vectorizada usando NumPy.
    Rutas como arrays de índices.
    Ranking y selección con DataFrames de pandas.
    """
    def __init__(self, municipios, tam_poblacion, n_elite, tasa_mutacion):
        """
        nueva implementacion del algoritmo genetico
        :param municipios: lista de municipios
        :param tam_poblacion: tamaño de la poblacion
        :param n_elite: número de elites
        :param tasa_mutacion:  tasa de mutacion
        """
        self.municipios = municipios
        self.tam_poblacion = tam_poblacion
        self.n_elite = n_elite
        self.tasa_mutacion = tasa_mutacion
        # Matriz NxN de distancias calculada una vez
        self.dist_matrix = self.crear_matriz_distancias()

    def crear_matriz_distancias(self):
        """
        Crea matriz de distancias con NumPy:
        dist[i, j] = coord_i - coord_j
        """
        coords = np.array([m.coord for m in self.municipios])  # (N, 2)
        diff_x = coords[:, 0][:, None] - coords[:, 0][None, :]
        diff_y = coords[:, 1][:, None] - coords[:, 1][None, :]
        dist = np.sqrt(diff_x ** 2 + diff_y ** 2)
        return dist

    # creacion de la poblacion inicial
    def crear_ruta(self):
        """
        creacion de una ruta aleatoria
        :return: ruta
        """
        indices = np.random.permutation(len(self.municipios))
        return Ruta(indices, self.dist_matrix, self.municipios)

    def crear_poblacion_inicial(self):
        """
        creacion de una poblacion inicial
        :return:
        """
        return [self.crear_ruta() for _ in range(self.tam_poblacion)]

    # evalucion
    def clasificar_rutas(self, poblacion):
        """
        Construye un DataFrame con índices y aptitudes,
        y devuelve lista de (Indice, Aptitud) ordenada desc.
        """
        aptitudes = np.array([ruta.aptitud() for ruta in poblacion])
        indices = np.arange(len(poblacion))
        df = pd.DataFrame({
            "Indice": indices,
            "Aptitud": aptitudes
        })
        df_sorted = df.sort_values("Aptitud", ascending=False).reset_index(drop=True)
        return list(df_sorted.itertuples(index=False, name=None))

    # seleccion

    def seleccionar_indices(self, pop_ranked):
        """
        mantenemos la idea original (elitismo + ruleta),
        pero usamos NumPy para hacer la selección más ligera y directa.
        Esto reduce overhead y hace más eficiente la búsqueda sin cambiar la lógica evolutiva.
        :param pop_ranked:
        :return:
        """
        pop_ranked = np.array(pop_ranked, dtype=float)
        indices = pop_ranked[:, 0].astype(int)
        aptitudes = pop_ranked[:, 1]

        total_fit = aptitudes.sum()
        if total_fit <= 0:
            return indices.tolist()

        # Elites: se copian directamente los mejores
        seleccionados = indices[:self.n_elite].tolist()

        # Ruleta proporcional a la aptitud
        probs = aptitudes / total_fit
        cum_probs = np.cumsum(probs)

        n_restantes = len(indices) - self.n_elite
        rands = np.random.rand(n_restantes)
        sel = np.searchsorted(cum_probs, rands)
        seleccionados.extend(indices[sel].tolist())

        return seleccionados

    def crear_grupo_apareamiento(self, poblacion, indices):
        """
        Devuelve la lista de rutas (padres) seleccionadas para cruce
        :param poblacion:
        :param indices:
        :return:
        """
        return [poblacion[i] for i in indices]

    # Crossover (cruce)

    def cruzar(self, padre1, padre2):
        """
        Mantiene un segmento del padre1 en su posición original.
        Rellena con el orden relativo del padre2.
        :param padre1:
        :param padre2:
        :return:
        """
        ruta1 = padre1.indices
        ruta2 = padre2.indices
        n = len(ruta1)

        a, b = np.random.randint(0, n, size=2)
        inicio, fin = min(a, b), max(a, b)

        hijo = -np.ones(n, dtype=int)

        # copiar segmento del padre1
        hijo[inicio:fin] = ruta1[inicio:fin]

        usados = set(hijo[inicio:fin])

        # rellenar con genes del padre2 respetando orden
        pos = fin
        for gen in ruta2:
            if gen not in usados:
                if pos >= n:
                    pos = 0
                hijo[pos] = gen
                pos += 1

        return Ruta(hijo, self.dist_matrix, self.municipios)

    def cruzar_poblacion(self, grupo):
        """
        Nueva población: n_elite copiados directos.
        Resto generado con cruce entre padres mezclados.
        Mantenemos la idea original: los primeros n_elite del resultado se consideran élite y deben preservarse en la mutación.
        :param grupo:
        :return:
        """
        hijos = []
        tamano = len(grupo) - self.n_elite
        orden = np.random.permutation(len(grupo))
        mezclados = [grupo[i] for i in orden]

        # Elites
        hijos.extend(mezclados[:self.n_elite])

        # Hijos
        for i in range(tamano):
            padre1 = mezclados[i]
            padre2 = mezclados[-i - 1]
            hijos.append(self.cruzar(padre1, padre2))

        return hijos

    # Mutación

    def mutar_ruta(self, ruta):
        """
        Mutación por swap
        Antes se iteraba sobre todos los genes y podía aplicar muchos swaps, lo que hacía la búsqueda demasiado caótica.
        :param ruta:
        :return:
        """
        indices = ruta.indices.copy()
        if np.random.rand() < self.tasa_mutacion:
            i = np.random.randint(0, len(indices))
            j = np.random.randint(0, len(indices))
            indices[i], indices[j] = indices[j], indices[i]
        return Ruta(indices, self.dist_matrix, self.municipios)

    def mutar_poblacion(self, poblacion):
        """
        CAMBIO CRÍTICO RESPECTO A LA ORIGINAL:
        No mutamos los primeros n_elite individuos.
        Así garantizamos que las mejores soluciones sobreviven intactas.
        :param poblacion:
        :return:
        """
        nueva_poblacion = []
        for i, ruta in enumerate(poblacion):
            if i < self.n_elite:
                nueva_poblacion.append(ruta)  # conservar élite sin cambios
            else:
                nueva_poblacion.append(self.mutar_ruta(ruta))
        return nueva_poblacion

    # Evolución

    def nueva_generacion(self, poblacion):
        """
        Genera una nueva generación a partir de la población actual
        1. Clasifica rutas.
        2. Selecciona índices.
        3. Crea grupo de apareamiento.
        4. Cruza población.
        5. Muta población.
        6. Devuelve nueva población.
        :param poblacion:
        :return:
        """
        ranking = self.clasificar_rutas(poblacion)
        indices_sel = self.seleccionar_indices(ranking)
        grupo = self.crear_grupo_apareamiento(poblacion, indices_sel)
        hijos = self.cruzar_poblacion(grupo)
        nueva_poblacion = self.mutar_poblacion(hijos)
        return nueva_poblacion

    def run(self, generaciones, verbose=True):
        """
        Ejecuta el AG y devuelve la mejor ruta.
        :param generaciones:
        :param verbose:
        :return:
        """
        poblacion = self.crear_poblacion_inicial()
        # Historial de mejor fitness por generación en DataFrame
        historial = []

        if verbose:
            ranking_ini = self.clasificar_rutas(poblacion)
            idx_ini, _ = ranking_ini[0]
            mejor_ini = poblacion[idx_ini]
            print(f"Distancia inicial: {mejor_ini.distancia_total():.4f}")

        for gen in range(1, generaciones + 1):
            poblacion = self.nueva_generacion(poblacion)
            ranking = self.clasificar_rutas(poblacion)
            idx_mejor, fit_mejor = ranking[0]
            mejor = poblacion[idx_mejor]
            historial.append({
                "Generacion": gen,
                "DistanciaMejor": mejor.distancia_total(),
                "FitnessMejor": mejor.aptitud()
            })

        ranking_final = self.clasificar_rutas(poblacion)
        idx_final, _ = ranking_final[0]
        mejor_ruta = poblacion[idx_final]

        if verbose:
            df_hist = pd.DataFrame(historial)
            print(f"Distancia final: {mejor_ruta.distancia_total():.4f}")
            print("Mejor ruta encontrada:")
            print(mejor_ruta)

        return mejor_ruta


if __name__ == '__main__':
    """
    correr el algoritmo genetico
    """
    municipios = [
        Municipio(40.4168, -3.7038, "Madrid"),
        Municipio(41.3874, 2.1686, "Barcelona"),
        Municipio(39.4699, -0.3763, "Valencia"),
        Municipio(37.3891, -5.9845, "Sevilla"),
        Municipio(43.2630, -2.9350, "Bilbao")
    ]

    ga = AlgoritmoGenetico(
        municipios=municipios,
        tam_poblacion=100,
        n_elite=20,
        tasa_mutacion=0.01
    )

    mejor_ruta = ga.run(generaciones=500, verbose=True)