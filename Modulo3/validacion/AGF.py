"""
version funcional de AG
estos son los cambios minimos identificados para el funcionamiento correcto del codigo original
ahora este codigo funciona
"""
import random
import numpy as np
import pandas as pd
import operator

class municipio:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    #cálculo de la distancia relativa mediante Teo.Pitágoras
    def distancia(self, municipio):
        xDis = abs(self.x - municipio.x)
        yDis = abs(self.y - municipio.y)
        distancia = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distancia

    #devuelve un listado con las coordenadas de los lugares
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Aptitud:
    def __init__(self, ruta):
        self.ruta = ruta
        self.distancia = 0
        self.f_aptitud= 0.0

    def distanciaRuta(self):
        if self.distancia == 0:
            distanciaRelativa = 0
            for i in range(0, len(self.ruta)):
                puntoInicial = self.ruta[i]
                if i + 1 < len(self.ruta):
                    puntoFinal = self.ruta[i + 1]
                else:
                    puntoFinal = self.ruta[0]
                distanciaRelativa += puntoInicial.distancia(puntoFinal)
            self.distancia = distanciaRelativa
        return self.distancia

    def rutaApta(self):
        if self.f_aptitud == 0:
            self.f_aptitud = 1 / float(self.distanciaRuta())
        return self.f_aptitud

def crearRuta(listaMunicipios):
    route = random.sample(listaMunicipios, len(listaMunicipios))
    return route

def poblacionInicial(tamanoPob,listaMunicipios):
    poblacion = []
    for i in range(0, tamanoPob):
        poblacion.append(crearRuta(listaMunicipios))
    return poblacion

def clasificacionRutas(poblacion):
    fitnessResults = {}
    for i in range(0,len(poblacion)):
        fitnessResults[i] = Aptitud(poblacion[i]).rutaApta()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def seleccionRutas(popRanked, indivSelecionados):
    resultadosSeleccion = []
    df = pd.DataFrame(np.array(popRanked), columns=["Indice","Aptitud"])
    df['cum_sum'] = df.Aptitud.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Aptitud.sum()

    for i in range(0, indivSelecionados):
        resultadosSeleccion.append(popRanked[i][0])
    """
    CORRECCIÓN:
    Antes: se usaba 'for i in range(...)' y luego otro 'for i in range(...)' dentro.
    Esto pisaba el índice externo (sombreado de variable).
    Ahora se usa '_' para el bucle externo (no se necesita el índice)
    y 'j' para el bucle interno, evitando la colisión.
    """
    for _ in range(0, len(popRanked) - indivSelecionados):
        seleccion = 100*random.random()
        for j in range(0, len(popRanked)):
            if seleccion <= df.iat[j,3]:
                resultadosSeleccion.append(popRanked[j][0])
                break
    return resultadosSeleccion

def grupoApareamiento(poblacion, resultadosSeleccion):
    grupoApareamiento = []
    for i in range(0, len(resultadosSeleccion)):
        index = resultadosSeleccion[i]
        grupoApareamiento.append(poblacion[index])
    return grupoApareamiento

def reproduccion(progenitor1, progenitor2):
    hijo = []
    hijoP1 = []
    hijoP2 = []

    generacionX = int(random.random() * len(progenitor1))
    generacionY = int(random.random() * len(progenitor2))

    generacionInicial = min(generacionX, generacionY)
    generacionFinal = max(generacionX, generacionY)

    for i in range(generacionInicial, generacionFinal):
        hijoP1.append(progenitor1[i])

    hijoP2 = [item for item in progenitor2 if item not in hijoP1]

    hijo = hijoP1 + hijoP2
    return hijo

def reproduccionPoblacion(grupoApareamiento, indivSelecionados):
    hijos = []
    tamano = len(grupoApareamiento) - indivSelecionados
    espacio = random.sample(grupoApareamiento, len(grupoApareamiento))

    for i in range(0,indivSelecionados):
        hijos.append(grupoApareamiento[i])

    for i in range(0, tamano):
        hijo = reproduccion(espacio[i], espacio[len(grupoApareamiento)-i-1])
        hijos.append(hijo)
    return hijos

def mutacion(individuo, razonMutacion):
    for swapped in range(len(individuo)):
        if(random.random() < razonMutacion):
            swapWith = int(random.random() * len(individuo))

            lugar1 = individuo[swapped]
            lugar2 = individuo[swapWith]

            individuo[swapped] = lugar2
            individuo[swapWith] = lugar1
    return individuo

def mutacionPoblacion(poblacion, razonMutacion):
    pobMutada = []
    for ind in range(0, len(poblacion)):
        individuoMutar = mutacion(poblacion[ind], razonMutacion)
        pobMutada.append(individuoMutar)
    return pobMutada

def nuevaGeneracion(generacionActual, indivSelecionados, razonMutacion):
    popRanked = clasificacionRutas(generacionActual)
    selectionResults = seleccionRutas(popRanked, indivSelecionados)
    grupoApa = grupoApareamiento(generacionActual, selectionResults)
    hijos = reproduccionPoblacion(grupoApa, indivSelecionados)
    nuevaGeneracion = mutacionPoblacion(hijos, razonMutacion)
    return nuevaGeneracion

def algoritmoGenetico(poblacion, tamanoPoblacion, indivSelecionados, razonMutacion, generaciones):
    pop = poblacionInicial(tamanoPoblacion, poblacion)
    print("Distancia Inicial: " + str(1 / clasificacionRutas(pop)[0][1]))

    for i in range(0, generaciones):
        pop = nuevaGeneracion(pop, indivSelecionados, razonMutacion)

    print("Distancia Final: " + str(1 / clasificacionRutas(pop)[0][1]))
    bestRouteIndex = clasificacionRutas (pop)[0][0]
    mejorRuta = pop[bestRouteIndex]
    return mejorRuta
"""
CORRECCIÓN :
# Antes: 'ciudades' se definía así ->
# ciudades = [
#     {"Madrid",  40.4168,  -3.7038},
#     {"Barcelona", 41.3784, 2.1925}
# ]
# Eso creaba sets y no objetos municipio por lo que .x y .y no existían.
# Ahora se crean correctamente como objetos municipio.
"""
ciudades = [
    municipio(40.4168, -3.7038),   # Madrid
    municipio(41.3784, 2.1925)     # Barcelona
]

algoritmoGenetico(
    poblacion=ciudades,
    tamanoPoblacion=100,
    indivSelecionados=20,
    razonMutacion=0.01,
    generaciones=500
)
