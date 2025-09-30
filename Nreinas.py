# Algorito tabu para la solucion de la n reinas
#Funcion para leer el numero de reinas
def leerReinas(prompt:str )-> int:
    while True:
        try:
            n=int(input(prompt))
            if n<=1:
                print('El numero debe ser mayor a 1')
            elif n in (2,3):
                print('no existe solucion para 2 o 3 reinas')
            else:
                return n
        except ValueError:
            print('Debes ingresar un numero valido')
#funcion para leer el número de iteraciones
def leerIteraciones(prompt:str)-> int:
    while True:
        try:
            n=int(input(prompt))
            return n
        except ValueError:
            print('Debes ingresar un numero valido')
#Funcion para calcular la solucion
def Nreinastabu(n,i):
    solucionActual =0
    mejorsoulcion=0
    listatabu=0
    iteracionesmaximas=0
    for iteracion in range(1, iteracionesmaximas + 1):
    # 1. Generate neighborhood
    # 2. Select the best candidate (not in tabu list or passes aspiration)
    # 3. Update current_solution
    # 4. Update tabu_list
    # 5. Update best_solution if improved
        print('Iteracion: ', iteracion)
    return



if __name__ == '__main__':
    N = leerReinas('ingresa el numero de reinas:')
    print('Calculando para ', N, 'Reinas')
    I = leerIteraciones('ingresa el numero de iteraciones:')
    print('Calculando para ', I, 'iteraciones')
    Nreinastabu(N,I)