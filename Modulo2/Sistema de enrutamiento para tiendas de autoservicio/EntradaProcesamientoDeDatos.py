import pandas as pd

class CargadorDatos:
    """
    """
    def __init__(self):
        self.MatrizCombustible = None
        self.MatrizDistancias = None

    def cargar_archivos(self):
        try:
            self.MatrizCombustible = pd.read_excel("Datos/matriz_costos_combustible.xlsx")
            self.MatrizDistancias = pd.read_excel("Datos/matriz_distancias.xlsx")
            print("Archivos cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar archivos: {e}")

    def obtener_datos(self):
        return self.MatrizCombustible, self.MatrizDistancias
class ProcesamientoDatos:
    """
    """
    def __init__(self, df_a: pd.DataFrame, df_b: pd.DataFrame, peso_a: float = 1.0, peso_b: float = 1.0):
        self.df_a = df_a
        self.df_b = df_b
        self.peso_a = float(peso_a)
        self.peso_b = float(peso_b)
        self.df_compuesta = None

    def componer(self) -> pd.DataFrame:
        """
        Crea la matriz compuesta por suma ponderada de las dos matrices.
        Alinea por índices y columnas automáticamente.
        """
        # convertir todo a numérico para evitar errores con texto
        a = self.df_a.apply(pd.to_numeric, errors="coerce")
        b = self.df_b.apply(pd.to_numeric, errors="coerce")

        # suma ponderada
        self.df_compuesta = (a * self.peso_a).add(b * self.peso_b, fill_value=0)
        print("Matriz compuesta creada correctamente.")
        return self.df_compuesta

    def guardar_excel(self, ruta: str = "Datos/matrizCompuesta.xlsx") -> None:
        """
        Guarda la matriz compuesta en un archivo Excel (.xlsx)
        """
        if self.df_compuesta is None:
            raise ValueError("Primero ejecuta componer() antes de guardar.")
        self.df_compuesta.to_excel(ruta, index=True, engine="openpyxl")
        print(f"Matriz compuesta guardada en: {ruta}")


if __name__ == "__main__":
    peso_combustible = 0.5
    peso_distancia = 0.5
    #Inicializar y cargar los datos
    cargador = CargadorDatos()
    cargador.cargar_archivos()
    # Obtener los DataFrames cargados
    df_combustible, df_distancia = cargador.obtener_datos()
    # Verificar si la carga fue exitosa antes de continuar
    if df_combustible is not None and df_distancia is not None:
        try:
            #Inicializar y procesar los datos
            procesador = ProcesamientoDatos(
                df_a=df_combustible,
                df_b=df_distancia,
                peso_a=peso_combustible,
                peso_b=peso_distancia
            )
            # Componer la matriz
            matriz_final = procesador.componer()
            # 4. Guardar el resultado
            procesador.guardar_excel()
            print("\nProceso de carga, procesamiento y guardado completado.")
        except ValueError as ve:
            print(f"\n--- ERROR DE PROCESAMIENTO ---")
            print(ve)
        except Exception as e:
            print(f"\nOcurrió un error inesperado durante el procesamiento: {e}")
    else:
        print("\nNo se pudo continuar con el procesamiento debido a errores en la carga de archivos.")
