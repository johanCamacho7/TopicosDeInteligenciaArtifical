import pandas as pd
import folium
import numpy as np

archivo_excel = 'Datos/datos_distribucion_tiendas.xlsx'
nombre_hoja = 'Sheet1'
archivo_salida_html = 'mapa_tiendas_satelital.html'
#
COL_LATITUD = 'Latitud_WGS84'
COL_LONGITUD = 'Longitud_WGS84'
COL_NOMBRE = 'Nombre'
COL_TIPO = 'Tipo'

try:
    # Leer el archivo Excel
    df = pd.read_excel(archivo_excel, sheet_name=nombre_hoja)

    # Convertir las columnas de coordenadas a numérico (maneja texto/errores con 'coerce')
    df[COL_LATITUD] = pd.to_numeric(df[COL_LATITUD], errors='coerce')
    df[COL_LONGITUD] = pd.to_numeric(df[COL_LONGITUD], errors='coerce')

    # Eliminar filas donde las coordenadas son nulas/inválidas
    df = df.dropna(subset=[COL_LATITUD, COL_LONGITUD])

    if df.empty:
        print("Error: No se encontraron coordenadas válidas en la hoja después de la limpieza.")
        exit()
    else:
        print(f"Se cargaron {len(df)} puntos válidos para graficar.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo '{archivo_excel}'. Asegúrate de que esté en la misma carpeta.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al cargar los datos: {e}")
    exit()


latitud_central = df[COL_LATITUD].mean()
longitud_central = df[COL_LONGITUD].mean()


esri_tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
esri_attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'

mapa = folium.Map(
    location=[latitud_central, longitud_central],
    zoom_start=10,  # Ajusta el nivel de zoom (10 es un buen punto de partida)
    tiles=esri_tiles,
    attr=esri_attr,
    control_scale=True  # Muestra la escala del mapa
)

def get_marker_style(tipo):
    """Retorna el color y el ícono basado en el tipo de establecimiento."""
    tipo_str = str(tipo)  # Asegura que sea una cadena

    # Definición de estilos para los tipos específicos
    if 'Tienda' in tipo_str:
        return {'color': 'blue', 'icon': 'shopping-cart'}  # O 'home', 'store', 'building'
    elif 'Distribución' in tipo_str:
        return {'color': 'red', 'icon': 'truck'}  # O 'warehouse', 'industry'
    else:
        # Estilo por defecto si no coincide con ninguno
        return {'color': 'gray', 'icon': 'question'}


# Iterar y agregar marcadores
for index, fila in df.iterrows():
    # Obtener el estilo (color e ícono)
    estilo = get_marker_style(fila[COL_TIPO])

    # Contenido HTML para la ventana emergente (Popup)
    popup_html = f"""
    <h4>{fila[COL_NOMBRE]}</h4>
    <strong>Tipo:</strong> {fila[COL_TIPO]}<br>
    <strong>Latitud:</strong> {fila[COL_LATITUD]:.6f}<br>
    <strong>Longitud:</strong> {fila[COL_LONGITUD]:.6f}
    """

    # Crear el marcador con color e ícono basados en el tipo
    folium.Marker(
        location=[fila[COL_LATITUD], fila[COL_LONGITUD]],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=fila[COL_NOMBRE],  # Aparece al pasar el mouse
        icon=folium.Icon(color=estilo['color'], icon=estilo['icon'], prefix='fa')
        # 'prefix=fa' asegura que use Font Awesome
    ).add_to(mapa)

folium.TileLayer('OpenStreetMap', name='Vista de Calles').add_to(mapa)
folium.LayerControl().add_to(mapa)

mapa.save(archivo_salida_html)

print("\n--- PROCESO FINALIZADO ---")
print(f"El mapa interactivo se ha generado y guardado como: '{archivo_salida_html}'")
print("Abre este archivo HTML en tu navegador para ver los puntos sobre la imagen satelital.")