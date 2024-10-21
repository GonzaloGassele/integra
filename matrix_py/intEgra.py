import streamlit as st
from PIL import Image
import pydeck as pdk
import geopandas as gpd

image = Image.open("matrix_py/imagenes/file.png")

# Set the page configuration to wide layout
st.set_page_config(layout="wide",
                   page_title="intEgra",
                   page_icon=image)

# Apply custom CSS
def apply_custom_css():
    st.markdown(
        """
        <style>
        .stDataFrame, .stTable, .stDataEditor {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    


def main():
    apply_custom_css()
    st.title("intEgra")
    
     # Cargar el archivo GeoJSON
    geojson_file = "tio miguel prueba.gpkg"  # Reemplaza con la ruta a tu archivo GeoJSON
    gdf = gpd.read_file(geojson_file)

    # Convertir a formato compatible con Pydeck
    geojson_data = gdf.__geo_interface__

    centroid = gdf.geometry.unary_union.centroid
    latitude = centroid.y
    longitude = centroid.x

     # Crear el mapa con el polígono
    layer = pdk.Layer(
        'GeoJsonLayer',
        geojson_data,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_scale=20,
        line_width_min_pixels=2,
        get_fill_color='[0, 0, 0, 0]',  # Color de relleno
        get_line_color='[255, 255, 255]',    # Color del borde
    )


   # Inicializar el mapa
    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=13,  # Ajusta el zoom según sea necesario
    )

    # Mostrar el mapa
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/satellite-v9'))

    


