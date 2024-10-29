# import os
# import time

# import geopandas as gpd
# import matplotlib.colors as mcolors
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import streamlit as st

# from functions import (load_ambientes, save_ambientes,
#                        update_lote_dropdown_options)

# st.set_page_config(layout="wide", page_title="AgroMatrix",
#                    page_icon="📈")

# def ambientacion_page():
#     col1, col2, col3 = st.columns([5, 1, 4])
#     with col1:
#         ambientacion()
#     with col3:
#         imagen_ambientes()

# def ambientacion():
#     st.markdown('<h2 style="font-size: 24px;">Generar Ambientación</h2>', unsafe_allow_html=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         lote_options = update_lote_dropdown_options()
#         lote = st.selectbox("Lote", lote_options, key="ambientacion_lote", index=None, placeholder="Seleccione un lote")
#     with col2:
#         nombre = st.text_input("Nombre de la Ambientación", key="ambientacion_nombre", placeholder="Ingrese un nombre para la ambientación")
    
#     col3, col4 = st.columns(2)
#     with col3:
#         metodo = st.selectbox("Método", ["NDVI", "Topografía", "Combinado"], key="ambientacion_metodo", index=None, placeholder="Seleccione el método")
#     with col4:
#         cantidad = st.number_input("Cantidad de Ambientes", key="ambientacion_cantidad", min_value=2, max_value=7, value=2)

#     # Cargar y mostrar el dataframe de ambientes filtrado por lote y nombre de ambientación
#     ambientes_df = load_ambientes()
#     filtered_df = ambientes_df[(ambientes_df["Nombre"] == f"{nombre} ({lote})")]

#     if st.session_state.get('mostrar_lista', False):
#         st.markdown('<h3 style="font-size: 15px;">Lista de Ambientes en orden creciente de rendimiento potencial</h3>', unsafe_allow_html=True)
#         if not filtered_df.empty:
#             st.dataframe(filtered_df[["Nombre del Ambiente"]], hide_index=True)
#         else:
#             st.markdown("Aún no se ha guardado ningún ambiente para esta ambientación")
    

#     if st.button(label="Nombrar ambientes", key="nombrar_ambientes"):
#         open_nombrar_ambientes_dialog(lote, nombre)
#     if st.session_state.get('ambientes_actualizados', False):
#         st.toast("Se guardaron los datos correctamente", icon="✅")
#         st.session_state['ambientes_actualizados'] = False

#     if st.button("Generar", key="generar_ambientacion", type="primary"):
#         st.session_state['ambientacion_cargada'] = True
    

# def imagen_ambientes():
#     st.write("")
#     st.write("")
#     display_ambientacion_image()

# def display_ambientacion_image():
#     # Comprobar si se ha cargado una ambientación
#     if st.session_state.get('ambientacion_cargada', False):
#         with st.spinner('Generando ambientación...'):
#             time.sleep(3)

#             Rinde = 'Rendimient'
#             Dosis = 'Dosis Pres'
#             Superficie = 'Area'
#             Altitud = 'Altitud'
#             Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']
#             figsize = (10, 7)
#             cmap = 'RdYlGn'

#             presc = gpd.read_file("shapes/ambientacion_plot/ambientacion_1.shp")

#             fig, ax = plt.subplots(figsize=figsize)

#             # Set background color
#             fig.patch.set_facecolor('#0e1117')
#             ax.set_facecolor('#0e1117')

#             unique_values = np.sort(presc[Dosis].unique())
#             amb_dict = dict(zip(unique_values, Amb))
#             presc['Ambiente'] = presc[Dosis].map(amb_dict)

#             cmap = plt.get_cmap(cmap)

#             presc.plot(ax=ax, column=Dosis, cmap=cmap, edgecolor='white', linewidth=1, alpha=0.8)

#             ax.set_title("Ambientación", color='white', fontsize=16, weight='bold', pad=20)
#             ax.set_xlabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)
#             ax.set_ylabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)

#             plt.xticks(rotation=75, ha='right')

#             ax.tick_params(axis='x', colors='white', labelsize=10)
#             ax.tick_params(axis='y', colors='white', labelsize=10)

#             plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)

#             presc.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.2)
#             presc.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linewidth=0.5, linestyle='None')

#             boundaries = np.arange(0, len(unique_values) + 1)
#             norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

#             sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#             sm.set_array([])
#             cbar = fig.colorbar(sm, ax=ax, shrink=1)
#             cbar.set_ticks([i + 0.5 for i in range(len(unique_values))])
#             cbar.set_ticklabels(Amb)
#             cbar.ax.tick_params(axis='both', colors='white', labelsize=10)
#             cbar.set_label('Ambiente', color='white', fontsize=12, weight='light', labelpad=10)

#             if presc.crs is None:
#                 presc = presc.set_crs('4326')
#             presc = presc.to_crs('32720')

#             # Render the plot
#             st.pyplot(fig)
#             st.toast("Ambientación generada!", icon="✅")
#         st.session_state['ambientacion_cargada'] = False


# @st.dialog("Nombrar ambientes")
# def open_nombrar_ambientes_dialog(lote, nombre):
#     st.write("Nombre cada ambiente en orden creciente de rendimiento potencial")
    
#     ambiente_names = []
#     for i in range(1, 6):
#         ambiente_name = st.text_input(f"Ambiente {i}", key=f"ambiente_name_{i}")
#         ambiente_names.append(ambiente_name)
    
#     if st.button("Guardar", key="guardar_ambientes"):
#         existing_df = load_ambientes()
        
#         # Eliminar la ambientación existente si ya está en el CSV
#         existing_df = existing_df[existing_df["Nombre"] != f"{nombre} ({lote})"]

#         # Crear el nuevo registro de la ambientación
#         new_ambientacion_df = pd.DataFrame({
#             "Orden de Prioridad": range(1, len(ambiente_names) + 1),
#             "Nombre del Ambiente": ambiente_names,
#             "Nombre": f"{nombre} ({lote})"
#         })

#         ambientes_df = pd.concat([existing_df, new_ambientacion_df], ignore_index=True)
#         save_ambientes(ambientes_df)

#         st.session_state['ambientes_actualizados'] = True
#         st.rerun()

# def load_ambientes():
#     try:
#         return pd.read_csv("csvs/ambientes.csv")
#     except FileNotFoundError:
#         return pd.DataFrame(columns=["Orden de Prioridad", "Nombre del Ambiente", "Nombre"])

# def save_ambientes(ambientes_df):
#     ambientes_df.to_csv("csvs/ambientes.csv", index=False)
#     st.session_state['mostrar_lista'] = True

# ambientacion_page()

import os
import time

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import login

from functions import (load_ambientes, save_ambientes,
                       update_lote_dropdown_options)

image = Image.open("matrix_py/imagenes/file.png")


st.set_page_config(layout="wide",
                   page_title="intEgra",
                   page_icon=image)


def ambientacion_page():
    col1, col2, col3 = st.columns([5, 1, 4])
    with col1:
        ambientacion()
    with col3:
        imagen_ambientes()

def ambientacion():
    st.markdown('<h2 style="font-size: 24px;">Generar Ambientación</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        lote_options = update_lote_dropdown_options()
        lote = st.selectbox("Lote", lote_options, key="ambientacion_lote", index=None, placeholder="Seleccione un lote")
    with col2:
        nombre = st.text_input("Nombre de la Ambientación", key="ambientacion_nombre", placeholder="Ingrese un nombre para la ambientación")
    
    col3, col4 = st.columns(2)
    with col3:
        metodo = st.selectbox("Método", ["NDVI", "Topografía", "Combinado"], key="ambientacion_metodo", index=None, placeholder="Seleccione el método")
    with col4:
        cantidad = st.number_input("Cantidad de Ambientes", key="ambientacion_cantidad", min_value=2, max_value=7, value=2)

    # Cargar y mostrar el dataframe de ambientes filtrado por lote y nombre de ambientación
    ambientes_df = load_ambientes()
    filtered_df = ambientes_df[(ambientes_df["Nombre"] == f"{nombre} ({lote})")]

    if st.session_state.get('mostrar_lista', False):
        st.markdown('<h3 style="font-size: 15px;">Lista de Ambientes en orden creciente de rendimiento potencial</h3>', unsafe_allow_html=True)
        if not filtered_df.empty:
            st.dataframe(filtered_df[["Nombre del Ambiente"]], hide_index=True)
        else:
            st.markdown("Aún no se ha guardado ningún ambiente para esta ambientación")
    

    if st.button(label="Nombrar ambientes", key="nombrar_ambientes"):
        open_nombrar_ambientes_dialog(lote, nombre)
    if st.session_state.get('ambientes_actualizados', False):
        st.toast("Se guardaron los datos correctamente", icon="✅")
        st.session_state['ambientes_actualizados'] = False

    if st.button("Generar", key="generar_ambientacion", type="primary"):
        st.session_state['ambientacion_cargada'] = True
    

def imagen_ambientes():
    st.write("")
    st.write("")
    display_ambientacion_image()

def display_ambientacion_image():
    # Comprobar si se ha cargado una ambientación
    if st.session_state.get('ambientacion_cargada', False):
        with st.spinner('Generando ambientación...'):
            time.sleep(3)

            Rinde = 'Rendimient'
            Dosis = 'Dosis Pres'
            Superficie = 'Area'
            Altitud = 'Altitud'

            # Obtener los nombres de los ambientes en el orden seleccionado
            ambientes_df = load_ambientes()
            nombre_ambientacion = f"{st.session_state.get('ambientacion_nombre')} ({st.session_state.get('ambientacion_lote')})"
            Amb = ambientes_df[ambientes_df["Nombre"] == nombre_ambientacion]["Nombre del Ambiente"].tolist()

            if not Amb:
                Amb = ['Malo', 'Regular', 'Bueno Menos', 'Bueno Mas', 'Excelente']  # Default in case no names were saved

            figsize = (10, 7)
            cmap = 'RdYlGn'

            presc = gpd.read_file("matrix_py/shapes/ambientacion_plot/ambientacion_1.shp")

            fig, ax = plt.subplots(figsize=figsize)

            # Set background color
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            unique_values = np.sort(presc[Dosis].unique())
            amb_dict = dict(zip(unique_values, Amb))
            presc['Ambiente'] = presc[Dosis].map(amb_dict)

            cmap = plt.get_cmap(cmap)

            presc.plot(ax=ax, column=Dosis, cmap=cmap, edgecolor='white', linewidth=1, alpha=0.8)

            ax.set_title("Ambientación", color='white', fontsize=16, weight='bold', pad=20)
            ax.set_xlabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)
            ax.set_ylabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)

            plt.xticks(rotation=75, ha='right')

            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=10)

            plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)

            presc.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.2)
            presc.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linewidth=0.5, linestyle='None')

            boundaries = np.arange(0, len(unique_values) + 1)
            norm = mcolors.BoundaryNorm(boundaries, ncolors=cmap.N, clip=True)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=1)
            cbar.set_ticks([i + 0.5 for i in range(len(unique_values))])
            cbar.set_ticklabels(Amb)
            cbar.ax.tick_params(axis='both', colors='white', labelsize=10)
            cbar.set_label('Ambiente', color='white', fontsize=12, weight='light', labelpad=10)

            if presc.crs is None:
                presc = presc.set_crs('4326')
            presc = presc.to_crs('32720')

            # Render the plot
            st.pyplot(fig)
            st.toast("Ambientación generada!", icon="✅")
        st.session_state['ambientacion_cargada'] = False


@st.dialog("Nombrar ambientes")
def open_nombrar_ambientes_dialog(lote, nombre):
    st.write("Nombre cada ambiente en orden creciente de rendimiento potencial")
    
    ambiente_names = []
    for i in range(1, 6):
        ambiente_name = st.text_input(f"Ambiente {i}", key=f"ambiente_name_{i}")
        ambiente_names.append(ambiente_name)
    
    if st.button("Guardar", key="guardar_ambientes"):
        existing_df = load_ambientes()
        
        # Eliminar la ambientación existente si ya está en el CSV
        existing_df = existing_df[existing_df["Nombre"] != f"{nombre} ({lote})"]

        # Crear el nuevo registro de la ambientación
        new_ambientacion_df = pd.DataFrame({
            "Orden de Prioridad": range(1, len(ambiente_names) + 1),
            "Nombre del Ambiente": ambiente_names,
            "Nombre": f"{nombre} ({lote})"
        })

        ambientes_df = pd.concat([existing_df, new_ambientacion_df], ignore_index=True)
        save_ambientes(ambientes_df)

        st.session_state['ambientes_actualizados'] = True
        st.rerun()

def load_ambientes():
    try:
        return pd.read_csv("csvs/ambientes.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Orden de Prioridad", "Nombre del Ambiente", "Nombre"])

def save_ambientes(ambientes_df):
    ambientes_df.to_csv("csvs/ambientes.csv", index=False)
    st.session_state['mostrar_lista'] = True

login.generarLogin()
if 'usuario' in st.session_state:
    ambientacion_page()
