import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from functions import load_lotes, save_lotes

st.set_page_config(layout="wide",page_title="AgroMatrix",
                   page_icon="📈")

def lotes_page():
    st.markdown('<h2 style="font-size: 24px;">Cargar Lotes</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([5, 1, 4])
    with col1:
         lotes()
    with col3:
         display_lote_image()


# Lote y Ambientación Page
def lotes():
        nombre = st.text_input("Nombre del Campo", key="campo_nombre", placeholder="Nombre del Campo")
        identificacion = st.text_input("Identificación del lote", key="lote_identificacion", placeholder="Identificación del Lote")
        shape = st.file_uploader("Shape", type=["shp"], key="lote_shape")
        
        if st.button("Cargar", key="lote_cargar", type="primary"):
            if shape is not None:
                shapes_folder = "shapes"
                os.makedirs(shapes_folder, exist_ok=True)
                shape_path = os.path.join(shapes_folder, shape.name)
                # st.session_state["shape_path"] = shape_path 
                with open(shape_path, "wb") as f:
                    f.write(shape.getbuffer())
                
                lotes_df = load_lotes()
                new_row = {"Nombre del Campo": nombre, "Identificación del Lote": identificacion, "Archivo": shape.name}
                #lotes_df = lotes_df.append(new_row, ignore_index=True)
                lotes_df = pd.concat([lotes_df,pd.DataFrame(new_row, index=[0])], ignore_index=True)
                save_lotes(lotes_df)

                st.session_state['lote_cargado'] = True
                
                st.toast("Lote cargado", icon="✅")
                st.rerun()
            else:
                st.toast("No se ha subido ningún archivo", icon="⚠️")
        
        lotes_df = load_lotes()
        st.markdown('<h3 style="font-size: 20px;">Lotes guardados</h3>', unsafe_allow_html=True)
        if not lotes_df.empty:
            st.dataframe(
                lotes_df
            )
        else:
            st.markdown("Aún no se ha subido ningún lote")

def display_lote_image():
    if st.session_state.get('lote_cargado', False):
        with st.spinner('Generando el polígono...'):
            time.sleep(3)
            st.toast("Polígono generado!", icon="✅")
            lotes = gpd.read_file("shapes/lote_plot/lote_2.shp")
            lote = lotes[lotes['Lote'] == '2bc AS']

            st.write("")
            
            fig, ax = plt.subplots(figsize=(10, 7))
            
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            
            lote.plot(color='g', ax=ax, edgecolor='white', linewidth=1.5, alpha=0.6)
            
            ax.set_title("Polígono correspondiente al Lote", color='white', fontsize=16, weight='bold', pad=20)
            ax.set_xlabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)
            ax.set_ylabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)
            
            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=10)

            plt.xticks(rotation=75, ha='right')
            
            plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)
            
            lote.boundary.plot(ax=ax, color='white', linewidth=3, alpha=0.2)
            lote.boundary.plot(ax=ax, marker='o', markersize=5, color='yellow', linestyle='None')

            # Render the plot
            st.pyplot(fig)
        st.session_state['lote_cargado'] = False

        # por si se quiere mostrar la imagen en lugar de armar el gráfico:
        # image_path = "imagenes/lote-2BC.png"
        # st.image(image_path, caption="Imagen del Lote cargado")


lotes_page()