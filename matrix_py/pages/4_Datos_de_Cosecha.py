import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, Normalize

from functions import update_lote_dropdown_options

st.set_page_config(layout="wide",page_title="AgroMatrix",
                   page_icon="📈")

def cosecha_page():
    col1, col2, col3 = st.columns([5, 1, 4])
    with col1:
         cosecha()
    with col3:
         display_cosecha_image()


def load_prescripciones():
    csv_folder = "csvs"
    csv_path = f"{csv_folder}/prescripcion.csv"
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["Nombre", "Ambientación", "Ambiente", "Tipo", "Método", "Máquina", "Dosis"])
    return pd.read_csv(csv_path)

def cosecha():
    st.markdown('<h2 style="font-size: 24px;">Cargar datos de Cosecha</h2>', unsafe_allow_html=True)
    
    lote_options = update_lote_dropdown_options()
    lote = st.selectbox("Lote", lote_options, key="analisis_datos_lote", index=None, placeholder="Seleccionar un lote")
    
    prescripciones_df = load_prescripciones()
    prescripcion_options = prescripciones_df["Nombre"].unique()
    
    prescripcion = st.selectbox("Prescripción", prescripcion_options, key="cosecha_prescripcion", index=None, placeholder="Seleccionar una prescripción")
    
    shape = st.file_uploader("Shape", type=["shp"], key="cosecha_shape")
    
    if st.button("Generar Mapa de Rendimiento", key="cargar_cosecha", type="primary"):
        st.session_state['cosecha_cargada'] = True

def display_cosecha_image():
    if st.session_state.get('cosecha_cargada', False):
        with st.spinner('Generando el mapa...'):
            # time.sleep(3)
            rinde = gpd.read_file("shapes/cosecha_plot/cosecha_1.shp")

            st.write("")
            st.write("")
            st.write("")

            fig, ax = plt.subplots(figsize=(10, 7))
            
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            rinde['Rendimient'] = np.where(rinde['Rendimient'] > 7,7, rinde['Rendimient'])
            rinde['Rendimient'] = np.where(rinde['Rendimient'] < 2,2, rinde['Rendimient'])
            
            rinde.plot(column='Rendimient', cmap='RdYlGn', ax=ax)
            
            norm = Normalize(vmin=0, vmax=8)
            cax = plt.gca().inset_axes([1, 0, 0.05, 1])
            colorbar = ColorbarBase(cax, cmap='RdYlGn', norm=norm, orientation='vertical')

            for label in cax.get_yticklabels():
                label.set_color('white')
            
            ax.set_title("Mapa de Rendimiento", color='white', fontsize=16, weight='bold', pad=20)
            ax.set_xlabel("Longitud", color='white', fontsize=12, weight='light', labelpad=15)
            ax.set_ylabel("Latitud", color='white', fontsize=12, weight='light', labelpad=15)

            plt.xticks(rotation=75, ha='right')
            
            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.tick_params(axis='y', colors='white', labelsize=10)
            
            plt.grid(True, color='white', linestyle='dashed', linewidth=0.7)
            
            # Render the plot
            st.pyplot(fig)
            st.toast("Mapa de rendimiento generado!", icon="✅")
        st.session_state['cosecha_cargada'] = False

cosecha_page()

# nota: habría que filtrar las prescripciones entre las del lote seleccionado, obviamente. No está hecho
