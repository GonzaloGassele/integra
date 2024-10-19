import time

import streamlit as st

from functions import update_lote_dropdown_options

st.set_page_config(layout="centered", page_title="AgroMatrix",
                   page_icon="📈")


def analisis_page():
  st.markdown('<h2 style="font-size: 24px;">Análisis de Rendimiento</h2>', unsafe_allow_html=True)
        
  lote_options = update_lote_dropdown_options()
  lote = st.selectbox("Lote", lote_options, key="analisis_datos_lote", index=None, placeholder="Seleccionar un lote")
  tipo_analisis = st.selectbox("Tipo de Análisis", ["Siembra", "Cosecha"], key="analisis_datos_tipo", index=None, placeholder="Seleccionar el tipo")
        
  if st.button("Realizar Análisis", key="realizar_analisis", type="primary"):
    st.toast("Generando análisis", icon="ℹ️")
    time.sleep(3)
    st.toast("Análisis finalizado con éxito", icon="✅")

analisis_page()