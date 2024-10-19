import os

import streamlit as st
from pdf2image import convert_from_path

st.set_page_config(layout="wide", page_title="AgroMatrix", page_icon="📈")

def visualizacion_page():
    st.markdown('<h2 style="font-size: 24px;">Análisis disponibles para visualizar</h2>', unsafe_allow_html=True)
    input()
    pdf_folder = "pdfs"
    if st.session_state.get("selected_pdf", False):
        # selected_pdf = st.session_state.get("selected_pdf")
        # pdf_path = os.path.join(pdf_folder, selected_pdf)
        # images = convert_from_path(pdf_path)  # Convert PDF to images
        # for i, image in enumerate(images):
        #     st.image(image, caption=f"Page {i+1}", use_column_width=True)
        st.image("imagenes/dashboard.png")

def input():
    col1, col2 = st.columns(2)
    with col1:
        # Esto es para mostrar el pdf
        pdf_folder = "pdfs"
        if not os.path.exists(pdf_folder):
            st.error(f"The folder '{pdf_folder}' does not exist.")
            return

        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        # Use a placeholder for the selectbox and don't pre-select any option
        selected_pdf = st.selectbox("Seleccionar Análisis", pdf_files, index=None, key="pdf_selection", placeholder="Seleccionar un análisis")
        st.session_state["selected_pdf"] = selected_pdf
    with col2:
        st.markdown('<h2 style="font-size: 0px;"> </h2>', unsafe_allow_html=True)
        if st.session_state.get("selected_pdf", False):
            pdf_path = os.path.join(pdf_folder, selected_pdf)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Descargar", data=pdf_file, file_name=selected_pdf, key="download_button")
        else:
            st.button("Descargar", disabled=True, key="disabled_button")
        

visualizacion_page()