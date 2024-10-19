import os
import time

import pandas as pd
import streamlit as st

st.set_page_config(layout="centered", page_title="AgroMatrix",
                   page_icon="📈")


def load_ambientes():
    csv_folder = "csvs"
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, "ambientes.csv")
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["Orden de Prioridad", "Nombre del Ambiente", "Nombre"])

def load_prescripciones():
    csv_folder = "csvs"
    os.makedirs(csv_folder, exist_ok=True)
    csv_path = os.path.join(csv_folder, "prescripcion.csv")
    return pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(columns=["Nombre", "Prescripción", "Ambientación", "Ambiente", "Tipo", "Máquina", "Dosis"])

def save_prescripciones(df):
    csv_folder = "csvs"
    csv_path = os.path.join(csv_folder, "prescripcion.csv")
    df.to_csv(csv_path, index=False)

def prescripcion_page():
    st.markdown('<h2 style="font-size: 24px;"> Cargar Prescripción</h2>', unsafe_allow_html=True)

    ambientes_df = load_ambientes()
    if ambientes_df.empty:
        st.warning("No hay ambientaciones disponibles. Por favor, cree una ambientación primero.")
        return
    
    prescripcion_nombre = st.text_input("Nombre de la Prescripción", key="prescripcion_nombre", placeholder="Ingrese un nombre para la prescripción")

    lote_options = ambientes_df["Nombre"].unique()
    
    ambientacion = st.selectbox("Ambientación", lote_options, key="prescripcion_ambientacion", index=None, placeholder="Seleccione una ambientación")
    tipo = st.selectbox("Tipo de Semilla", ["Soja", "Maiz", "Trigo", "Girasol", "Sorgo", "Otro"], key="prescripcion_tipo", index=None, placeholder="Seleccione el tipo de semilla")
    tipo_maquina = st.selectbox("Monitor", ["GS2 1800", "GS2 2600", "Command Center 4200"], key="prescripcion_tipo_maquina", index=None, placeholder="Seleccione el tipo de monitor")
    
    # Filtrar los ambientes correspondientes a la ambientación seleccionada
    filtered_df = ambientes_df[ambientes_df["Nombre"] == ambientacion][["Nombre del Ambiente"]]
    
    if filtered_df.empty:
        st.warning("No hay ambientes disponibles para la ambientación seleccionada.")
        return
    
    # Añadir una columna 'Dosis' para que el usuario pueda completarla o modificarla
    filtered_df["Dosis"] = ""

    st.markdown('<h3 style="font-size: 20px;">Ingrese la dosis correspondiente a cada ambiente</h3>', unsafe_allow_html=True)
    
    filtered_df = filtered_df.reset_index(drop=True)
    
    # Mostrar solo las columnas "Nombre del Ambiente" y "Dosis", y ocultar el índice
    edited_df = st.data_editor(filtered_df[["Nombre del Ambiente", "Dosis"]], num_rows="dynamic", hide_index=True)

    col1, col2 = st.columns([1, 6.5])

    with col1:
        if st.button("Guardar", key="prescripcion_cargar", type="primary"):
            if not prescripcion_nombre:
                st.warning("Por favor, ingrese un nombre para la prescripción.")
                return
            
            prescripciones_df = load_prescripciones()
            
            # Verificar si existe una prescripción con el mismo nombre
            if not prescripciones_df.empty:
                existing_rows = prescripciones_df[(prescripciones_df["Nombre"] == prescripcion_nombre) & (prescripciones_df["Ambientación"] == ambientacion)]
            
                if not existing_rows.empty:
                    # Si existe, actualizamos los valores
                    for index, row in edited_df.iterrows():
                        prescripciones_df.loc[(prescripciones_df["Nombre"] == prescripcion_nombre) & 
                                            (prescripciones_df["Ambientación"] == ambientacion) & 
                                            (prescripciones_df["Ambiente"] == row["Nombre del Ambiente"]), 
                                            ["Tipo", "Máquina", "Dosis"]] = [tipo, tipo_maquina, row["Dosis"]]
                else:
                    # Si no existe, creamos nuevas filas
                    for index, row in edited_df.iterrows():
                        new_row = {
                            "Nombre": prescripcion_nombre,
                            "Ambientación": ambientacion,
                            "Ambiente": row["Nombre del Ambiente"],
                            "Tipo": tipo,
                            "Máquina": tipo_maquina,
                            "Dosis": row["Dosis"]
                        }
                        # prescripciones_df = prescripciones_df.append(new_row, ignore_index=True)
                        prescripciones_df = pd.concat([prescripciones_df,pd.DataFrame(new_row, index=[0])], ignore_index=True)
            else:
                for index, row in edited_df.iterrows():
                    new_row = {
                        "Nombre": prescripcion_nombre,
                        "Ambientación": ambientacion,
                        "Ambiente": row["Nombre del Ambiente"],
                        "Tipo": tipo,
                        "Máquina": tipo_maquina,
                        "Dosis": row["Dosis"]
                    }
                    # prescripciones_df = prescripciones_df.append(new_row, ignore_index=True)
                    prescripciones_df = pd.concat([prescripciones_df,pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
            save_prescripciones(prescripciones_df)
            st.session_state["prescripciones_guardadas"] = True
        
        if st.session_state.get("prescripciones_guardadas", False):
            st.toast("Se guardaron los datos exitosamente", icon="✅")
            st.session_state["habilitar_download"] = True
            st.session_state["prescripciones_guardadas"] = False
    
    with col2:
        if st.session_state.get("habilitar_download", False):
            filename = prescripcion_nombre + "-" + ambientacion
            with open("archivos/prescripcion.txt") as txt_file:
                st.download_button("Descargar", data=txt_file, file_name=filename, key="download_button")
        else:
            st.button("Descargar", disabled=True, key="disabled_button")

prescripcion_page()