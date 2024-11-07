import streamlit as st
import login
from PIL import Image
from streamlit_extras.metric_cards import style_metric_cards
import numpy as np
import pandas as pd

image = Image.open("matrix_py/imagenes/file.png")


st.set_page_config(layout="wide",
                   page_title="intEgra",
                   page_icon=image)

def Dashword_page():
    st.header("Indicadores economicos")

    card(50000,30000,70000,20)


    with st.container():
     st.header("Flujo de fondo de la campaña")

    # You can call any Streamlit command, including custom components:
     chart_data = pd.DataFrame(np.random.randn(12, 3),columns=["Alquileres", "Insumos", "Labores"])
     st.bar_chart(chart_data)

     

def card(InverP,InverE,Potencial,Riesgo):
    col1,col2,col3,col4 = st.columns(4)

    col1.metric(label="Inversion proyectada", value=InverP, delta=10)
    col2.metric(label="Inversion ejecutada", value=InverE, delta=50)
    col3.metric(label="Potencial", value=Potencial, delta=-30)
    col4.metric(label="Riesgo", value=Riesgo, delta=25)

    style_metric_cards()




login.generarLogin()
if 'usuario' in st.session_state:
    Dashword_page()