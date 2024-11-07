import streamlit as st
import login
from PIL import Image
from streamlit_extras.metric_cards import style_metric_cards

image = Image.open("matrix_py/imagenes/file.png")


st.set_page_config(layout="wide",
                   page_title="intEgra",
                   page_icon=image)

def Dashword_page():

    col1, col2, col3 = st.columns(3)

    with col1:
         st.header("Campo 1")
         card(5000,3500,6000,30)
         
    with col2:
         st.header("Campo 2")
         card(3000,3100,3500,15)
         #st.markdown('<h2 style="font-size: 24px;">Campo 2</h2>', unsafe_allow_html=True)
    with col3:
         st.header("Campo 3")
         card(7000,5000,9000,20)
         #st.markdown('<h2 style="font-size: 24px;">Campo 3</h2>', unsafe_allow_html=True)

def card(InverP,InverE,Potencial,Riesgo):
    st.container()
    st.metric(label="Inversion proyectada", value=InverP, delta=10)
    st.metric(label="Inversion ejecutada", value=InverE, delta=50)
    st.metric(label="Potencial", value=Potencial, delta=-30)
    st.metric(label="Riesgo", value=Riesgo, delta=25)

    style_metric_cards()




login.generarLogin()
if 'usuario' in st.session_state:
    Dashword_page()