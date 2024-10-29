import streamlit as st
import login as login
from intEgra import main

st.header('Int:blue[E]gra, donde tus datos se convierten en :blue[Decisiones]')
login.generarLogin()
if 'usuario' in st.session_state:
    if __name__ == "__main__":
        main()