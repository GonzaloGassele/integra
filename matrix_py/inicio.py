import streamlit as st
import login as login
from intEgra import main

st.header('Página :orange[principal]')
login.generarLogin()
if 'usuario' in st.session_state:
    if __name__ == "__main__":
        main()