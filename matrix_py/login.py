import streamlit as st
import pandas as pd 


# Validación simple de usuario y clave con un archivo csv

def validarUsuario(usuario,clave):    
    """Permite la validación de usuario y clave

    Args:
        usuario (str): usuario a validar
        clave (str): clave del usuario

    Returns:
        bool: True usuario valido, False usuario invalido
    """    
    dfusuarios = pd.read_csv('matrix_py/usuarios.csv')
    if len(dfusuarios[(dfusuarios['usuario']==usuario) & (dfusuarios['clave']==clave)])>0:
        return True
    else:
        return False

def generarMenu(usuario):
    """Genera el menú dependiendo del usuario

    Args:
        usuario (str): usuario utilizado para generar el menú
    """        
    with st.sidebar:
        # Cargamos la tabla de usuarios
        dfusuarios = pd.read_csv('matrix_py/usuarios.csv')
        # Filtramos la tabla de usuarios
        dfUsuario =dfusuarios[(dfusuarios['usuario']==usuario)]
        # Cargamos el nombre del usuario
        nombre= dfUsuario['nombre'].values[0]
        #Mostramos el nombre del usuario
        st.write(f"Hola **:blue-background[{nombre}]** ")
        # Mostramos los enlaces de páginas
        #st.page_link("pages/1_Dashword.py", label="Dashword")
        st.page_link("pages/2_Lotes.py", label="Lotes")
        st.page_link("pages/3_Ambientacion.py", label="Ambientación")
        st.page_link("pages/4_Prescripcion.py", label="Prescipción")
        st.page_link("pages/5_Datos_de_Cosecha.py", label="Datos de Cosecha")
        st.page_link("pages/6_Analisis.py",  label="Análisis de Datos")
        st.page_link("pages/7_Visualizacion.py",  label="Visualizar")
        st.page_link("pages/8_Panel_financiero.py",  label="Financiero")
        # Botón para cerrar la sesión
        btnSalir=st.button("Salir")
        if btnSalir:
            st.session_state.clear()
            # Luego de borrar el Session State reiniciamos la app para mostrar la opción de usuario y clave
            st.rerun()

def generarLogin():
    """Genera la ventana de login o muestra el menú si el login es valido
    """    
    # Validamos si el usuario ya fue ingresado    
    if 'usuario' in st.session_state:
        generarMenu(st.session_state['usuario']) # Si ya hay usuario cargamos el menu        
    else: 
        # Cargamos el formulario de login       
        with st.form('frmLogin'):
            parUsuario = st.text_input('Usuario')
            parPassword = st.text_input('Password',type='password')
            btnLogin=st.form_submit_button('Ingresar',type='primary')
            if btnLogin:
                if validarUsuario(parUsuario,parPassword):
                    st.session_state['usuario'] =parUsuario
                    # Si el usuario es correcto reiniciamos la app para que se cargue el menú
                    st.rerun()
                else:
                    # Si el usuario es invalido, mostramos el mensaje de error
                    st.error("Usuario o clave inválidos",icon=":material/gpp_maybe:")                    