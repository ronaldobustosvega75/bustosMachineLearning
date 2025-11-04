import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'ejercicios'))

st.set_page_config(page_title="ML Dataset Processing", layout="wide")

st.title("Procesamiento de Datasets en Machine Learning")
st.markdown("---")

ejercicio = st.sidebar.selectbox(
    "Selecciona un ejercicio",
    ["Inicio", "Ejercicio 1: Titanic", "Ejercicio 2: Student Performance", "Ejercicio 3: Iris"]
)

if ejercicio == "Inicio":
    st.markdown("""
    ## Bienvenido al Sistema de Procesamiento de Datasets
    
    Esta aplicación implementa las 6 etapas fundamentales del procesamiento de datos:
    
    1. **Carga del dataset**
    2. **Exploración inicial** (info, describe, nulls, tipos de datos)
    3. **Limpieza de datos** (valores nulos, duplicados, outliers)
    4. **Codificación** de variables categóricas
    5. **Normalización** o estandarización
    6. **División** en conjuntos de entrenamiento y prueba
    
    """)
    
elif ejercicio == "Ejercicio 1: Titanic":
    from ejercicio1 import ejecutar_ejercicio1
    ejecutar_ejercicio1()
    
elif ejercicio == "Ejercicio 2: Student Performance":
    from ejercicio2 import ejecutar_ejercicio2
    ejecutar_ejercicio2()
    
elif ejercicio == "Ejercicio 3: Iris":
    from ejercicio3 import ejecutar_ejercicio3
    ejecutar_ejercicio3()