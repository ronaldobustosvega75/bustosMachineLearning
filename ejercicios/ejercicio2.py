import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def ejecutar_ejercicio2():
    st.header("Ejercicio 2: Dataset Student Performance")
    st.markdown("**Objetivo:** Predecir la nota final (G3) de los estudiantes")
    
    try:
        df = pd.read_csv("data/student-mat.csv")
    except:
        st.error("No se encontró student-mat.csv en la carpeta data/")
        return
    
    st.subheader("Carga y análisis inicial")
    st.write(f"Dimensiones del dataset: {df.shape}")
    st.dataframe(df.head())
    
    st.write("**Análisis de variables categóricas:**")
    cat_cols = df.select_dtypes(include=['object']).columns
    col1, col2 = st.columns(2)
    with col1:
        for col in cat_cols[:len(cat_cols)//2]:
            st.write(f"- **{col}**: {df[col].nunique()} categorías → {list(df[col].unique()[:3])}")
    with col2:
        for col in cat_cols[len(cat_cols)//2:]:
            st.write(f"- **{col}**: {df[col].nunique()} categorías → {list(df[col].unique()[:3])}")
    
    st.write("**Estadísticas de variables numéricas:**")
    st.dataframe(df.describe())
    
    st.subheader("Limpieza y preprocesamiento")
    
    st.write("**Eliminación de duplicados:**")
    duplicados = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    st.write(f"Duplicados encontrados y eliminados: {duplicados}")
    st.write(f"Registros restantes: {df_clean.shape[0]}")
    
    st.write("**Normalización de variables numéricas:**")
    numeric_cols = ['age', 'absences', 'G1', 'G2']
    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    st.write(f"Variables normalizadas: {', '.join(numeric_cols)}")
    st.dataframe(df_clean[numeric_cols].describe())
    
    st.write("**One Hot Encoding de variables categóricas:**")
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
    st.write(f"Dimensión antes del encoding: {df_clean.shape}")
    st.write(f"Dimensión después del encoding: {df_encoded.shape}")
    st.write(f"Nuevas columnas creadas: {df_encoded.shape[1] - df_clean.shape[1]}")
    
    st.subheader("Resultados del preprocesamiento")
    st.write("**Dataset procesado (primeras filas):**")
    st.dataframe(df_encoded.head())
    
    st.subheader("Separación y división del dataset")
    X = df_encoded.drop('G3', axis=1)
    y = df_encoded['G3']
    
    st.write(f"**Características (X):** {X.shape[1]} variables")
    st.write(f"**Variable objetivo (y):** G3 (nota final)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    st.write("**Dimensiones de los conjuntos:**")
    results = pd.DataFrame({
        'Conjunto': ['Entrenamiento (80%)', 'Prueba (20%)'],
        'X shape': [X_train.shape, X_test.shape],
        'y shape': [y_train.shape, y_test.shape]
    })
    st.dataframe(results)
    
    st.subheader("Reto adicional: Análisis de correlación")
    df_original = pd.read_csv("data/student-mat.csv")
    correlation = df_original[['G1', 'G2', 'G3']].corr()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Matriz de correlación G1, G2, G3:**")
        st.dataframe(correlation.style.background_gradient(cmap='coolwarm').format("{:.3f}"))
        st.write(f"- Correlación G1-G2: **{correlation.loc['G1', 'G2']:.3f}**")
        st.write(f"- Correlación G1-G3: **{correlation.loc['G1', 'G3']:.3f}**")
        st.write(f"- Correlación G2-G3: **{correlation.loc['G2', 'G3']:.3f}**")
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', 
                   square=True, linewidths=2, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlación entre Notas', fontweight='bold')
        st.pyplot(fig)
    
    st.success(f"✓ Dataset procesado: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")