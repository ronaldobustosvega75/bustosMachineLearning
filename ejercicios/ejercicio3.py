import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def ejecutar_ejercicio3():
    st.header("Ejercicio 3: Dataset Iris")
    st.markdown("**Objetivo:** Implementar flujo completo de preprocesamiento y visualización")
    
    st.subheader("Carga del dataset")
    iris = load_iris()
    st.write("Dataset cargado desde sklearn.datasets.load_iris()")
    st.write(f"**Descripción:** {iris.DESCR[:200]}...")
    
    st.subheader("Conversión a DataFrame")
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    st.write(f"Dimensiones: {df.shape}")
    st.write(f"Clases: {list(iris.target_names)}")
    st.dataframe(df.head(10))
    
    st.write("**Estadísticas descriptivas (datos originales):**")
    st.dataframe(df.describe())
    
    st.subheader("Estandarización de características")
    X = df.drop('target', axis=1)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_scaled = pd.DataFrame(X_scaled, columns=iris.feature_names)
    df_scaled['target'] = y
    
    st.write("**Estandarización aplicada con StandardScaler()**")
    st.write("Todas las características transformadas a media=0 y desviación estándar=1")
    
    st.write("**Estadísticas descriptivas del dataset estandarizado:**")
    st.dataframe(df_scaled[iris.feature_names].describe())
    
    st.subheader("División del dataset")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
    
    st.write("**Dimensiones de los conjuntos:**")
    results = pd.DataFrame({
        'Conjunto': ['Entrenamiento (70%)', 'Prueba (30%)'],
        'X shape': [X_train.shape, X_test.shape],
        'y shape': [y_train.shape, y_test.shape]
    })
    st.dataframe(results)
    
    st.write("**Distribución de clases:**")
    train_dist = pd.Series(y_train).value_counts().sort_index()
    test_dist = pd.Series(y_test).value_counts().sort_index()
    dist_df = pd.DataFrame({
        'Clase': iris.target_names,
        'Train': train_dist.values,
        'Test': test_dist.values
    })
    st.dataframe(dist_df)
    
    st.subheader("Visualización: Gráfico de dispersión")
    st.write("**Sepal Length vs Petal Length diferenciado por clase**")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^']
    
    for i, (color, marker, name) in enumerate(zip(colors, markers, iris.target_names)):
        mask = df_scaled['target'] == i
        ax.scatter(
            df_scaled[mask]['sepal length (cm)'],
            df_scaled[mask]['petal length (cm)'],
            color=color,
            label=name,
            alpha=0.7,
            s=80,
            marker=marker,
            edgecolors='black',
            linewidth=1
        )
    
    ax.set_xlabel('Sepal Length (estandarizado)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Petal Length (estandarizado)', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de características por clase', fontsize=14, fontweight='bold')
    ax.legend(title='Especie', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("**Observaciones:**")
    st.write("- La clase **Setosa** (rojo) es claramente separable de las otras dos")
    st.write("- Las clases **Versicolor** (verde) y **Virginica** (azul) presentan cierto solapamiento")
    st.write("- La combinación de estas dos características permite una buena discriminación entre clases")
    
    st.success(f"✓ Preprocesamiento completado: Dataset listo para clasificación")