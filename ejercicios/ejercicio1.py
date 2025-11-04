import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def ejecutar_ejercicio1():
    st.header("Ejercicio 1: Dataset Titanic")
    st.markdown("**Objetivo:** Preparar datos para predecir supervivencia de pasajeros")
    
    try:
        df = pd.read_csv("data/titanic.csv")
    except:
        st.error("No se encontró titanic.csv en la carpeta data/")
        return
    
    st.subheader("Carga y exploración inicial")
    st.write(f"Dimensiones originales: {df.shape}")
    st.dataframe(df.head())
    
    st.write("**Información del dataset:**")
    col1, col2 = st.columns(2)
    with col1:
        st.text(f"Total de registros: {df.shape[0]}\nTotal de columnas: {df.shape[1]}")
        st.write("**Tipos de datos:**")
        st.write(df.dtypes.value_counts())
    with col2:
        st.write("**Valores nulos:**")
        st.write(df.isnull().sum()[df.isnull().sum() > 0])
    
    st.subheader("Preprocesamiento de datos")
    
    st.write("**Paso 1: Eliminar columnas irrelevantes**")
    df_clean = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    st.write(f"Columnas eliminadas: PassengerId, Name, Ticket, Cabin")
    st.write(f"Nueva dimensión: {df_clean.shape}")
    
    st.write("**Paso 2: Tratamiento de valores nulos**")
    imputer = SimpleImputer(strategy='mean')
    df_clean['Age'] = imputer.fit_transform(df_clean[['Age']])
    df_clean['Fare'] = imputer.fit_transform(df_clean[['Fare']])
    st.write(f"- Age: Rellenado con media = {df_clean['Age'].mean():.2f}")
    st.write(f"- Fare: Rellenado con media = {df_clean['Fare'].mean():.2f}")
    
    if 'Embarked' in df_clean.columns:
        mode_val = df_clean['Embarked'].mode()[0]
        df_clean['Embarked'].fillna(mode_val, inplace=True)
        st.write(f"- Embarked: Rellenado con moda = '{mode_val}'")
    
    st.write(f"Valores nulos restantes: {df_clean.isnull().sum().sum()}")
    
    st.write("**Paso 3: Codificación de variables categóricas**")
    le_sex = LabelEncoder()
    df_clean['Sex'] = le_sex.fit_transform(df_clean['Sex'])
    st.write(f"Sex codificado: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    
    if 'Embarked' in df_clean.columns:
        le_embarked = LabelEncoder()
        df_clean['Embarked'] = le_embarked.fit_transform(df_clean['Embarked'])
        st.write(f"Embarked codificado: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
    
    st.write("**Paso 4: Estandarización de variables numéricas**")
    scaler = StandardScaler()
    df_clean[['Age', 'Fare']] = scaler.fit_transform(df_clean[['Age', 'Fare']])
    st.write("Variables estandarizadas: Age, Fare (media=0, std=1)")
    
    st.subheader("Resultados del preprocesamiento")
    st.write("**Primeros 5 registros procesados:**")
    st.dataframe(df_clean.head())
    
    st.write("**Estadísticas después de la estandarización:**")
    st.dataframe(df_clean[['Age', 'Fare']].describe())
    
    st.subheader("División del dataset")
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    st.write("**Dimensiones de los conjuntos:**")
    results = pd.DataFrame({
        'Conjunto': ['Entrenamiento (70%)', 'Prueba (30%)'],
        'X shape': [X_train.shape, X_test.shape],
        'y shape': [y_train.shape, y_test.shape]
    })
    st.dataframe(results)
    
    st.success(f"✓ Dataset procesado: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")