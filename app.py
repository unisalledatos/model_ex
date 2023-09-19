import streamlit as st
import numpy as np 
import pandas as pd 
import pickle 

with open('./model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Predicción de ingreso al mercado laboral")

st.header("Seleccione la información")

edad = st.slider("Seleccione edad", 16, 85)
educ = st.slider("Seleccione años de educación", 0, 30)
huswage = st.slider("Seleccione salario de pareja", 0, 500000)
kidslt6 = st.slider("Seleccione número de hijos menores a 6 años", 0, 5)

datos = np.array([[kidslt6, edad, educ, huswage]])
pred = round(model.predict_proba(datos)[0][1],4)
st.text(f"Resultado: {pred * 100}%")