# ============================
# Módulos Importados
# ============================
#manipulación de modelo Machine Learning
import joblib
import pickle

#Generación de graficas y reportes
import os
import streamlit as st
import matplotlib.pyplot as plt
#Manipulación de datos
import numpy as np
import pandas as pd


def load_model(model_path, method='joblib'):
    """
    Cargar un modelo de aprendizaje automático desde una ruta especificada usando joblib o pickle.
    
    Parámetros:
    model_path (str): La ruta al archivo del modelo.
    method (str): El método a usar para cargar el modelo ('joblib' o 'pickle').
    
    Retorna:
    model: El modelo de aprendizaje automático cargado.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")
    
    if method == 'joblib':
        return joblib.load(model_path, mmap_mode=None)
    elif method == 'pickle':
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise ValueError("The method parameter should be either 'joblib' or 'pickle'")

# Definir los parámetros con deslizadores y rangos realistas para paredes de concreto reforzado en la barra lateral
tw = st.sidebar.slider('tw (Espesor de la pared)', min_value=0.08, max_value=0.15, value=0.1)
hw = st.sidebar.slider('hw (Altura de la pared)', min_value=1.0, max_value=3.0, value=1.6)
lw = st.sidebar.slider('lw (Longitud de la pared)', min_value=1.0, max_value=3.0, value=1.6)
lbe = st.sidebar.slider('lbe (Longitud del elemento de borde)', min_value=0.0, max_value=0.4, value=0.1)
P = st.sidebar.slider('P (Carga axial)', min_value=0.0, max_value=1500.0, value=500.0)
fc = st.sidebar.slider('fc (Resistencia a la compresión del concreto)', min_value=10.0, max_value=60.0, value=28.9)
fyh = st.sidebar.slider('fyh (Resistencia a la fluencia de la armadura horizontal)', min_value=100.0, max_value=600.0, value=500.0)
fyv = st.sidebar.slider('fyv (Resistencia a la fluencia de la armadura vertical)', min_value=100.0, max_value=600.0, value=500.0)
fbe = st.sidebar.slider('fbe (Resistencia a la fluencia de la armadura del elemento de borde)', min_value=0.0, max_value=700.0, value=420.0)
ρh = st.sidebar.slider('ρh (Relación de armadura horizontal)', min_value=0.0, max_value=0.5, value=0.2)
ρvbe = st.sidebar.slider('ρvbe (Relación de armadura vertical del elemento de borde)', min_value=0.0, max_value=0.3, value=0.8)
ρhbe = st.sidebar.slider('ρhbe (Relación de armadura horizontal del elemento de borde)', min_value=0.0, max_value=1.0, value=0.68)
Tag_web_reinf = st.sidebar.radio('Tag_web_reinf (Tipo de refuerzo: 0 para WWM, 1 para DB)', options=[0, 1], index=1)

# Calcular parámetros adicionales
Ag = tw * lw  # Área bruta de la sección de la pared
P_fc_Ag = P / (fc * Ag)
ρvbe_fbe_fc = ρvbe * fbe / fc
ρh_fyh_fc = ρh * fyh / fc
tw_lw = tw / lw
tw_hw = tw / hw
lbe_lw = lbe / lw
hw_lw = hw / lw

# Organizar los parámetros según el orden especificado
parameters_state_1 = [1, ρh_fyh_fc, ρvbe_fbe_fc, tw_lw, tw_hw, lbe_lw, P_fc_Ag, hw_lw, fyv, fbe, fc, Tag_web_reinf, ρhbe]
parameters_state_0 = [0, ρh_fyh_fc, ρvbe_fbe_fc, tw_lw, tw_hw, lbe_lw, P_fc_Ag, hw_lw, fyv, fbe, fc, Tag_web_reinf, ρhbe]

# Cargar el modelo una vez
model_path = r'xgb_final_model_pickle.pkl' #xgb_final_model_pickle.pkl xgb_final_model_joblib.joblib
model = load_model(model_path, method='pickle')

# Función para actualizar las predicciones
def update_predictions():
    prediction_state_1 = model.predict([parameters_state_1])
    prediction_state_0 = model.predict([parameters_state_0])
    
    # Convertir predicciones de escala logarítmica a escala natural
    prediction_state_1 = np.exp(prediction_state_1)
    prediction_state_0 = np.exp(prediction_state_0)
    
    # Promediar las predicciones
    average_prediction = [(p1 + p0) / 2 for p1, p0 in zip(prediction_state_1, prediction_state_0)]
   
    # Convertir average_prediction a un array de numpy
    average_prediction = np.array(average_prediction)
    
    # Mostrar las predicciones promedio en una tabla
    st.write('Predicciones realizadas:')
    st.table(average_prediction)
    
    # Extraer valores X e Y del array average_prediction
    x_values = average_prediction[0][1::2]  # Extraer cada segundo elemento comenzando desde el índice 0
    y_values = average_prediction[0][0::2]  # Extraer cada segundo elemento comenzando desde el índice 1
    
    # Agregar (0,0) a los valores X e Y
    x_values = np.insert(x_values, 0, 0)
    y_values = np.insert(y_values, 0, 0)
    
    # Generar el gráfico usando la funcionalidad de gráficos incorporada de Streamlit
    chart_data = pd.DataFrame({
        'Derivas %': x_values,
        'Fuerza Cortante kN': y_values
    })
    st.line_chart(chart_data.set_index('Derivas %'))
    
    # Agregar títulos de los ejes
    st.write("**Eje X:** Derivas %")
    st.write("**Eje Y:** Fuerza Cortante kN")
    
    # Agregar texto explicativo para los ejes
    st.write("Genera un archivo TXT con los resultados obtenidos:")
    
    # Crear un botón para descargar los resultados como un archivo TXT
    if st.button('Descargar resultados'):
        with open('resultados.txt', 'w', encoding='utf-8') as f:
            f.write('Parámetros definidos:\n')
            f.write(f'tw: {tw}\n')
            f.write(f'hw: {hw}\n')
            f.write(f'lw: {lw}\n')
            f.write(f'lbe: {lbe}\n')
            f.write(f'P: {P}\n')
            f.write(f'fc: {fc}\n')
            f.write(f'fyh: {fyh}\n')
            f.write(f'fyv: {fyv}\n')
            f.write(f'fbe: {fbe}\n')
            f.write(f'ρh: {ρh}\n')
            f.write(f'ρvbe: {ρvbe}\n')
            f.write(f'ρhbe: {ρhbe}\n')
            f.write(f'Tag_web_reinf: {Tag_web_reinf}\n')
            f.write('\nCurva de capacidad resultante:\n')
            f.write('Derivas %\tFuerza Cortante kN\n')
            for x, y in zip(x_values, y_values):
                f.write(f'{x}\t{y}\n')
        
        with open('resultados.txt', 'rb') as f:
            st.download_button('Descargar archivo', f, file_name='resultados.txt')

# Llamar a la función update_predictions cada vez que se modifique un parámetro de entrada
update_predictions()
