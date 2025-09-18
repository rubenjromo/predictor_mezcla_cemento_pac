#pip install ucimlrepo

import pandas as pd
import numpy as np
import joblib
import os
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

# --- Configuración de Archivos ---
MODEL_FILE = 'modelo_concreto_unificado.pkl'
SCALER_FILE = 'escalador_concreto_unificado.pkl'
DATABASE_FILE = 'base_de_datos_unificada.csv'

def cargar_y_preparar_datos_uci():
    """
    Carga el dataset "Concrete Compressive Strength" de UCI y lo prepara para la unificación.
    """
    print("Cargando y preparando el dataset de referencia de UCI...")
    try:
        concrete_repo = fetch_ucirepo(id=165)
        X = concrete_repo.data.features
        y = concrete_repo.data.targets
        df_uci = pd.concat([X, y], axis=1)

        df_uci.columns = [
            'cemento_kg_m3', 'escoria_alto_horno_kg_m3', 'ceniza_volante_kg_m3', 'agua_kg_m3',
            'superplastificante_kg_m3', 'agregado_grueso_kg_m3', 'agregado_fino_kg_m3',
            'edad_curado_dias', 'resistencia_compresion_mpa'
        ]

        df_uci['pac_kg_m3'] = 0
        df_uci['activador_quimico_pct'] = 0
        df_uci['fuente_dato'] = 'UCI_Real'

        return df_uci

    except Exception as e:
        print(f"No se pudo cargar el dataset de UCI. Verifica la conexión a internet y la librería 'ucimlrepo'. Error: {e}")
        return None

def generar_datos_sinteticos_pac(n_samples=2000):
    """
    Genera la base de datos sintética enfocada en el concreto con PAC.
    """
    print("Generando datos sintéticos para concreto con PAC...")
    data = {
        'cemento_kg_m3': np.random.uniform(300, 450, n_samples),
        'pac_kg_m3': np.random.uniform(10, 135, n_samples),
        'relacion_agua_aglomerante': np.random.uniform(0.4, 0.55, n_samples),
        'agregado_fino_kg_m3': np.random.uniform(600, 1350, n_samples),
        'agregado_grueso_kg_m3': np.random.uniform(900, 1250, n_samples),
        'activador_quimico_pct': np.random.choice([0, 3, 5, 8], n_samples, p=[0.5, 0.25, 0.15, 0.1]),
        'edad_curado_dias': np.random.choice([7, 28, 56, 90], n_samples)
    }
    df_pac = pd.DataFrame(data)

    df_pac['agua_kg_m3'] = df_pac['relacion_agua_aglomerante'] * (df_pac['cemento_kg_m3'] + df_pac['pac_kg_m3'])
    df_pac = df_pac.drop('relacion_agua_aglomerante', axis=1)

    df_pac['escoria_alto_horno_kg_m3'] = 0
    df_pac['ceniza_volante_kg_m3'] = 0
    df_pac['superplastificante_kg_m3'] = 0

    base_strength = 15
    cemento_factor = 0.08
    pac_factor = -0.05
    agua_factor = -0.15
    edad_factor = 0.35
    activador_factor = 2.8
    interaction_factor = 0.015

    df_pac['resistencia_compresion_mpa'] = (
        base_strength +
        df_pac['cemento_kg_m3'] * cemento_factor +
        df_pac['pac_kg_m3'] * pac_factor +
        df_pac['agua_kg_m3'] * agua_factor +
        df_pac['edad_curado_dias'] * edad_factor +
        df_pac['activador_quimico_pct'] * activador_factor +
        (df_pac['pac_kg_m3'] * df_pac['activador_quimico_pct'] * interaction_factor) +
        np.random.normal(0, 2, n_samples)
    )
    df_pac['resistencia_compresion_mpa'] = df_pac['resistencia_compresion_mpa'].clip(lower=5)
    df_pac['fuente_dato'] = 'PAC_Sintetico'

    return df_pac

def unificar_datasets(df_uci, df_pac):
    """
    Combina el dataset de UCI y el de PAC en un único DataFrame maestro.
    """
    print("Unificando datasets...")
    column_order = [
        'cemento_kg_m3', 'pac_kg_m3', 'escoria_alto_horno_kg_m3', 'ceniza_volante_kg_m3',
        'agua_kg_m3', 'superplastificante_kg_m3', 'agregado_grueso_kg_m3',
        'agregado_fino_kg_m3', 'activador_quimico_pct', 'edad_curado_dias',
        'fuente_dato', 'resistencia_compresion_mpa'
    ]

    df_pac_reordered = df_pac[column_order]
    df_uci_reordered = df_uci[column_order]

    df_unificado = pd.concat([df_uci_reordered, df_pac_reordered], ignore_index=True)
    df_unificado.to_csv(DATABASE_FILE, index=False)
    print(f"Base de datos unificada con {len(df_unificado)} muestras guardada en '{DATABASE_FILE}'.")
    return df_unificado

def entrenar_o_cargar_modelo_unificado(X_train, y_train):
    """
    Entrena o carga un modelo basado en el dataset unificado.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        print(f"Cargando modelo y escalador unificados existentes...")
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return model, scaler, False

    print("Entrenando nuevo modelo unificado...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam',
        max_iter=1500, random_state=42, early_stopping=True,
        n_iter_no_change=20, learning_rate_init=0.001, alpha=0.0001
    )

    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"Modelo unificado guardado en '{MODEL_FILE}'.")
    return model, scaler, True

def predecir_interactivo(model, scaler, columns):
    """
    Permite al usuario introducir datos de una mezcla (sin la edad) y genera una tabla
    comparativa de la evolución de la resistencia a diferentes edades de curado.
    """
    print("\n" + "="*60)
    print("PREDICCIÓN INTERACTIVA DE EVOLUCIÓN DE RESISTENCIA")
    print("="*60)
    print("Introduce los valores para tu mezcla (sin la edad de curado). Si un componente no se usa, introduce 0.")

    try:
        inputs = {}
        # Define las columnas que el usuario debe introducir
        input_columns = [col for col in columns if col != 'edad_curado_dias']

        for col in input_columns:
            inputs[col] = float(input(f"  {col.replace('_', ' ').title()}: "))

        # Define las edades de curado para la tabla, ahora hasta 90 días
        edades_curado = [7, 14, 30, 60, 90]

        # --- Preparar datos para la mezcla del usuario (puede tener o no PAC) ---
        mezcla_usuario_list = []
        for edad in edades_curado:
            fila = inputs.copy()
            fila['edad_curado_dias'] = edad
            mezcla_usuario_list.append(fila)
        df_usuario = pd.DataFrame(mezcla_usuario_list, columns=columns) # Asegurar orden

        # --- Preparar datos para la mezcla SIN PAC (control) ---
        inputs_sin_pac = inputs.copy()
        inputs_sin_pac['pac_kg_m3'] = 0
        inputs_sin_pac['activador_quimico_pct'] = 0 # Un control no lleva activador

        mezcla_sin_pac_list = []
        for edad in edades_curado:
            fila = inputs_sin_pac.copy()
            fila['edad_curado_dias'] = edad
            mezcla_sin_pac_list.append(fila)
        df_sin_pac = pd.DataFrame(mezcla_sin_pac_list, columns=columns)

        # --- Realizar predicciones ---
        predicciones_usuario = model.predict(scaler.transform(df_usuario)).clip(min=0)
        predicciones_sin_pac = model.predict(scaler.transform(df_sin_pac)).clip(min=0)

        # --- Calcular la mejora porcentual ---
        # Se añade un valor pequeño (epsilon) al denominador para evitar división por cero
        epsilon = 1e-6
        mejora_pct = ((predicciones_usuario - predicciones_sin_pac) / (predicciones_sin_pac + epsilon)) * 100

        # --- Mostrar resultados en una tabla ---
        resultados_df = pd.DataFrame({
            'Edad de Curado (días)': edades_curado,
            'Resistencia Mezcla Usuario (MPa)': predicciones_usuario,
            'Resistencia Control sin PAC (MPa)': predicciones_sin_pac,
            'Mejora/Reducción (%)': mejora_pct
        })

        print("\n-----------------------------------------------------------------")
        print("RESULTADO DE LA PREDICCIÓN DE EVOLUCIÓN DE RESISTENCIA")
        print(resultados_df.round(2).to_string(index=False))
        print("-----------------------------------------------------------------")
        print("\n* 'Mezcla Usuario' corresponde a los datos que introdujiste.")
        print("* 'Control sin PAC' es la misma mezcla pero sin PAC ni activador químico.")

    except ValueError:
        print("\nError: Por favor, introduce solo valores numéricos.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    # Flujo principal del programa
    df_uci_data = cargar_y_preparar_datos_uci()

    if df_uci_data is not None:
        df_pac_data = generar_datos_sinteticos_pac()
        df_final = unificar_datasets(df_uci_data, df_pac_data)

        X = df_final.drop(['resistencia_compresion_mpa', 'fuente_dato'], axis=1)
        y = df_final['resistencia_compresion_mpa']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_final['fuente_dato'])

        modelo, escalador, fue_entrenado = entrenar_o_cargar_modelo_unificado(X_train, y_train)

        # --- Bucle para predicción interactiva ---
        feature_columns = X.columns
        while True:
            predecir_interactivo(modelo, escalador, feature_columns)
            continuar = input("\n¿Deseas predecir otra mezcla? (s/n): ").lower()
            if continuar != 's':
                break

        print("\nPrograma finalizado.")
