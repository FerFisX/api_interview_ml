import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path, test_size=0.2, random_state=None):
    """
    Carga el dataset, aplica correcciones específicas de valores faltantes (EDA),
    divide los datos usando train_test_split, y define el preprocesador (sin aplicarlo).

    Args:
        data_path (str): Ruta al archivo CSV del dataset.
        test_size (float): Proporción del dataset a incluir en el conjunto de prueba (entre 0.0 y 1.0).
        random_state (int): Semilla aleatoria para la división, para reproducibilidad.

    Returns:
        tuple: Contiene los conjuntos de entrenamiento y prueba para características (X_train, X_test)
               y la variable objetivo (y_train, y_test), y el preprocesador sin ajustar.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: El archivo '{data_path}' no fue encontrado.")
        return None, None, None, None, None
    # Correciones basadas en el EDA
    # Para la columna 'hora'
    df.loc[313:318, 'hora'] = range(13, 13 + 6)
    df.loc[599:603, 'hora'] = range(2, 2 + 5)

    # Para la columna 'dia_semana'
    df.loc[3:9, 'dia_semana'] = 6.0
    df.loc[128:133, 'dia_semana'] = 4.0

    # Para la columna 'total_alquileres'
    media_total_alquileres = df['total_alquileres'].mean()
    df['total_alquileres'] = df['total_alquileres'].fillna(media_total_alquileres)
    # --- Fin de las correcciones de valores faltantes ---

    # 1. Definir características (X) y variable objetivo (y)
    X = df.drop(columns=['indice', 'u_casuales', 'u_registrados', 'total_alquileres', 'fecha'])
    y = df['total_alquileres']

    # 2. Identificar columnas categóricas y numéricas
    categorical_features = ['temporada', 'clima', 'feriado', 'dia_trabajo', 'mes', 'dia_semana']
    numerical_features = ['anio', 'hora', 'temperatura', 'sensacion_termica', 'humedad', 'velocidad_viento']

    # 3. Crear el preprocesador utilizando ColumnTransformer (SIN AJUSTARLO)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. Dividir los datos en conjuntos de entrenamiento y prueba
    df['fecha'] = pd.to_datetime(df['fecha'])
    df = df.sort_values(by='fecha')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)

    return X_train, X_test, y_train, y_test, preprocessor

