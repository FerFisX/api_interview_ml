import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(data_path):
    """
    Carga el dataset, realiza un preprocesamiento básico (manejo de nulos ya realizado),
    codifica variables categóricas, divide los datos y escala las características numéricas.

    Args:
        data_path (str): Ruta al archivo CSV del dataset.

    Returns:
        tuple: Contiene los conjuntos de entrenamiento y prueba para características (X_train, X_test)
               y la variable objetivo (y_train, y_test), y el preprocesador ajustado.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: El archivo '{data_path}' no fue encontrado.")
        return None, None, None, None, None

    # 1. Definir características (X) y variable objetivo (y)
    X = df.drop(columns=['indice', 'u_casuales', 'u_registrados', 'total_alquileres', 'fecha'])
    y = df['total_alquileres']

    # 2. Identificar columnas categóricas y numéricas
    categorical_features = ['temporada', 'clima', 'feriado', 'dia_trabajo', 'mes', 'dia_semana']
    numerical_features = ['anio', 'hora', 'temperatura', 'sensacion_termica', 'humedad', 'velocidad_viento']

    # 3. Crear un preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. Dividir los datos en conjuntos de entrenamiento y prueba (considerando la naturaleza temporal)
    # Ordenamos los datos por fecha (si aún no lo están) y luego dividimos.
    # Para una división temporal simple, podemos tomar un porcentaje para entrenamiento y el resto para prueba.
    df['fecha'] = pd.to_datetime(df['fecha']) # Asegurarse de que la fecha sea datetime
    df = df.sort_values(by='fecha') # Asegurarse de que esté ordenado por fecha
    train_size = int(0.8 * len(df))
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    # 5. Ajustar y transformar los datos de entrenamiento y solo transformar los datos de prueba
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor
