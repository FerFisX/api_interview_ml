from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from src.data_processing import load_and_preprocess_data

def create_ml_pipeline():
    """
    Crea un pipeline de machine learning para la predicción de alquileres de monopatines.

    Returns:
        Pipeline: Un objeto Pipeline de scikit-learn que incluye el preprocesamiento
                  y un modelo XGBoost.
    """
    # 1. Definir las columnas categóricas y numéricas
    categorical_features = ['temporada', 'clima', 'feriado', 'dia_trabajo', 'mes', 'dia_semana']
    numerical_features = ['anio', 'hora', 'temperatura', 'sensacion_termica', 'humedad', 'velocidad_viento']

    # 2. Crear el preprocesador utilizando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 3. Crear el modelo XGBoost
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        random_state=42
    )

    # 4. Crear el pipeline combinando el preprocesador y el modelo
    ml_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', xgb_model)])

    return ml_pipeline

