
# API de Predicción de Alquileres de Monopatines

Este proyecto consiste en una API construida con FastAPI que utiliza un modelo de machine learning para predecir el número total de alquileres de bicicletas basado en diversas características. También incluye una interfaz sencilla para interactuar con la API.

## Descripción

La API expone un endpoint `/predict` que recibe datos en formato JSON con las características necesarias para la predicción y devuelve el número total de alquileres predicho. Además, se incluye un endpoint raíz `/` para verificar que la API está funcionando.

## Tecnologías Utilizadas

* **Python:** Lenguaje de programación principal.
* **FastAPI:** Framework web moderno y de alto rendimiento para construir APIs.
* **Uvicorn:** Servidor ASGI para ejecutar la aplicación FastAPI.
* **Pandas:** Librería para manipulación y análisis de datos.
* **Scikit-learn:** Librería de machine learning para Python (utilizada para el pipeline del modelo).
* **Joblib:** Librería para serializar y deserializar objetos de Python (utilizada para guardar y cargar el modelo).
* **XGBoost:** Librería de gradient boosting (si tu modelo lo utiliza).
* **NumPy:** Librería para computación numérica.

## Estructura del Proyecto

```plaintext
api_predict/
├── app.py                  # Código principal de la API (FastAPI)
├── models/                 # Carpeta que contiene el modelo de machine learning serializado
│   └── ml_pipeline.joblib
├── static/                 # Carpeta para archivos estáticos (CSS)
│   └── style.css
├── index.html              # Interfaz de usuario HTML
├── requirements.txt        # Lista de dependencias de Python
└── README.md               # Este archivo
```

## Instalación

1.  **Clona el repositorio (si lo tienes en un repositorio Git):**

    ```bash
    git clone <URL_del_repositorio>
    cd <nombre_del_proyecto>
    ```

2.  **Crea un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # En Linux/macOS
    venv\Scripts\activate.bat  # En Windows
    ```

3.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

## Ejecución Local

1.  **Ejecuta la API con Uvicorn:**

    ```bash
    uvicorn app:app --reload
    ```

    Esto iniciará el servidor en `http://127.0.0.1:8000`.

2.  **Abre la interfaz de usuario:**

    Abre `index.html` en tu navegador web (generalmente haciendo doble clic en el archivo). Podrás interactuar con la API a través de esta interfaz.

## Uso de la API

### Endpoint `/` (GET)

Devuelve un mensaje indicando que la API está funcionando.

### Endpoint `/predict` (POST)

Recibe un objeto JSON con las siguientes características y devuelve la predicción del total de alquileres.

**Ejemplo de cuerpo de la petición JSON:**

```json
{
    "temporada": 1,
    "anio": 0,
    "mes": 1,
    "hora": 0,
    "feriado": 0,
    "dia_trabajo": 0,
    "clima": 1,
    "temperatura": 0.24,
    "sensacion_termica": 0.2879,
    "humedad": 0.81,
    "velocidad_viento": 0.0,
    "dia_semana": 6
}
```

Ejemplo de respuesta JSON:

```json
{
    "total_alquileres_predicho": 42.5
}
```

Puedes interactuar con este endpoint usando herramientas como curl, Postman o la interfaz de usuario proporcionada (index.html).

## Despliegue

Este proyecto está diseñado para ser desplegado en plataformas como Render. Asegúrate de tener un archivo requirements.txt con todas las dependencias necesarias y configura el comando de inicio de la siguiente manera:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Asegúrate de que la ruta al modelo (`models/ml_pipeline.joblib`) sea correcta en el entorno de despliegue.

## Notas Adicionales

- Puedes personalizar la interfaz de usuario (`index.html` y `static/style.css`) para adaptarla a tus necesidades.
- Para mejorar la API, considera agregar validación de datos de entrada más robusta, manejo de errores más detallado y documentación (por ejemplo, usando Swagger UI con FastAPI).
- Si el modelo requiere un preprocesamiento específico, asegúrate de que este se realice correctamente dentro de la función `predict_rentals` en `app.py`.

## Autor

Adrian F. Acarapi Roca
