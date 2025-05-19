Markdown

# API de Predicción de Alquileres de Bicicletas

Este proyecto consiste en una API construida con FastAPI que utiliza un modelo de machine learning para predecir el número total de alquileres de bicicletas basado en diversas características. También incluye una interfaz de usuario básica construida con HTML, CSS y JavaScript para interactuar fácilmente con la API.

## Descripción

La API expone un endpoint `/predict` que recibe datos en formato JSON con las características necesarias para la predicción y devuelve el número total de alquileres predicho. Además, se incluye una interfaz web sencilla para facilitar la prueba y demostración de la API.

## Tecnologías Utilizadas

* **Python:** Lenguaje de programación principal.
* **FastAPI:** Framework web moderno y de alto rendimiento para construir APIs.
* **Uvicorn:** Servidor ASGI para ejecutar la aplicación FastAPI.
* **Pandas:** Librería para manipulación y análisis de datos.
* **Scikit-learn:** Librería de machine learning para Python (utilizada para el pipeline del modelo).
* **Joblib:** Librería para serializar y deserializar objetos de Python (utilizada para guardar y cargar el modelo).
* **XGBoost:** Librería de gradient boosting (si tu modelo lo utiliza).
* **NumPy:** Librería para computación numérica.
* **HTML:** Lenguaje de marcado para la estructura de la interfaz de usuario.
* **CSS:** Hojas de estilo en cascada para el diseño de la interfaz de usuario.
* **JavaScript:** Lenguaje de programación para la interactividad en la interfaz de usuario.

## Estructura del Proyecto

tu_proyecto/
├── app.py                  # Código principal de la API (FastAPI)
├── models/                 # Carpeta que contiene el modelo de machine learning serializado
│   └── ml_pipeline.joblib
├── static/                 # Carpeta para archivos estáticos (CSS)
│   └── style.css
└── index.html              # Interfaz de usuario HTML
└── requirements.txt        # Lista de dependencias de Python
└── README.md               # Este archivo


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
Ejemplo de respuesta JSON:

JSON

{
    "total_alquileres_predicho": 42.5
}
Puedes interactuar con este endpoint usando herramientas como curl, Postman o la interfaz de usuario proporcionada (index.html).

Despliegue
Este proyecto está diseñado para ser desplegado en plataformas como Render. Asegúrate de tener un archivo requirements.txt con todas las dependencias necesarias y configura el comando de inicio de tu servicio web para que ejecute:

uvicorn app:app --host 0.0.0.0 --port $PORT
Asegúrate de que la ruta al modelo (models/ml_pipeline.joblib) sea correcta en el entorno de despliegue.

Notas Adicionales
Puedes personalizar la interfaz de usuario (index.html y static/style.css) para adaptarla a tus necesidades.
Para mejorar la API, considera agregar validación de datos de entrada más robusta, manejo de errores más detallado y documentación (por ejemplo, usando Swagger UI con FastAPI).
Si el modelo requiere un preprocesamiento específico, asegúrate de que este se realice correctamente dentro de la función predict_rentals en app.py.
Autor
[Tu Nombre o el Nombre de tu Organización]

Licencia
[Tu Licencia (opcional)]


**Cómo usar este archivo `README.md`:**

1.  Crea un archivo llamado `README.md` en la raíz de tu proyecto.
2.  Copia y pega el contenido de arriba en ese archivo.
3.  **Edita el archivo:**
    * Reemplaza los marcadores de posición (como `<URL_del_repositorio>`, `<nombre_del_proyecto>`, `[Tu Nombre o el Nombre de tu Organización]`, `[Tu Licencia (opcional)]`) con la información real de tu proyecto.
    * Añade cualquier otra sección o detalle que consideres importante para tu proyecto.
    * Si utilizaste otras librerías, asegúrate de incluirlas en la sección "Tecnologías Utilizadas" y en el ejemplo de `requirements.txt`.
    * Describe con más detalle cómo se entrenó el modelo y qué significan las características de entrada si lo consideras necesario.

Este archivo `README.md` proporcionará una buena visión general de tu proyecto para cualquier 
