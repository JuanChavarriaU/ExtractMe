# ExtractMe

ExtractMe es una herramienta para extraer tablas de archivos PDF mediante modelos de reconocimiento y OCR. Incluye un backend en **Flask** y una interfaz en **Streamlit**.

## Instalación

1. Crea y activa un entorno virtual de Python.
2. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt           # dependencias para la interfaz Streamlit
pip install -r requirements_backend.txt   # dependencias para la API
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto e indica la dirección del backend:

```bash
UPLOAD_ENDPOINT=http://localhost:5000/upload
```

## Uso

1. Inicia la API ejecutando:

```bash
python api.py
```

2. En otra terminal, lanza la aplicación Streamlit:

```bash
streamlit run app.py
```

Al abrir la interfaz podrás subir un PDF y descargar un archivo ZIP con los CSV resultantes.

## Estructura

- `api.py`: API Flask encargada de procesar los PDFs.
- `app.py`: Interfaz de usuario en Streamlit para subir archivos.
- `tableExtraction.py`: Lógica de detección de tablas y OCR.
- `requirements*.txt`: Archivos con las dependencias.

