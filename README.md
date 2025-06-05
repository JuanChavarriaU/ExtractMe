# ExtractMe

ExtractMe es una herramienta para extraer tablas de archivos PDF mediante el modelo TATR (TAble TRansformer) de Microsoft. Se compone por un backend en **Flask** y una interfaz en **Streamlit**.

## Instalación con pip
> Recomendación usar uv.

1. Crea y activa un entorno virtual de Python.
2. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt           # dependencias para la interfaz Streamlit
pip install -r requirements_backend.txt   # dependencias para la API
```
## Instalación con uv

1. Instala uv si no lo tienes.

- MacOS/Linux: ```curl -LsSf https://astral.sh/uv/install.sh | sh```
- Windows: ```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

2. instala las dependencias.
   
```bash
uv pip install -r requirements.txt
uv pip install -r requirements_backend.txt 
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

2. Abre otra terminal, y ejecuta la aplicación Streamlit:

```bash
streamlit run app.py
```

Al abrir la interfaz podrás subir un PDF y descargar un archivo ZIP con los CSV resultantes.

## Estructura

- `api.py`: API Flask encargada de procesar los PDFs.
- `app.py`: Interfaz de usuario en Streamlit para subir archivos.
- `tableExtraction.py`: Lógica de detección de tablas y OCR.
- `requirements*.txt`: Archivos con las dependencias.

