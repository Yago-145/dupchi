![Logo](app/assets/dupchi_logo.png)

# 🧬 DUPCHI - Denoise, Upscale and Predict Cancer from Histopathological Images

Bienvenido al repositorio de DUPCHI, una aplicación avanzada de aprendizaje profundo destinada a la predicción de cáncer de colon y pulmón a partir de imágenes histopatológicas.

## 🤖 Modelos empleados

- **Autoencoder**: Utilizado para la eliminación de ruido en imágenes.
- **ESRGAN**: Empleada para el aumento de la resolución de imágenes.
- **ResNet152_v2**: Arquitectura de CNN para la predicción precisa del tipo de cáncer.

## ⚙️ Proceso Técnico

1. **Downsampling**: Las imágenes originales de 768x768 se reducen a 224x224.
2. **Adición de Ruido y Limpieza**: Se añade ruido artificialmente y se utiliza un autoencoder para eliminarlo.
3. **Upsampling**: Las imágenes se procesan a través de ESRGAN para mejorar la resolución.
4. **Predicción del Cáncer**: Finalmente, se usa una ResNet_v2 para determinar el tipo de cáncer en las imágenes.

## 🚀 Comenzando

Para usar DUPCHI, sigue estos pasos:

1. Clona el repositorio:
```bash
git clone https://github.com/Yago-145/dupchi.git
```
2. Crea un entorno con Anaconda:
```bash
conda create --name dupchi
```
3. Activa el entorno:
```bash
conda activate dupchi
```
4. Instala el proyecto:
```bash
pip install -e .
```
5. Instala Poetry:
```bash
pip install poetry
```
6. Ejecuta Poetry:
```bash
poetry install
```
7. Inicia la aplicación:
```bash
make app
```

## ⚠️ Advertencia

La aplicación está diseñada para procesar imágenes almacenadas en un bucket de GCP. Es necesario hacer login con una service account de Google Cloud Platform para testear la aplicación correctamente.

Asegúrate de tener configurado correctamente el acceso a GCP para evitar problemas de autenticación y acceso a los datos.

## 🎫 Licencia

`dupchi` ha sido creado por Javier Yago Córcoles. Tiene licencia según los términos de la licencia MIT.

## ©️ Créditos

`dupchi` creado a partir de [Angel Martinez-Tenor's Data Science Template](https://github.com/angelmtenor/ds-template), que a su vez fue desarrollado en base a la [py-pkgs-cookiecutter template](https://github.com/py-pkgs/py-pkgs-cookiecutter)

