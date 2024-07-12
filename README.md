# TFG-Diseño e implementación de un sistema de detección de objetos mediante vehículos aéreos no tripulados basado en modelos de aprendizaje federado

## 1. Introducción
Este repositorio contiene el código necesario para el Trabajo Final de Grado en Ingeniería y Sistemas de Datos de la Universidad Politécnica de Madrid. En este proyecto se realiza un sistema de detección de objetos sobre imagenes captadas por drones basado en modelos de aprendizaje federado.

## 2. Instalación del entorno
Para instalar las dependencias necesarias, ejecutar el siguiente comando:

```pip install -r requirements.txt```

## 3. Dataset
Descargar la versión `Task 1: Object Detection in Images` de VisDrone en: https://github.com/VisDrone/VisDrone-Dataset

## 4. Arquitectura centralizada
En la carpeta `/Pytorch` se encuentra el cuadernillo `Train_VisDrone.ipynb` de Jupyter necesario para el entrenamiento de los modelos. Dentro de este cuadernillo se puede seleccionar la red y los hiperparámetros asociados.

## 5. Arquitectura federada
Dentro de la carpeta `/Flower` se encuentran los scripts de Python para el servidor y los dos clientes, dentro de cada uno se pueden configurar sus parámetros.

Para iniciar el proceso de entrenamiento es necesario ejecutar el archivo `server_flower.py` ,y posteriormente `client_flower_1.py` y `client_flower_2.py` correspondiento a los clientes de Flower.

```
python server_flower.py
python client_flower_1.py
python client_flower_2.py
```

## 5. Evaluación de los modelos 
En la carpeta `/Pytorch` se encuentra el cuadernillo `Test_VisDrone.ipynb` de Jupyter necesario para el entrenamiento de los modelos, simplemente es necesario indicar el directorio del modelo selecionado.

## 6. Detección de objetos
La detección de objetos sobre imágenes se realiza ejecutando el script `/Detection/detection_image.py`.