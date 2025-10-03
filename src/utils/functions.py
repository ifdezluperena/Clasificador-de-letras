import os
import cv2
import numpy as np


def x_generator(path, width, height):
    """
    Carga todas las imágenes de un directorio, las redimensiona y devuelve una lista de arrays.

    Args:
        path (str): Ruta del directorio que contiene las imágenes.
        width (int): Ancho al que se redimensionará cada imagen.
        height (int): Alto al que se redimensionará cada imagen.

    Returns:
        list[np.ndarray]: Lista de imágenes en formato NumPy array, todas con el mismo tamaño (width x height x 3).
    """

    x = []  # Lista donde se almacenarán las imágenes procesadas

    for i in os.listdir(path):  # Recorre todos los archivos del directorio
        img = cv2.imread(path + '/' + i)  # Carga la imagen
        img_resized = cv2.resize(img, (width, height))  # Redimensiona a (width, height)
        x.append(img_resized)  # Añade la imagen a la lista

    return x  # Devuelve la lista completa de imágenes



def labels_generator(path):
    """
    Genera una lista de etiquetas únicas a partir de los nombres de archivo en un directorio.
    Se asume que cada nombre de archivo tiene la forma 'algo_etiqueta_algo.png' 
    y que la etiqueta está en la segunda posición tras hacer split('_').

    Args:
        path (str): Ruta del directorio que contiene los archivos.

    Returns:
        list[str]: Lista de etiquetas únicas extraídas de los nombres de archivo.
    """

    labels = []  # Lista para almacenar las etiquetas únicas

    for i in os.listdir(path):  # Recorre todos los archivos del directorio
        j = i.split('_')[1]  # Extrae la parte que corresponde a la etiqueta
        if j not in labels:  # Evita duplicados
            labels.append(j)

    return labels  # Devuelve la lista de etiquetas



def y_generator(path, labels):
    """
    Genera los vectores one-hot correspondientes a las etiquetas de los archivos 
    en un directorio, según una lista de etiquetas conocidas.

    Se asume que el nombre de cada archivo contiene un guion bajo ('_') y que 
    la etiqueta está en la segunda posición al hacer split('_').

    Args:
        path (str): Ruta del directorio que contiene los archivos.
        labels (list[str]): Lista de etiquetas posibles, en el orden deseado 
                            para la codificación one-hot.

    Returns:
        list[np.ndarray]: Lista de vectores one-hot, uno por cada archivo en el directorio.
    """

    y = []  # Lista que almacenará los vectores one-hot

    for i in os.listdir(path):  # Recorre todos los archivos del directorio
        j = i.split('_')[1]  # Extrae la etiqueta del nombre de archivo

        # Busca la posición de esa etiqueta dentro de la lista de labels
        for x, z in enumerate(labels):
            if z == j:
                arr = np.zeros(len(labels))  # Vector de ceros
                arr[x] = 1                  # Marca la posición de la etiqueta
                y.append(arr)               # Añade el vector a la lista
    return y  # Devuelve todos los vectores one-hot
