import os
import cv2
import numpy as np


def x_generator(path,width,height):

    x = []

    for i in os.listdir(path):

        img = cv2.imread(path+'/'+i)
        img_resized = cv2.resize(img,(width,height))
        x.append(img_resized)
    return x

def labels_generator(path):

    labels = []

    for i in os.listdir(path):

        j = i.split('_')[1]

        if j not in labels:
            labels.append(j)
    return labels

def y_generator(path, labels):

    y = []

    for i in os.listdir(path):

        j = i.split('_')[1]

        for x,y in enumerate(labels):
            if y == j:
                arr = np.zeros(len(labels))
                arr[x] = 1
                y.append(arr)
    return y