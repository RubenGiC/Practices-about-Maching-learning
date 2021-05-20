#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 08:58:04 2021

@author: Ruben Girela Castellón
"""
import numpy as np
#para abrir el archivo csv
import csv
#para normalizar los datos
from sklearn.preprocessing import Normalizer

# para visualizar los datos en una grafica necesitamos:
import pandas as pd # crea el formato 2D
from sklearn.decomposition import PCA #para la dimensionalidad
import matplotlib.pyplot as plt #donde visualiza la grafica


#cargo los datos de los archivos
def readData(archivo, tipo):
    y = []
    x = []
    count = 0
    datos = []
    
    #open the file
    with open(archivo,'r') as f:
        #open file txt
        if(tipo == "txt"):
            #read each line of the file
            for line in f:
                data = []
                #read each word
                for word in line.split():
                    #the last word is the label
                    if count == 48:
                        y.append(word)
                    #     x.append(data)
                    # else:
                    #     data.append(word)
                    else:
                        data.append(word)
    
                    count += 1
                count = 0
                datos.append(data)
            datos = np.array(datos, dtype='f4')
            
            #normalizamos los datos
            normalizer = Normalizer().fit(datos)
            datos = normalizer.transform(datos)
            
            #Y separamos los datos de las etiquetas
            x = datos[:, :-1]
            y = np.array(y, dtype='f4') #datos[:, datos.shape[1]-1:datos.shape[1]]
            
        #read file csv
        else:
            #csv_read = csv.reader(f,delimiter='\t')
            csv_read = csv.reader(f)
            for row in csv_read:
                data = []
                if count > 0:
                    for word in row:
                        #print(word)
                        data.append(float(word))
                    datos.append(data)
                count +=1
                
            #print(datos[0])
            #indicamos el formato de los datos de tipo float con 4 decimales
            datos = np.array(datos, dtype='f4')
            
            #normalizamos los datos
            normalizer = Normalizer().fit(datos)
            datos = normalizer.transform(datos)
            
            #Y separamos los datos de las etiquetas
            x = datos[:, :-1]
            y = datos[:, datos.shape[1]-1:datos.shape[1]]
        
    return x, y

x_class,y_class = readData('datos/clasificacion/Sensorless_drive_diagnosis.txt','txt')
x_reg,y_reg = readData('datos/regresion/train.csv','csv')

print(y_class.size, " - ", x_class[:,1].size)
print(y_class)

#MOSTAR GRAFICA 2D de los datos de la clasificación
#transformamos los datos de kd a 2d
pca = PCA(n_components=2)#indicamos la dimensionalidad
#transformamos los datos
principalComponents = pca.fit_transform(x_class)
#creamos la estructura para una visualización correcta de los datos en 2D
principalXlabel = pd.DataFrame(data = principalComponents, columns = ['componente 1', 'componente 2'])
#esto es para la etiqueta
pca = PCA(n_components=1)#indicamos la dimensionalidad
#y creamos la estructura con el mismo formato
principalYlabel = pd.DataFrame(data = y_class, columns = ['target'])

#concatenamos los datos y las etiquetas
finalDf = pd.concat([principalXlabel, principalYlabel], axis=1)

print('DATOS PARA LA CLASIFICACIÓN')
print(finalDf)

#Visualización grafica,
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_title('Datos para la Clasificación')
ax.scatter(finalDf['componente 1'],finalDf['componente 2'],c=finalDf['target'])
fig.show()


#MOSTAR GRAFICA 2D de los datos para la regresión
#transformamos los datos de kd a 2d
pca = PCA(n_components=2)#indicamos la dimensionalidad
#transformamos los datos
principalComponents = pca.fit_transform(x_reg)
#creamos la estructura para una visualización correcta de los datos en 2D
principalXlabel = pd.DataFrame(data = principalComponents, columns = ['componente 1', 'componente 2'])
#esto es para la etiqueta
pca = PCA(n_components=1)#indicamos la dimensionalidad
#y creamos la estructura con el mismo formato
principalYlabel = pd.DataFrame(data = y_reg, columns = ['target'])

#concatenamos los datos y las etiquetas
finalDf = pd.concat([principalXlabel, principalYlabel], axis=1)
print('DATOS PARA LA REGRESIÓN')
print(finalDf)

#Visualización grafica,
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_title('Datos para la Regresión')
ax.scatter(finalDf['componente 1'],finalDf['componente 2'],c=finalDf['target'])
fig.show()





























