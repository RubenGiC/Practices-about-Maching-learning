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
from sklearn.preprocessing import StandardScaler

# para visualizar los datos en una grafica necesitamos:
import pandas as pd # crea el formato 2D
from sklearn.decomposition import PCA #para la dimensionalidad
import matplotlib.pyplot as plt #donde visualiza la grafica
#modelos lineales
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

#para medir como de buena es la predicción
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#para eliminar las anomalias de los datos para la clasificación
from sklearn.neighbors import LocalOutlierFactor


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
            
            #Y separamos los datos de las etiquetas
            x = datos
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
            
            #Y separamos los datos de las etiquetas
            x = datos[:, :-1]
            y = datos[:, datos.shape[1]-1:datos.shape[1]]
        
    return x, y

x_class,y_class = readData('datos/clasificacion/Sensorless_drive_diagnosis.txt','txt')
x_reg,y_reg = readData('datos/regresion/train.csv','csv')

#divido la muestra total en 2 muestras una para el training y otra para el test
#Para ello crearemos 2 listas de indices que se barajaran y en principio 
#seleccionaremos el 80% de la muestra para el training
IS_class = np.random.permutation(x_class.shape[0])
IS_reg = np.random.permutation(x_reg.shape[0])

#calculamos el numero de datos en la muestra el 80%
n_training_class = int(0.8*IS_class.size)
n_training_reg = int(0.8*IS_reg.size)
# print(n_training_class)
# print(n_training_reg)

x_training_class = []
y_training_class = []

x_test_class = []
y_test_class = []

#Y seleccionamos los datos para el training
for i in np.arange(IS_class.size):
    if(i < n_training_class):
        x_training_class.append(x_class[IS_class[i]])
        y_training_class.append(y_class[IS_class[i]])
    else:
        x_test_class.append(x_class[IS_class[i]])
        y_test_class.append(y_class[IS_class[i]])
        
x_training_reg = []
y_training_reg = []

x_test_reg = []
y_test_reg = []

#Y seleccionamos los datos para el training
for i in np.arange(IS_reg.size):
    if(i < n_training_reg):
        x_training_reg.append(x_reg[IS_reg[i]])
        y_training_reg.append(y_reg[IS_reg[i]])
    else:
        x_test_reg.append(x_reg[IS_reg[i]])
        y_test_reg.append(y_reg[IS_reg[i]])

x_training_class = np.array(x_training_class)
y_training_class = np.array(y_training_class)
x_test_class = np.array(x_test_class)
y_test_class = np.array(y_test_class)

x_training_reg = np.array(x_training_reg)
y_training_reg = np.array(y_training_reg)
x_test_reg = np.array(x_test_reg)
y_test_reg = np.array(y_test_reg)

#normalizamos los datos

scaler = StandardScaler()

x_training_class = scaler.fit_transform(x_training_class)
x_test_class = scaler.fit_transform(x_test_class)

x_training_reg = scaler.fit_transform(x_training_reg)
x_test_reg = scaler.fit_transform(x_test_reg)


print("Numero de datos para el training (clasificacion): ",y_training_class.size)
print("Numero de datos para el test (clasificacion): ",y_test_class.size)
print("Numero de datos para el training (regresión): ",y_training_reg.size)
print("Numero de datos para el test (regresion): ",y_test_reg.size)




#MOSTAR GRAFICA 2D de los datos de la clasificación
#transformamos los datos de kd a 2d
pca = PCA(n_components=2)#indicamos la dimensionalidad
#transformamos los datos
principalComponents = pca.fit_transform(x_training_class)
#creamos la estructura para una visualización correcta de los datos en 2D
principalXlabel = pd.DataFrame(data = principalComponents, columns = ['componente 1', 'componente 2'])
#esto es para la etiqueta
pca = PCA(n_components=1)#indicamos la dimensionalidad
#y creamos la estructura con el mismo formato
principalYlabel = pd.DataFrame(data = y_training_class, columns = ['target'])

#concatenamos los datos y las etiquetas
finalDf = pd.concat([principalXlabel, principalYlabel], axis=1)

print('DATOS PARA LA CLASIFICACIÓN CON ANOMALIAS')
#print(finalDf)

for i in finalDf['componente 1']:
    if(i>20):
        print(i)

#Visualización grafica,
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_title('Datos Training para la Clasificación')
ax.scatter(finalDf['componente 1'],finalDf['componente 2'],c=finalDf['target'])
fig.show()


#------> APLICAR LocalOutlierFactor PARA QUITAR ESAS ANOMALIAS

clf = LocalOutlierFactor()

# #MOSTAR GRAFICA 2D de los datos de la clasificación
# #transformamos los datos de kd a 2d
# pca = PCA(n_components=2)#indicamos la dimensionalidad
# #transformamos los datos
# principalComponents = pca.fit_transform(x_training_class)
# #creamos la estructura para una visualización correcta de los datos en 2D
# principalXlabel = pd.DataFrame(data = principalComponents, columns = ['componente 1', 'componente 2'])
# #esto es para la etiqueta
# pca = PCA(n_components=1)#indicamos la dimensionalidad
# #y creamos la estructura con el mismo formato
# principalYlabel = pd.DataFrame(data = y_training_class, columns = ['target'])

# #concatenamos los datos y las etiquetas
# finalDf = pd.concat([principalXlabel, principalYlabel], axis=1)

# print('DATOS PARA LA CLASIFICACIÓN CON ANOMALIAS')
# #print(finalDf)

# for i in finalDf['componente 1']:
#     if(i>20):
#         print(i)

# #Visualización grafica,
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('componente 1')
# ax.set_ylabel('componente 2')
# ax.set_title('Datos Training para la Clasificación')
# ax.scatter(finalDf['componente 1'],finalDf['componente 2'],c=finalDf['target'])
# fig.show()


# #MOSTAR GRAFICA 2D de los datos para la regresión
# #transformamos los datos de kd a 2d
# pca = PCA(n_components=2)#indicamos la dimensionalidad
# #transformamos los datos
# principalComponents = pca.fit_transform(x_training_reg)
# #creamos la estructura para una visualización correcta de los datos en 2D
# principalXlabel = pd.DataFrame(data = principalComponents, columns = ['componente 1', 'componente 2'])
# #esto es para la etiqueta
# pca = PCA(n_components=1)#indicamos la dimensionalidad
# #y creamos la estructura con el mismo formato
# principalYlabel = pd.DataFrame(data = y_training_reg, columns = ['target'])

# #concatenamos los datos y las etiquetas
# finalDf = pd.concat([principalXlabel, principalYlabel], axis=1)
# print('DATOS PARA LA REGRESIÓN')
# print(finalDf)

# #Visualización grafica,
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('componente 1')
# ax.set_ylabel('componente 2')
# ax.set_title('Datos Training para la Regresión')
# ax.scatter(finalDf['componente 1'],finalDf['componente 2'],c=finalDf['target'])
# fig.show()

# input("\n--- Pulsar tecla para continuar ---\n")



#RidgeClassifier
#creamos nuestro modelo
logistic = RidgeClassifier().fit(x_training_class, y_training_class)
#lo probamos con el conjunto de test
predicted = logistic.predict(x_test_class)
#y comprobamos como de buenos son los resultados
result = classification_report(y_test_class,predicted)

#Mostrarmos los resultados
print("RidgeClassifier:")
print("\tEin: ",accuracy_score(y_training_class, logistic.predict(x_training_class)))
print("\tEtest: ",accuracy_score(y_test_class, logistic.predict(x_test_class)))
print("\tEout: ",accuracy_score(y_class, logistic.predict(x_class)))
print("\tPrecisión media training: ",logistic.score(x_training_class,y_training_class))
print("\tPrecisión media test: ",logistic.score(x_test_class,y_test_class))
print("\tTabla resultados:\n",result)

# input("\n--- Pulsar tecla para continuar ---\n")


# #Regresión Logistica
# #creamos nuestro modelo
# logistic = LogisticRegression(max_iter=100000).fit(x_training_class, y_training_class.ravel())
# #clasificamos el conjunto test con nuestro modelo
# predicted = logistic.predict(x_test_class)
# #y comprobamos como de buenos son los resultados
# result = classification_report(y_test_class,predicted)

# #Mostrarmos los resultados
# print("Regresion Logistica:")
# print("\tEin: ",accuracy_score(y_training_class, logistic.predict(x_training_class)))
# print("\tEtest: ",accuracy_score(y_test_class, logistic.predict(x_test_class)))
# print("\tEout: ",accuracy_score(y_class, logistic.predict(x_class)))
# print("\tPrecisión media training: ",logistic.score(x_training_class,y_training_class))
# print("\tPrecisión media test: ",logistic.score(x_test_class,y_test_class))
# print("\tTabla resultados:\n",result)

# input("\n--- Pulsar tecla para continuar ---\n")

# #Perceptron
# #creamos nuestro modelo
# logistic = Perceptron().fit(x_training_class, y_training_class.ravel())
# #clasificamos el conjunto test con nuestro modelo
# predicted = logistic.predict(x_test_class)
# #y comprobamos como de buenos son los resultados
# result = classification_report(y_test_class,predicted)

# #Mostrarmos los resultados
# print("Perceptron:")
# print("\tEin: ",accuracy_score(y_training_class, logistic.predict(x_training_class)))
# print("\tEtest: ",accuracy_score(y_test_class, logistic.predict(x_test_class)))
# print("\tEout: ",accuracy_score(y_class, logistic.predict(x_class)))
# print("\tPrecisión media training: ",logistic.score(x_training_class,y_training_class))
# print("\tPrecisión media test: ",logistic.score(x_test_class,y_test_class))
# print("\tTabla resultados:\n",result)

# input("\n--- Pulsar tecla para continuar ---\n")

# #SGD (STOCHASTIC GRADIENT DESCENT)
# #creamos nuestro modelo
# logistic = SGDClassifier().fit(x_training_class, y_training_class.ravel())
# #clasificamos el conjunto test con nuestro modelo
# predicted = logistic.predict(x_test_class)
# #y comprobamos como de buenos son los resultados
# result = classification_report(y_test_class,predicted)

# #Mostrarmos los resultados
# print("SGD (Gradiente Descendiente Estocástico):")
# print("\tEin: ",accuracy_score(y_training_class, logistic.predict(x_training_class)))
# print("\tEtest: ",accuracy_score(y_test_class, logistic.predict(x_test_class)))
# print("\tEout: ",accuracy_score(y_class, logistic.predict(x_class)))
# print("\tPrecisión media training: ",logistic.score(x_training_class,y_training_class))
# print("\tPrecisión media test: ",logistic.score(x_test_class,y_test_class))
# print("\tTabla resultados:\n",result)

# input("\n--- Pulsar tecla para continuar ---\n")















