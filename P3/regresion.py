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
#modelos para la regresión
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR

#para medir como de buena es la predicción
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

#para medir el tiempo que tarda cada modelo
from time import time


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
            old_y = datos[:, datos.shape[1]-1:datos.shape[1]]
            
            
            for i in old_y:
                y.append(i[0])
        
    return x, y

x_reg,y_reg = readData('datos/regresion/train.csv','csv')

#divido la muestra total en 2 muestras una para el training y otra para el test
#Para ello crearemos 2 listas de indices que se barajaran y en principio 
#seleccionaremos el 80% de la muestra para el training
#fijo una semilla para que me de siempre los mismos resultados
np.random.seed(30)#la semilla se ha elegido aleatoriamente
IS_reg = np.random.permutation(x_reg.shape[0])

#calculamos el numero de datos en la muestra el 80%
n_training_reg = int(0.8*IS_reg.size)
        
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


x_training_reg = np.array(x_training_reg)
y_training_reg = np.array(y_training_reg)
x_test_reg = np.array(x_test_reg)
y_test_reg = np.array(y_test_reg)

#normalizamos los datos

scaler = StandardScaler()

#normalizamos los datos del training y test de regresión
x_training_reg = scaler.fit_transform(x_training_reg)
x_test_reg = scaler.fit_transform(x_test_reg)

#normalizamos todos los datos para posteriormente calcular el Eout
x_reg = scaler.fit_transform(x_reg)

print("Numero de datos para el training (regresión): ",y_training_reg.size)
print("Numero de datos para el test (regresion): ",y_test_reg.size)


#MOSTAR GRAFICA 2D de los datos para la regresión
#transformamos los datos de kd a 2d
pca = PCA(n_components=2)#indicamos la dimensionalidad
#transformamos los datos
principalComponents2 = pca.fit_transform(x_training_reg)
#creamos la estructura para una visualización correcta de los datos en 2D
principalXlabel2 = pd.DataFrame(data = principalComponents2, columns = ['componente 1', 'componente 2'])
#esto es para la etiqueta
pca = PCA(n_components=1)#indicamos la dimensionalidad
#y creamos la estructura con el mismo formato
principalYlabel2 = pd.DataFrame(data = y_training_reg, columns = ['target'])

#concatenamos los datos y las etiquetas
finalDf2 = pd.concat([principalXlabel2, principalYlabel2], axis=1)
# print('DATOS PARA LA REGRESIÓN')
# print(finalDf2)

#Visualización grafica,
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_title('Datos Training para la Regresión con anomalias')
ax.scatter(finalDf2['componente 1'],finalDf2['componente 2'],c=finalDf2['target'])
fig.show()

#Eliminamos los datos que son mayores de 15 en la componente 1
#ya que se consideran datos ruidosos
for i in np.arange(finalDf2['componente 2'].size):
    if(finalDf2['componente 2'][i]>15):
        np.delete(x_training_reg,i,axis=0)
        np.delete(y_training_reg,i)
        
#tambien lo elimino del dataFrame para la visualización correcta de los datos
finalDf2 = finalDf2.drop(finalDf2[finalDf2['componente 2']>15].index)

#Visualización grafica,
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_title('Datos Training para la Regresión sin anomalias aparentemente')
ax.scatter(finalDf2['componente 1'],finalDf2['componente 2'],c=finalDf2['target'])
fig.show()

input("\n--- Pulsar tecla para continuar ---\n")

#REGRESIÓN -------------------------------------------------------------------

print("REGRESIÓN-------------------------------------------------------------")

#LASSO
start_time = time()
lasso = Lasso()
#aplicamos cross validation con 5 fold
scores = abs(cross_val_score(lasso, x_training_reg, y_training_reg, cv=5, scoring='neg_root_mean_squared_error'))
#creamos nuestro modelo
logistic = lasso.fit(x_training_reg, y_training_reg.ravel())
#clasificamos el conjunto test con nuestro modelo
predicted = logistic.predict(x_test_reg)
elapsed_time = time() - start_time

#Mostrarmos los resultados
print("Lasso:")
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print("\tEcvs:\n\t\tEcv1: ",scores[0],"\n\t\tEcv2: ",scores[1])
print("\t\tEcv3: ",scores[2],"\n\t\tEcv4: ",scores[3],"\n\t\tEcv5: ",scores[4])
print("\tEcv media: ",(np.mean(scores)))
print("\tEin: ",mean_squared_error(y_training_reg,logistic.predict(x_training_reg), squared=False))
print("\tEtest: ",mean_squared_error(y_test_reg,predicted, squared=False))
print("\tEout: ",mean_squared_error(y_reg,logistic.predict(x_reg), squared=False))


input("\n--- Pulsar tecla para continuar ---\n")

#RIDGE
start_time = time()
ridge = Ridge(alpha=0.00001)
#aplicamos cross validation con 5 fold
scores = abs(cross_val_score(ridge, x_training_reg, y_training_reg, cv=5, scoring='neg_root_mean_squared_error'))
#creamos nuestro modelo
logistic = ridge.fit(x_training_reg, y_training_reg.ravel())
#clasificamos el conjunto test con nuestro modelo
predicted = logistic.predict(x_test_reg)
elapsed_time = time() - start_time

#Mostrarmos los resultados
print("Ridge:")
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print("\tEcvs:\n\t\tEcv1: ",scores[0],"\n\t\tEcv2: ",scores[1])
print("\t\tEcv3: ",scores[2],"\n\t\tEcv4: ",scores[3],"\n\t\tEcv5: ",scores[4])
print("\tEcv media: ",(np.mean(scores)))
print("\tEin: ",mean_squared_error(y_training_reg,logistic.predict(x_training_reg), squared=False))
print("\tEtest: ",mean_squared_error(y_test_reg,predicted, squared=False))
print("\tEout: ",mean_squared_error(y_reg,logistic.predict(x_reg), squared=False))


input("\n--- Pulsar tecla para continuar ---\n")

#Regresion Lineal
start_time = time()
lr = LinearRegression()
#aplicamos cross validation con 5 fold
scores = abs(cross_val_score(lr, x_training_reg, y_training_reg, cv=5, scoring='neg_root_mean_squared_error'))
#creamos nuestro modelo
logistic = lr.fit(x_training_reg, y_training_reg.ravel())
#clasificamos el conjunto test con nuestro modelo
predicted = logistic.predict(x_test_reg)
elapsed_time = time() - start_time

# #Mostrarmos los resultados
print("Regresion Lineal:")
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print("\tEcvs:\n\t\tEcv1: ",scores[0],"\n\t\tEcv2: ",scores[1])
print("\t\tEcv3: ",scores[2],"\n\t\tEcv4: ",scores[3],"\n\t\tEcv5: ",scores[4])
print("\tEcv media: ",(np.mean(scores)))
print("\tEin: ",mean_squared_error(y_training_reg,logistic.predict(x_training_reg), squared=False))
print("\tEtest: ",mean_squared_error(y_test_reg,predicted, squared=False))
print("\tEout: ",mean_squared_error(y_reg,logistic.predict(x_reg), squared=False))


input("\n--- Pulsar tecla para continuar ---\n")

#Linear Support Vector Regression
start_time = time()
svr = LinearSVR(epsilon=0.5, loss='squared_epsilon_insensitive', max_iter=10000)
#aplicamos cross validation con 5 fold
scores = abs(cross_val_score(svr, x_training_reg, y_training_reg, cv=5, scoring='neg_root_mean_squared_error'))
#creamos nuestro modelo
logistic = svr.fit(x_training_reg, y_training_reg.ravel())
#clasificamos el conjunto test con nuestro modelo
predicted = logistic.predict(x_test_reg)
elapsed_time = time() - start_time

#Mostrarmos los resultados
print("Linear Support Vector Regression:")
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print("\tEcvs:\n\t\tEcv1: ",scores[0],"\n\t\tEcv2: ",scores[1])
print("\t\tEcv3: ",scores[2],"\n\t\tEcv4: ",scores[3],"\n\t\tEcv5: ",scores[4])
print("\tEcv media: ",(np.mean(scores)))
print("\tEin: ",mean_squared_error(y_training_reg,logistic.predict(x_training_reg), squared=False))
print("\tEtest: ",mean_squared_error(y_test_reg,predicted, squared=False))
print("\tEout: ",mean_squared_error(y_reg,logistic.predict(x_reg), squared=False))


input("\n--- Pulsar tecla para continuar ---\n")

#SGD Regressor
start_time = time()
sgdr = SGDRegressor(alpha=0.00000000001)
#aplicamos cross validation con 5 fold
scores = abs(cross_val_score(sgdr, x_training_reg, y_training_reg, cv=5, scoring='neg_root_mean_squared_error'))
#creamos nuestro modelo
logistic = sgdr.fit(x_training_reg, y_training_reg.ravel())
#clasificamos el conjunto test con nuestro modelo
predicted = logistic.predict(x_test_reg)
elapsed_time = time() - start_time

#Mostrarmos los resultados
print("SGD Regressor:")
print("Calculo Elapsed time: %0.10f seconds" %elapsed_time)
print("\tEcvs:\n\t\tEcv1: ",scores[0],"\n\t\tEcv2: ",scores[1])
print("\t\tEcv3: ",scores[2],"\n\t\tEcv4: ",scores[3],"\n\t\tEcv5: ",scores[4])
print("\tEcv media: ",(np.mean(scores)))
print("\tEin: ",mean_squared_error(y_training_reg,logistic.predict(x_training_reg), squared=False))
print("\tEtest: ",mean_squared_error(y_test_reg,predicted, squared=False))
print("\tEout: ",mean_squared_error(y_reg,logistic.predict(x_reg), squared=False))

