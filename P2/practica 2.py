# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: 
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time


# Fijamos la semilla
np.random.seed(1)

#genera una lista de vectores con dimension dim y de tamaño N aleatoriamente
#de forma uniforme en los rangos rango[0] y rango[1]
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

#genera una lista de vectores de dimensión dim generadas aleatoriamente, 
#extrayendo de una distribución gausiana de media 0 y varianza sigma
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out

#simula una recta, de forma aleatoria, en los intervalos (intervalo[0], intervalo[1])
# y aplicando la función de esos puntos aleatorios y = ax + b
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

#simulate the point cloud randomly evelyn
x = simula_unif(50, 2, [-50,50])

#simulate the point cloud randomly using the Gaussian distributon
x_gaus = simula_gaus(50, 2, np.array([5,7]))

#Draw the graph
# fig1a = plt.figure()
# ax1a = fig1a.add_subplot()
# ax1a.scatter(x[:,0],x[:,1])
# ax1a.set_title('Nube de puntos aleatoria uniforme')

# fig1b = plt.figure()
# ax1b = fig1b.add_subplot()
# ax1b.scatter(x_gaus[:,0],x_gaus[:,1])
# ax1b.set_title('Nube de puntos aleatoria Gaussiana')

# input("\n--- Pulsar tecla para continuar ---\n")


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

#calculate miss clasification rate in %
def missClasificationRate(y_training,y_test):
    error = 0;
    for i in np.arange(y_test.size):
        if(y_training[i] != y_test[i]):
            error += 1
    return (error/y_test.size)*100

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
    if x >= 0:
        return 1
    return -1

def f(x, y, a, b):
 	return signo(y - a*x - b)

#generates the a and b values to calculate the tags of each point
a, b = simula_recta([-50,50])

#simulate point cloud
x = simula_unif(100,2,[-50,50])

#generates the tags
tag = np.zeros(x[:,0].size)
for i in np.arange(x[:,0].size):
    tag[i] = f(x[i,0], x[i,1], a, b)
    
    
# print("Porcentaje de error de clasificación (sin ruido): ",missClasificationRate(tag,tag),"%")
    

# #Draw the graph
# fig2a = plt.figure()
# ax2a = fig2a.add_subplot()
# ax2a.scatter(x[:,0],x[:,1],c=tag)
# #calculate the perfect parting line
# X = np.linspace(-50, 50, tag.size)
# #solves the function for the variable Y
# #f(x,y) = y - ax -b
# Y = (a*X) + b
# ax2a.plot(X, Y)
# ax2a.set_title('Nube de puntos aleatoria uniforme, ajuste perfecto (100)')


# input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

#tags with 10% noise to positive tags and 10% to negative tags
def noise(tags,percent):
    
    #copy the tags
    tags_noise = tags.copy()
    
    n_positive = 0
    n_negative = 0
    
    for i in tags:
        if(i>0):
            n_positive = n_positive + 1
        else:
            n_negative = n_negative + 1

    #max number of points to change positive and negative (10%)
    max_positive = int(n_positive*percent)
    max_negative = int(n_negative*percent)
    
    #print("POSITIVOS: ",n_positive,", NEGATIVOS: ",n_negative,", 10%: ",max_positive,", ",max_negative)
    
    #indices of sample with noise (random integer without repeating)
    ind_noise = np.random.choice(tags_noise.size, tags_noise.size, replace=False)
    #print(ind_noise)
    #count change positive and negative points
    i = 0
    n = 0
    p = 0
    
    #as long as 10% of the noise isn't reached for the negative and positive points
    while(p < max_positive or n < max_negative):
        
        #print("Indice: ",ind_noise[i],", value: ",tags_noise[ind_noise[i]])
        #change the vaule of the label if the vaule is positive and has not reached 10%
        if(tags_noise[ind_noise[i]]>0 and p < max_positive):
            #print("POSITIVE")
            tags_noise[ind_noise[i]]=-tags_noise[ind_noise[i]]#change the sign
            p = p + 1
        #change the vaule of the label if the vaule is negative and has not reached 10%
        elif(n < max_negative):
            #print("NEGATIVE")
            tags_noise[ind_noise[i]]=-tags_noise[ind_noise[i]]#change the sign
            n = n + 1
        i = i + 1
        
    #print(p,", ",n," vs max: ",max_positive,", ",max_negative)
    
    return tags_noise#return tag with noise


#add 10% noise to the positive points and another 10% noise for the negative points
tag_noise = noise(tag,0.1)

# print("Porcentaje de error de clasificación (con ruido): ",missClasificationRate(tag_noise,tag),"%")

# #Draw the graph
# fig2a = plt.figure()
# ax2a = fig2a.add_subplot()
# ax2a.scatter(x[:,0],x[:,1],c=tag_noise)
# #calculate the perfect parting line
# X = np.linspace(-50, 50, tag_noise.size)
# #solves the function for the variable Y
# #f(x,y) = y - ax -b
# Y = (a*X) + b
# ax2a.plot(X, Y)
# ax2a.set_title('Nube de puntos aleatoria uniforme, con 10% de ruido para los puntos positivos y negativos')

# input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
        xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
        ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
        xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
#create the complex funtions of the line

#f(x,y) = (x-1)² + (y-20)² - 400
def f1(x):
    return (np.power(x[:,0]-10,2)+np.power(x[:,1]-20,2)-400)

#f(x,y) = 0.5*(x+10)² + (y-20)² - 400
def f2(x):
    return ((np.dot(np.power(x[:,0]+10,2),0.5))+np.power(x[:,1]-20,2)-400)

#f(x,y) = 0.5*(x-10)² - (y+20)² - 400
def f3(x):
    return ((np.dot(np.power(x[:,0]-10,2),0.5))-np.power(x[:,1]+20,2)-400)

#f(x,y) = y - 20*x² - 5*x + 3
def f4(x):
    return (x[:,1]-(np.dot(20,np.power(x[:,0],2)))-(x[:,0].dot(5))+3)


#calculate miss clasification rate in %
def missClasificationRate2(y_training,y_test):
    error = 0;
    for i in np.arange(y_test.size):
        
        #print(y_training[i]," vs ",y_test[i])
        
        if(y_training[i]>0 and y_test[i]<0):
            #print("entra1")
            error += 1
        elif(y_training[i]<0 and y_test[i]>0):
            #print("entra2")
            error += 1
    return (error/y_test.size)*100

#calculate and show
# y1 = f1(x)    
# print("Porcentaje de error de clasificación (función 1): ",missClasificationRate2(y1,tag),"%")
# plot_datos_cuad(x, y1, f1,'f(x,y) = (x-10)² + (y-20)² - 400, with 20% noise')
# y2 = f2(x)
# print("Porcentaje de error de clasificación (función 2): ",missClasificationRate2(y2,tag),"%")
# plot_datos_cuad(x, y2, f2,'f(x,y) = 0.5*(x+10)² + (y-20)² - 400, with 20% noise')
# y3 = f3(x)
# print("Porcentaje de error de clasificación (función 3): ",missClasificationRate2(y3,tag),"%")
# plot_datos_cuad(x, y3, f3,'f(x,y) = 0.5*(x-10)² - (y+20)² - 400, with 20% noise')
# y4 = f4(x)
# print("Porcentaje de error de clasificación (función 4): ",missClasificationRate2(y4,tag),"%")
# plot_datos_cuad(x, y4, f4,'f(x,y) = y - 20*x² - 5*x + 3, with 20% noise')

# input("\n--- Pulsar tecla para continuar ---\n")

# ###############################################################################
# ###############################################################################
# ###############################################################################

# EJERCICIO 2.1: ALGORITMO PERCEPTRON

#sign(w^T*xi) = w0 + x1*w1 + x2*w2
def sign(x,y,w):
    #print(np.sum(x.dot(w)))
    #return np.sum(w.dot(x))
    #print(np.dot(x,np.transpose(w))*y)
    # lo de multiplicarlo por la etiqueta la encontre en esta página en la sección 2.8:
    # https://medium.com/analytics-vidhya/perceptron-learning-algorithm-2-67e49cbce2fd
    
    return np.dot(np.transpose(w),x)*y

#update w_new = w_old + xi*yi
def update_w(w,element, label):
    return(w+np.dot(label,element))
    

def ajusta_PLA(datos, label, max_iter, vini):
    w = vini
    #w_old = w
    i = 0 #index of each label
    iterations = 0
    change = 0

    #while w change
    while(iterations<max_iter and change < label.size):
        #go through the entire sample
        for element in datos:
            
            #if the classification is wrong
            if(sign(element,label[i],w) <= 0):
                change = 0
                #update w
                w = update_w(w,element,label[i])
            else:
                change += 1
            #if w does not change or exceed the maximum number of iterations, ends
            if(iterations>=max_iter or change >= label.size):
                #print(w," vs ",w_old)
                return w, iterations
            i += 1
            iterations += 1 #counts the number of iterations of each element accessed
        
        i = 0
        
        #w_old = w
    #print(w," vs ",w_old)
    #print("Iterations: ",iterations)
    return w, iterations

#simulate the point cloud randomly evelyn
x = simula_unif(100, 2, [-50,50])
#generates the a and b values to calculate the tags of each point
a, b = simula_recta([-50,50])
#generates the tags
tag = np.zeros(x[:,0].size)
for i in np.arange(x[:,0].size):
    tag[i] = f(x[i,0], x[i,1], a, b)
    
#create x0
ones = np.ones(int(x[:,0].size))
#add x0, x1 and x2
x_complet = np.array([ones,x[:,0],x[:,1]])
#swap the axes, for x(x0, x1, x2)
x_complet=x_complet.swapaxes(0,1)
w_initial = np.zeros(3)

#without noise

# start = time()
# w, iterations = ajusta_PLA(x_complet,tag,10000,w_initial)
# elapsed = time() - start
# print("Elapsed time",elapsed)
# print("W = ",w_initial,": W final: ",w,", iteraciones: ",iterations)

# # Random initializations
# iterations = []
# for i in range(0,10):
#     rand_initialize = np.random.uniform(0,1.000000000000000001,3)
#     w, iterate = ajusta_PLA(x_complet,tag,100000,rand_initialize)
#     #print("W =",rand_initialize,": W final: ",w,", iteraciones: ",iterate)
#     iterations.append(iterate)
    
# print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

# input("\n--- Pulsar tecla para continuar ---\n")

# # Ahora con los datos del ejercicio 1.2.b
# #with 10% of noise
# tag_noise = noise(tag, 0.1)

# start = time()
# w, iterations = ajusta_PLA(x_complet,tag_noise,100000,w_initial)
# elapsed = time() - start
# print("Elapsed time",elapsed)
# print("W = ",w_initial,": W final: ",w,", iteraciones: ",iterations)


# iterations = []
# for i in range(0,10):
#     rand_initialize = np.random.uniform(0,1.000000000000000001,3)
#     w, iterate = ajusta_PLA(x_complet,tag_noise,100000,rand_initialize)
#     #print("W =",rand_initialize,": W final: ",w,", iteraciones: ",iterate)
#     iterations.append(iterate)
    
# print('Valor medio de iteraciones necesario para converger (con ruido): {}'.format(np.mean(np.asarray(iterations))))

# input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
  
def gradient(x, y, w):
    
    #print("X = ",x)
    #print("Y = ",y)
    #print("W = ",w)
    result = 0
    result = y*x/(1+np.exp(y*w.dot(x)))
    #print("result = ",result)
    return -result

#the SGD Logisti Regression function receives as a parameters:
#   x: data set
#   y: the labels
#   w: the initial weights
#   t: the threshold
#   eta: Learning Rate
def sgdRL(x,y,w,t,eta):
    #initialize the variables
    delta = 999
    w_old = np.copy(w)
    iterations = 0
    
    #while (w(t-1) - wt) >= t
    while(delta >= t):
        
        RSI = np.random.permutation(x.shape[0])
        
        #compute the gradient
        for i in RSI:
            #and update the weights (w(t+1))
            w += w - (eta * gradient(x[i],y[i],w))
        
        iterations += 1
        
        # ||w(t-1) - wt|| and normalize the vector
        # ||x|| = |x|
        delta = np.linalg.norm(w_old-w)
        if(delta < t):
            return w, iterations
        # print("W = ",w)
        # print("W_old = ",w_old)
        # print("delta = ",delta)
        w_old = np.copy(w)
        
    #print(RSI)
    #print(delta," VS ", t)    
    return w, iterations

# Funcion para calcular el error cuadratico medio
def Err(x,y,w):
    
    #Tema 2 pag 61
    #Eout = ln(1+e^(-yi*wi*xi))
    result = np.float64(np.log(1+np.exp(-(x.dot(w) * y))))
    
    result.dtype = np.float64

    #result/n
    return result.mean()


# input("\n--- Pulsar tecla para continuar ---\n")
    
# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

w_initial = np.zeros(3)
eta = 0.01

#simulate point cloud
x_test = simula_unif(100,2,[-50,50])
ones = np.ones(int(x_test[:,0].size))

#generates the a and b values to calculate the tags of each point
a, b = simula_recta([-50,50])

#generates the tags
tag_test = np.zeros(x_test[:,0].size)
for i in np.arange(x_test[:,0].size):
    tag_test[i] = f(x_test[i,0], x_test[i,1], a, b)
 #now we have the perfect sample and tags

    
#add x0, x1 and x2
x_complet = np.array([ones,x_test[:,0],x_test[:,1]])
#swap the axes, for x(x0, x1, x2)
x_complet=x_complet.swapaxes(0,1)

w, iterations = sgdRL(x_complet,tag_test,w_initial,0.01,eta)
print(w)
print("Numero de iteracones con N = 100: ",iterations)
print("Eout = ",Err(x_complet, tag_test, w))

mean_iterations = 0;
mean_Eout = 0


start_time = time()
for i in np.arange(0,100):
    
    #simulate point cloud
    x_test = simula_unif(1000,2,[-50,50])
    ones = np.ones(int(x_test[:,0].size))
    
    #generates the a and b values to calculate the tags of each point
    #a, b = simula_recta([-50,50])
    
    #generates the tags
    tag_test = np.zeros(x_test[:,0].size)
    for i in np.arange(x_test[:,0].size):
        tag_test[i] = f(x_test[i,0], x_test[i,1], a, b)
      #now we have the perfect sample and tags
    
    #add x0, x1 and x2
    x_complet2 = np.array([ones,x_test[:,0],x_test[:,1]])



    #swap the axes, for x(x0, x1, x2)
    x_complet2=x_complet2.swapaxes(0,1)

    w, iterations = sgdRL(x_complet2,tag_test,w_initial,0.01,0.01)
    mean_iterations += iterations
    mean_Eout += Err(x_complet2, tag_test, w)
    print(iterations," vs ", Err(x_complet2, tag_test, w))
    
#print(mean_iterations," vs ",mean_Eout)
mean_iterations = mean_iterations/100
mean_Eout = mean_Eout/100

elapsed_time = time() - start_time
print("Regresion Lineal Elapsed time: %0.10f seconds" %elapsed_time)
print("Media del numero de iteracones con N = 1000 ejecutado 100 veces: ",mean_iterations)
print("Media del Eout con N = 1000 ejecutado 100 veces: ",mean_Eout)


# #CODIGO DEL ESTUDIANTE


# input("\n--- Pulsar tecla para continuar ---\n")


# ###############################################################################
# ###############################################################################
# ###############################################################################
# #BONUS: Clasificación de Dígitos


# # Funcion para leer los datos
# def readData(file_x, file_y, digits, labels):
# 	# Leemos los ficheros	
# 	datax = np.load(file_x)
# 	datay = np.load(file_y)
# 	y = []
# 	x = []	
# 	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
# 	for i in range(0,datay.size):
# 		if datay[i] == digits[0] or datay[i] == digits[1]:
# 			if datay[i] == digits[0]:
# 				y.append(labels[0])
# 			else:
# 				y.append(labels[1])
# 			x.append(np.array([1, datax[i][0], datax[i][1]]))
# 			
# 	x = np.array(x, np.float64)
# 	y = np.array(y, np.float64)
# 	
# 	return x, y

# # Lectura de los datos de entrenamiento
# x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# # Lectura de los datos para el test
# x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


# #mostramos los datos
# fig, ax = plt.subplots()
# ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
# ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
# ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
# ax.set_xlim((0, 1))
# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
# ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
# ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
# ax.set_xlim((0, 1))
# plt.legend()
# plt.show()

# input("\n--- Pulsar tecla para continuar ---\n")

# #LINEAR REGRESSION FOR CLASSIFICATION 

# #CODIGO DEL ESTUDIANTE


# input("\n--- Pulsar tecla para continuar ---\n")



# #POCKET ALGORITHM
  
# #CODIGO DEL ESTUDIANTE




# input("\n--- Pulsar tecla para continuar ---\n")


#COTA SOBRE EL ERROR

# CODIGO DEL ESTUDIANTE
