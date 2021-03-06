# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Ruben Girela Castllón
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from sympy import *
from time import time

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.2\n')

def E(u,v):#calcula los valores u y v con la función E
    return ((np.power(u,3)*np.exp(v-2))-(2*np.power(v,2)*np.exp(-u)))**2#function   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    #para crear la función indico los simbolos u y v
    x=Symbol('u')
    y=Symbol('v')
    #creo la función sin derivar
    f = ((x**3*exp(y-2))-(2*y**2*exp(-x)))**2
    
    # Derivo la función y la convierto en una función numerica, indicando que 
    # los simbolos son variables que recibe por parametro
    df = lambdify([x,y],f.diff(x))
    
    #print(df(u,v))
    #print(f.diff(x))
    #aplico la función derivable con los valores recibidos
    return df(u,v)#Derivada parcial de E con respecto a u resultado
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    #para crear la función indico los simbolos u y v
    x=Symbol('u')
    y=Symbol('v')
    #creo la función sin derivar
    f = ((x**3*exp(y-2))-(2*y**2*exp(-x)))**2
    
    # Derivo la función y la convierto en una función numerica, indicando que 
    # los simbolos son variables que recibe por parametro
    df = lambdify([x,y],f.diff(y))
    
    #print(df(u,v))
    #print(f.diff(y))
    #aplico la función derivable con los valores recibidos
    return df(u,v)#Derivada parcial de E con respecto a v resultado

#Gradiente de E (gradiente de la función)
def gradE(u,v):
    return np.array([np.float64(dEu(u,v)), np.float64(dEv(u,v))])

#necesita un w inicial, una constante pequeña (eta), numero maximo de iteraciones,
#la función a minimizar y el gradiente de dicha función,  Parte el error2get 
#como condicion de parada cuando la funcion llega a ser menor a error2get o 
#supere el numero maximo de iteraciones
def gradient_descent(w, eta, maxIter, error2get, F, gradF):
    #
    # gradiente descendente
    # 
    
    iterations=0#numero de iteraciones que hace

    #mientras no supere el maximo de iteraciones y el resultado de la funcion
    #sea mayor al error2get que es 10⁻¹⁴    
    while(F(w[0],w[1])>error2get and iterations < maxIter):
        iterations += 1#cuento el numero de iteraciones que hace
        
        #el gradiente de la funcion gradE(u,v) es Ein(w)/wj
        w = np.float64(w - (eta * gradF(w[0],w[1])))
        
        #el float64 es para que me muestre todos los decimales hasta 64 bits
        #aunque en mi caso no me hace falta ya que me da el mismo resultado, 
        #pero lo pongo porque eso dependera del computador
        
        #print("Funcion: " + str(F(w[0],w[1])))
    
    #devuelvo el valor de w y las iteraciones que hace
    return w, iterations    


eta = 0.1 #constante pequeña o learning rate
maxIter = 10000000000 #numero maximo de iteraciones
error2get = 1e-14
initial_point = np.array([1.0,1.0])#es el w inicial

w, it = gradient_descent(initial_point, eta, maxIter, error2get, E, gradE)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)

Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 1.3.A\n')

def F(x,y):#calcula los valores x e y con la función F
    return (np.power((x+2),2)+(2*np.power((y-2),2))+(2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)))#function   

#Derivada parcial de F con respecto a x
def dFx(x,y):
    
    return 2*(x+2)+(4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y))#Derivada parcial de F con respecto a x resultado
    
#Derivada parcial de F con respecto a y
def dFy(x,y):
    
    return 4*(y-2)+4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)#Derivada parcial de F con respecto a y resultado

#Gradiente de F (gradiente de la función)
def gradF(x,y):
    return np.array([np.float64(dFx(x,y)), np.float64(dFy(x,y))])

def gradient_descentF(w, eta, maxIter, F, gradF):
    #
    # gradiente descendente
    # 
    w_old= np.array([0.0,0.0])
    iterations=0#numero de iteraciones que hace
    points = []

    #mientras no supere el maximo de iteraciones
    while(iterations < maxIter):
        
        #guardo el valor de F con los valores obtenidos de inicio y los nuevos
        #en cada iteración, y en la iteración que se ha hecho
        points.append([F(w[0],w[1]),iterations])
        #obtengo el nuevo valor de w
        w = np.float64(w - (eta * gradF(w[0],w[1])))
        
        
        iterations += 1#cuento el numero de iteraciones que hace
    #guardo el valor de F de la w final
    points.append([F(w[0],w[1]),iterations])
    #devuelvo el valor de w, las iteraciones que hace y los valores de F en cada w
    return w, iterations, points

eta = 0.01 #constante pequeña o learning rate
maxIter = 50 #numero maximo de iteraciones
initial_point = np.array([-1.0,1.0])#es el w inicial

w, it, points = gradient_descentF(initial_point, eta, maxIter, F, gradF)
arr_points = np.array(points)#lo convierto a un array para mostrarlos en la grafica

print('Learning Rate ',eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: F(', w[0], ', ', w[1],') = ',F(w[0],w[1]))


# DISPLAY FIGURE 3D 3.A eta=0.01
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)

Z = F(X, Y)
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3.a Función sobre la que se calcula el descenso de gradiente F(x,y) eta=0\'01')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(x,y)')
fig.suptitle('Figura 3D Ejercicio 1.3.a eta=0\'01')

# DISPLAY FIGURE 2D 3.A eta=0.01
# for i in arr_points:
#     print(i[0],', ',i[1])
fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.plot(arr_points[:,1],arr_points[:,0])
ax1.set_xlabel('Iteraciones')
ax1.set_ylabel('F(x,y)')
ax1.set_title('Evolución del Gradiente Descendiente en F(x,y), con eta=0\'01')

eta = 0.1 #constante pequeña o learning rate

w, it, points = gradient_descentF(initial_point, eta, maxIter, F, gradF)
arr_points2 = np.array(points)

print('Learning Rate ',eta)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: F(', w[0], ', ', w[1],') = ',F(w[0],w[1]))


# DISPLAY FIGURE 3D 3.A eta=0.1
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)

Z = F(X, Y)
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], F(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.3.a Función sobre la que se calcula el descenso de gradiente F(x,y) eta=0\'1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('F(x,y)')
fig.suptitle('Figura 3D Ejercicio 1.3.a eta=0\'1')

# DISPLAY FIGURE 2D 3.A eta=0.1
# for i in arr_points2:
#     print(i[0],', ',i[1])
    
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(arr_points2[:,1],arr_points2[:,0])
ax2.set_xlabel('Iteraciones')
ax2.set_ylabel('F(x,y)')
ax2.set_title('Evolución del Gradiente Descendiente en F(x,y), con eta=0\'1')

# DISPLAY FIGURE 2D 3.A compare eta = 0.01 y 0.1
fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.plot(arr_points2[:,1],arr_points2[:,0],color='orange')
ax3.plot(arr_points[:,1],arr_points[:,0],color='blue')
ax3.set_xlabel('Iteraciones')
ax3.set_ylabel('F(x,y)')
ax3.set_title('Comparación de la evolución del Gradiente Descendiente en F(x,y), con eta=0\'01 y 0\'1')

input("\n--- Pulsar tecla para continuar ---\n")

print('Ejercicio 1.3.B\n')

eta = 0.01 #constante pequeña o learning rate
maxIter = 50 #numero maximo de iteraciones
#creo un array de w0s
initial_points = np.array([[-0.5,-0.5],[1.0,1.0],[2.1,-2.1],[-3.0,3.0],[-2.0,2.0]])

#creo una matriz 5x5 donde cada fila guarda:
#el w inicial, el w final y el valor de F del w final
results_w = np.array(np.zeros((5,5)))
results_p = []#lista de puntos para la grafica
pos=0

for ip in initial_points:# recorro el array de puntos iniciales
    #calculo el w final, el numero de iteraciones y la evolución de w
    w, it, points = gradient_descentF(ip, eta, maxIter, F, gradF)
    
    #guardo el resultado para la tabla
    results_w[pos]=[ip[0],ip[1],w[0],w[1],F(w[0],w[1])]
    results_p.append(points)#guardo los puntos
    pos +=1#para la matriz
    
    print('Initial Point ',ip)
    print ('Numero de iteraciones: ', it)
    print ('Coordenadas obtenidas: F(', w[0], ', ', w[1],') = ',F(w[0],w[1]),'\n')
    
#print(results_w)
    
#tablas que muestran los resultados de cada uno de los puntos iniciales
fig3b=plt.figure(dpi=200)#pongo esto porque sino pierde resolución
ax3b = fig3b.add_subplot()
ax3b.axis('off')
labels = ['w0 x','w0 y','wf x','wf y','F(wf)']
tabla = ax3b.table(
    cellText=(results_w), 
    loc='upper left', 
    colLabels=(labels), 
    colColours=["palegreen"]*5,
    cellLoc='center'
    )
tabla.auto_set_font_size(False)
tabla.set_fontsize(5)

#grafica de la evolución de cada punto inicial
arr_results_p = np.array(results_p)
fig3b1 = plt.figure()
ax3b1 = fig3b1.add_subplot()
ax3b1.plot(arr_results_p[0,:,1],arr_results_p[0,:,0],color='orange',label='(-0.5,-0.5)')
ax3b1.plot(arr_results_p[1,:,1],arr_results_p[1,:,0],color='blue',label='(1.0,1.0)')
ax3b1.plot(arr_results_p[2,:,1],arr_results_p[2,:,0],color='red',label='(2.1,-2.1)')
ax3b1.plot(arr_results_p[3,:,1],arr_results_p[3,:,0],color='green',label='(-3.0,3.0)')
ax3b1.plot(arr_results_p[4,:,1],arr_results_p[4,:,0],color='yellow',label='(-2.0,2.0)')
ax3b1.set_xlabel('Iteraciones')
ax3b1.set_ylabel('F(x,y)')
ax3b1.legend()
ax3b1.set_title('Comparación de la evolución del Gradiente Descendiente en F(x,y)')

input("\n--- Pulsar tecla para continuar ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
    # Leemos los ficheros	
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []	
    # Solo guardamos los datos cuya clase sea la 1 o la 5
    for i in range(0,datay.size):
        if(datay[i] == 5 or datay[i] == 1):
            if datay[i] == 5:
                y.append(label5)
            else:
                y.append(label1)
            x.append(np.array([1, datax[i][0], datax[i][1]]))
 			
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
 	
    return x, y

# Funcion para calcular el error cuadratico medio
def Err(x,y,w):
    
    #(yn - ŷ)²
    result = np.float64(np.power((x.dot(w) - y), 2))
    
    result.dtype = np.float64

    #result/n
    return result.mean()

# mini-batch gradient (derivate of the error function)
def gradient(x,y,w, mini_batch):
    
    sum_total_batch = 0
    
    #iterate the mini-batch
    for i in mini_batch:
        #sumatory(Xn * (h(Xn) - Yn)), where n is the iteration of the mini-batch
        sum_total_batch += x[i]*(np.sum(w*x[i])-y[i])
        #print(sum_total_batch)
    
    #update the w = w - eta * sumatory
    return (sum_total_batch*2/mini_batch.size)
        

# Gradiente Descendente Estocastico
# Need x (input data) and y (the labels (desired output))
def sgd(x,y,eta,size_batch, maxIter, w, error2get):
    
    
    #shuffle the indexes
    indices = np.random.permutation(len(x))
    min_batch_n = 0 # indexe of mini-batch
    iterate=0
    #maximum number of iterate for each mini-batch
    max_iteration_batch = size_batch
    
    print("calculate... (aprox 25 seconds)")
    
    # while the error is greater than 10¹⁴ and doesn't exceed the
    #maximum number of iterations
    while(Err(x,y,w)>error2get and iterate < maxIter):
        
        #if the mini-batch limit is greater than the size of x and the first index
        #is less than the size of x
        if(max_iteration_batch>=len(x) and (min_batch_n*size_batch)<len(x)):
            #create the mini-batch from the first index to (the size of x) -1
            max_iteration_batch = len(x)
        
        #create the mini-batch
        mini_batch = indices[min_batch_n:max_iteration_batch]
        
        #update the w = w - eta * (derivate of square error)
        w = w - (eta*gradient(x,y,w,mini_batch))
        #print(w)
            
        #increase the limit of each mini-batch and minimum
        max_iteration_batch = max_iteration_batch + size_batch
        min_batch_n += size_batch
        
        #if min_batch_n exceeds the minimum value above the sample size
        if(min_batch_n > len(x)):
            # reset indexs
            min_batch_n = 0
            max_iteration_batch = size_batch
            
        #increment in each iteration
        iterate = iterate + 1
    
    return w #it return the optimal w

# Pseudoinversa	
def pseudoinverse(x,y):
    #compute the pseudo-inverse with this function (np.linalg.inv)
    #X' = (X^T*X)^-1 * X^T
    x_ps_inv = np.float64(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T))
    
    #w = X' * y
    return np.float64(x_ps_inv.dot(y))

   

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

eta = 0.01 #learning rate
size_batch=32 #it is the size for each mini-batch
maxIter = 50000
w = np.zeros(3, dtype=np.float64) #initialize w to 0

start_time = time()
w = sgd(x,y,eta,size_batch, maxIter, w, error2get)
elapsed_time = time() - start_time
print("SGD Elapsed time: %0.10f seconds" %elapsed_time)


start_time = time()
w2 = pseudoinverse(x,y)
w2.dtype=np.float64
elapsed_time = time() - start_time
print("Psud.-Inv Elapsed time: %0.10f seconds" %elapsed_time)


print ('\nBondad del resultado para grad. descendente estocastico:\n')
print ("w: ", w)
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

print ('\nBondad del resultado para Pseudo-Inverse:\n')
print ("w: ", w2)
print ("Ein: ", np.float64(Err(x,y,w2)))
print ("Eout: ", np.float64(Err(x_test, y_test, w2)))

input("\n--- Pulsar tecla para continuar ---\n")

#grafica SGD

fig21 = plt.figure()
ax21 = fig21.add_subplot()
ax21.scatter(x[:,1],x[:,2],c=y)
ax21.plot([0,1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])

ax21.set_xlabel('Intensity')
ax21.set_ylabel('Simmetry')
ax21.set_title('SGD')

#grafica pseudo-inverse

fig212 = plt.figure()
ax212 = fig212.add_subplot()
ax212.scatter(x[:,1],x[:,2],c=y)
ax212.plot([0,1], [-w2[0]/w2[2], -w2[0]/w2[2]-w2[1]/w2[2]])

ax212.set_xlabel('Intensity')
ax212.set_ylabel('Simmetry')
ax212.set_title('Pseudo-Inverse')



print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
 	return np.random.uniform(-size,size,(N,d))

def sign(x):
 	if x >= 0:
         return 1
 	return -1
 
#tags with 10% noise
def noise(tags,percent):
    
    #copy the tags
    tags_noise = tags.copy()
    #number of points with noise 10%
    n_noise = int(tags_noise.size*percent)
    
    #indices of sample with noise (random integer without repeating)
    ind_noise = np.random.choice(tags_noise.size, n_noise, replace=False)
    #print(ind_noise)
    
    for i in ind_noise:
        tags_noise[i]=-tags_noise[i]#change the sign
    
    return tags_noise

#function that calculate the tags
def F2(x1, x2):
    #f = (x1 -0.2)²+x2²-0.6
    result = np.power(x1-0.2,2) + np.power(x2,2) - 0.6
    
    #tag array
    tags = np.zeros(x1.size)
    
    ind = 0 #index for noise
    
    for i in result:#loop throught the results of the function
        
        #add the tag
        tags[ind]=sign(i)
        ind = ind + 1
        
    #print(tags)
    return tags

sample = simula_unif(1000,2,1)

#Section A: show the 2D points of the sample of 1000 random points
fig22a = plt.figure()
ax22a = fig22a.add_subplot()
ax22a.scatter(sample[:,0],sample[:,1])
ax22a.set_title('1000 Points random')

#add the tags
tags = F2(sample[:,0],sample[:,1])
tags_noise = noise(tags,0.1)

#Section B: show the 2D points of the sample with the labels with 10% noise
fig22a = plt.figure()
ax22a = fig22a.add_subplot()
ax22a.scatter(sample[:,0],sample[:,1], c=tags_noise)
ax22a.set_title('1000 Points random with 10% noise')


#Section C: 
#x0
ones = np.ones(int(sample.size/2))
#add x0, x1 and x2
sample_c = np.array([ones,sample[:,0],sample[:,1]])
#swap the axes, for sample_c(x0, x1, x2)
sample_c=sample_c.swapaxes(0,1)

    
# mini-batch gradient (derivate of the error function)
def gradientF(x,y,w):
        
    #sumatory(Xn * (h(Xn) - Yn)*2)/n, where n is the iteration of the mini-batch
    sum_total_batch = np.sum(x.dot(w))-y
    
    #derivate
    return (sum_total_batch.dot(x)*2/x[:,0].size)

def sgdF(x,y,eta,size_batch, maxIter, w, error2get):
    
    
    #shuffle the indexes
    indices = np.random.permutation(x[:,0].size)
    min_batch_n = 0 # indexe of mini-batch
    iterate=0
    #maximum number of iterate for each mini-batch
    max_iteration_batch = size_batch
    
    # while the error is greater than 10¹⁴ and doesn't exceed the
    #maximum number of iterations
    while(Err(x,y,w)>error2get and iterate < maxIter):
        
        if(max_iteration_batch>=x[:,0].size and (min_batch_n*size_batch)<len(x)):
            #create the mini-batch from the first index to (the size of x) -1
            max_iteration_batch = x[:,0].size
        
        #create mini-batches
        batch_x = x[indices[min_batch_n:max_iteration_batch],:]
        batch_y = y[indices[min_batch_n:max_iteration_batch]]
        
        #update the w = w - eta * (derivate of square error)
        w = w - (eta*gradientF(batch_x,batch_y,w))        
        #print(w)
            
        #increase the limit of each mini-batch and minimum
        max_iteration_batch = max_iteration_batch + size_batch
        min_batch_n += size_batch
        
        #if min_batch_n exceeds the minimum value above the sample size
        if(min_batch_n > len(x)):
            # reset indexs
            min_batch_n = 0
            max_iteration_batch = size_batch
        
        #increment in each iteration
        iterate = iterate + 1
            
    #print("Iterations: ",iterate)
    return w #it return the optimal w

eta = 0.001
size_batch = 64
max_it = 5000
w = np.zeros(3, dtype=np.float64) #initialize w to 0

start_time = time()
w = sgdF(sample_c, tags_noise, eta, size_batch, max_it,w, error2get)
w.dtype=np.float64
elapsed_time = time() - start_time
print("SGD Elapsed time (10 percent noise): %0.10f seconds" %elapsed_time)


print ('\nBondad del resultado para grad. descendente estocastico:\n')
print ("w: ", w)
print ("Ein: ", Err(sample_c,tags_noise,w))

#graph SGD section C
fig22c = plt.figure()
ax22c = fig22c.add_subplot()
ax22c.scatter(sample_c[:,1],sample_c[:,2],c=tags_noise)
X = np.linspace(-1, 1, y.size)
Y = (-w[0]-w[1]*X)/w[2]
ax22c.plot(X, Y)
ax22c.set_ylim(-1.0,1.0)

ax22c.set_title('SGD')

number_iterations = 1000
max_it = 400

Ein = np.empty(number_iterations)
Eout = np.empty(number_iterations)

print("\ncalculate 1000 samples diferents: (aprox 2 minuts)")
start_time = time()
#Section D: run the experiment 1000 times with different samples and calculate the Ein and Eout
for i in np.arange(number_iterations):
    sample = simula_unif(1000,2,1)
    #x0
    ones = np.ones(int(sample.size/2))
    #add x0, x1 and x2
    sample_c = np.array([ones,sample[:,0],sample[:,1]])
    #swap the axes, for sample_c(x0, x1, x2)
    sample_c=sample_c.swapaxes(0,1)

    #add the tags
    tags = F2(sample[:,0],sample[:,1])
    tags_noise = noise(tags,0.1)
    
    w = sgdF(sample_c, tags_noise, eta, size_batch, max_it,w, error2get)
    w.dtype=np.float64
    
    Ein[i]=Err(sample_c,tags_noise,w)
    
    #Eout
    sample_test = simula_unif(1000,2,1)
    #x0
    ones = np.ones(int(sample.size/2))
    #add x0, x1 and x2
    sample_test = np.array([ones,sample_test[:,0],sample_test[:,1]])
    #swap the axes, for sample_c(x0, x1, x2)
    sample_test=sample_test.swapaxes(0,1)
    #add tags with 10% noise
    tags_test = F2(sample_test[:,0],sample_test[:,1])
    tags_test = noise(tags_test,0.1)
    
    Eout[i] = Err(sample_test, tags_test, w)#calculate Eout
    
    
elapsed_time = time() - start_time
print("SGD (1000 samples) Elapsed time: %0.10f seconds" %elapsed_time)
print ("Ein medio: ", Ein.mean())
print ("Eout medio: ", Eout.mean())

print('\nEjercicio 2 NO LINEAL\n')

# Simulate the data
def vectorFeature(x1, x2):
    
 	#feature vector to return
    vf = np.zeros((x1.size,6))
    vf.dtype= np.float64
    vf[:,0] = 1
    vf[:,1] = x1
    vf[:,2] = x2
    vf[:,3] = x1*x2
    vf[:,4] = np.power(x1,2)
    vf[:,5] = np.power(x2,2)
    #print(vf)
    return vf  


Ein2 = np.empty(number_iterations)
Eout2 = np.empty(number_iterations)
w = np.zeros(6, dtype=np.float64) #initialize w to 0
number_iterations = 1000
max_it = 500
size_batch = 32
eta = 0.001
    
print("calculate 1000 samples diferents non-linear: (aprox 1 minut)")
start_time = time()

# #Section D: repeat the experiment but with non-linear characteristics
for i in np.arange(number_iterations):
    #create the random sample    
    sample2 = simula_unif(1000,2,1)
    #create the sample with feature = (1, x1, x2, x1x2, x1²,x2²)
    sample_c2=vectorFeature(sample2[:,0],sample2[:,1])
    
    #add the tags
    tags2 = F2(sample2[:,0],sample2[:,1])
    tags_noise2 = noise(tags2,0.1)
    
    w = sgdF(sample_c2, tags_noise2, eta, size_batch, max_it, w, error2get)
    w.dtype=np.float64
    
    #print(w)
    
    Ein2[i]=Err(sample_c2,tags_noise2,w)
    
    #create the random sample test   
    sample_test2 = simula_unif(1000,2,1)
    #create the sample test with feature = (1, x1, x2, x1x2, x1²,x2²)
    sample_test_c2=vectorFeature(sample_test2[:,0],sample_test2[:,1])
    
    #add the tags test
    tags_test2 = F2(sample_test2[:,0],sample_test2[:,1])
    tags_test_noise2 = noise(tags2,0.1)
    
    Eout2[i] = Err(sample_test_c2, tags_test_noise2, w)#calculate Eout
    
    
elapsed_time = time() - start_time
print("SGD (1000 samples) Elapsed time: %0.10f seconds" %elapsed_time)
print ("Ein medio: ", Ein2.mean())
print ("Eout medio: ", Eout2.mean())


Ein2 = np.empty(number_iterations)
Eout2 = np.empty(number_iterations)
w = np.zeros(6, dtype=np.float64) #initialize w to 0
number_iterations = 1000
max_it = 500
size_batch = 32
eta = 0.001