import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import scipy
from math import sqrt
from scipy.linalg import eigh

#%% General
def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

#%% Ejercicio 3
def top3museos(p): # tomamos los 3 mejores museos de cada pagerank armado
    return maximos(p)

def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    I = np.eye(len(A),k=0)
    C = calcula_matriz_C(A)
    N = len(A) # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (I - ((1-alfa) * calcula_matriz_C(A)))
    L, U = calculaLU(M) # Calculamos descomposición LU a partir de C y d
    b = (alfa / N) * np.ones(N) # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p

def ordenar_lista(l,p): # funcion para ordenar una lista de mayor a menor, a partir de los indices (es decir la lista me pasa valores q son los indices de otra lista p, y los ordeno segun p)
    for i in range(len(l)):
        for j in range(i,len(l)): # voy haciendo comparaciones entre todos los elementos de la lista y los swapeo de ser necesario
            if p[l[j]] > p[l[i]]: # consulto es elemento de la lista l (indice de p) en la lista p
                k = l[i]
                l[i] = l[j]
                l[j] = k
    return l


def maximos(l): # funcion para calcular los 1ros 3 maximos de una lista, segun su indice de otra lista 
    res = []
    max1 = l[0] # tomo los 3 1ros numeros de la lista y sus indices
    max2 = l[1]
    max3 = l[2]
    indice_max1 = 0
    indice_max2 = 1
    indice_max3 = 2
    res.append(indice_max1)
    res.append(indice_max2)
    res.append(indice_max3)
    
    res = ordenar_lista(res,l) # los ordeno segun su indice en "p"
    
    for i in range(3,len(l)): # voy comparando los elementos de res con el resto de los elementos de la lista
        if l[i] > l[res[2]]:
            res[2] = i
            res = ordenar_lista(res,l)
    return res 

def top3museos(p):
    return maximos(p)

#top 3 museos para cada m
def sortmuseos(ranking): # funcion para buscar los top 3 museos a medida q varía m
    museos = []
    for p in ranking: # voy recorriendo cada valor p d cada pagerank q le pasamos arriba como parámetro
        for i in range(0,3):
            if museos == []:
                museos.append(top3museos(p)[i]) # voy agregando los top 3 museos segun cada p a mi lista museos (si ya estaban no los agrego)
            if top3museos(p)[i] not in museos:
                museos.append(top3museos(p)[i])
    return museos


#%% Ejercicio 5

def calculaLU(m):
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    
    n = len(m)
    L = np.eye(n,k=0) # me armo la matriz L como identidad
    U = m.astype(float).copy() # armo U como una copia de m
    for j in range(n): # algoritmo para ir permutando filas en caso de q sea necesario (0 en la diagonal) (no se si era necesario esto al final, pero por las dudas)
            if U[j][j] == 0:  # si el valor = 0, busco una fila debajo q no tenga 0 en esa posicion
                for s in range(j+1,n):
                    if abs(U[s][j])>abs(U[j][j]): # busco la fila, debajo de la q me apareció un 0 en la diagonal, cuyo elemento en fila[j] tenga un valor absoluto mayor a esa misma posicion pero de las otras filas restantes  
                        U[[j, s]] = U[[s, j]] # intercambio filas
                        L[[j, s], :j] = L[[s, j], :j] # tamb intercambio las filas en L
                        
            for i in range(j+1,n):
                L[i,j] = U[i,j] / U[j,j]
                U[i,:] = U[i,:] - L[i,j]*U[j,:] # a medida q triangulo, me voy armando L y U
    return L, U
        
def grado_inv(m): # funcion para calcular la inversa de la matriz grado
     n = len(m)
     res = np.eye(n,k=0) # inicializo res como matriz identidad de tamaño n x n
     contador = 0 # inicializo contador para contar la cantidad de elementos en cada fila

     for i in range(n): 
          contador = 0
          for j in range(n): #voy recorriendo fila por fila
               contador += m[i][j] # y voy sumando los numeros de cada fila 
          res[i][i] *= contador # agrego el contador de cada fila a su diagonal correspondiente
          res[i][i] = 1/res[i][i] # invierto la diagonal
     return res
 
def grado(m): # funcion para calcular la matriz grado
     n = len(m)
     res = np.eye(n,k=0) # inicializo res como matriz identidad de tamaño n x n
     contador = 0 # inicializo contador para contar la cantidad de elementos en cada fila

     for i in range(n): 
          contador = 0
          for j in range(n): #voy recorriendo fila por fila
               contador += m[i][j] # y voy sumando los numeros de cada fila 
          res[i][i] *= contador # agrego el contador de cada fila a su diagonal correspondiente
     return res

def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    Kinv = grado_inv(A) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = np.transpose(A) @ Kinv  # Calcula C multiplicando Kinv y A
    return C

def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = grado_inv(F) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = F @ Kinv # Calcula C multiplicando Kinv y F
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matriz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    C_in = np.eye(C.shape[0])
    for i in range(1,cantidad_de_visitas):
        C_in = C_in @ C
        B += C_in # Sumamos las matrices de transición para cada cantidad de pasos
    return B

def calcula_v(w,B): # función para calcular el vector v (vector con el número de entrada de personas para cada museo)
    L,U = calculaLU(B) 
    Up = scipy.linalg.solve_triangular(L,w,lower=True) # Primera inversión usando L.
    #Resolvemos el sistema lineal con LU para hallar v, igualando al vector w (cantidad de personas totales que pasaron por cada museo)
    v = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U 
    
    return sum(v)  
    
#%% Ejercicio 6

def inversa(m): # funcion para calcular la inversa de una matriz a partir de LU
    n = len(m)
    L, U = calculaLU(m) # calculo L y U para resolver el sistema lineal
    inv = np.zeros((n,n)) # inicializo mi matriz inversa con 0
    
    for i in range(n): # voy armando vector por vector
        e = np.zeros(n) 
        e[i] = 1 #me armo el vector canonico 
        Up = scipy.linalg.solve_triangular(L,e,lower=True) # Primera inversión usando L, igualo al vector canonico para hallar su inversa
        b = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
        inv[:,i] = b   # guarda el res en la i-esima columna de la matriz inv
    
    return inv


def norma_1(m): # funcion para calcular la norma 1 de una matriz
    max = 0 
    for i in range(len(m)):
        contador = 0
        for j in range(len(m)):
            contador += abs(m[j][i]) # voy sumando los valores absolutos de cada columna por separado
        if contador > max: # si la suma de los valores absolutos de esa columna es mayor al maximo q tenia, lo reemplazo
            max = contador

    return max

def num_condicion(m): # funcion para calcular el numero de condicion de una matriz
    return norma_1(m) * norma_1(inversa(m)) 

#%% TP 2
# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])
#%% Ejercicio 3

def calcula_L(A): 
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    L = grado(A) - A # K - A
    return L

def calcula_R(A):
    # La funcion recibe la matriz de adyacencia A y calcula la matriz de modularidad
    n = len(A)
    P = np.zeros((n,n))
    mgrado = grado(A) # K
    E = np.sum(A)
    
    for i in range(0,len(mgrado)):
        for j in range(len(mgrado)):
            P[i][j] = (mgrado[i][i]*mgrado[j][j]) / (2*E)
            
    R = A - P
    return R

def autovect_min(L):
    autovalores, autovectores = eigh(L) #calcula los autoval y autovect de L, ordenados segun el autovalor
    return autovectores[:,1] #devuelve la 2da columna de la matriz con los autovect en las columnas (no tomo el 1er autovector pq tiene autovalor = 0 -> no realiza corte)
    
def signos(v): # recibe un vector de tamaño n y devuelve un vector con 1 y -1 segun los signos de cada pos del vector original
    n = len(v)
    res = [0]*n
    for i in range(len(v)):
        if v[i] < 0:
            res[i] = -1
        else:
            res[i] = 1
    return res

def calcula_lambda(L,v): 
    # Recibe L y v y retorna el corte asociado
    s = signos(v) # vector de los signos de v (1 si es positivo o 0 y -1 en caso contrario)
    lambdon = (1/4) * (np.transpose(s) @ L @ s) # corte asociado
    
    return lambdon

def calcula_Q(R,v): 
    # La funcion recibe R y s y retorna la modularidad (a menos de un factor 2E)
    # Corregimos la funcion teniendo en cuenta las notas del enunciado, ahora no es necesario calcular E
    s = np.sign(v)
    Q = (v.T@R@v)
    return Q

def norma_2(v): # funcion que calcula la norma 2 de un vector
    contador = 0
    for i in range(len(v)):
        contador += (v[i])**2
    return sqrt(contador)

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   n = len(A)
   v = np.random.uniform(-1, 1, size=n) # Generamos un vector de partida aleatorio, entre -1 y 1
   v = v / norma_2(v) # Lo normalizamos
   v1 = A @ v # Aplicamos la matriz una vez
   v1 = v1 / norma_2(v1) # normalizamos
   l = np.transpose(v) @ A @ v # Calculamos el autovalor estimado 
   l1 = np.transpose(v1) @ A @ v1 # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada 
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A @ v1 # Calculo nuevo v1
      v1 = v1 / norma_2(v1) # Normalizo
      l1 = np.transpose(v1) @ A @ v1 # Calculo autovector
      
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = np.transpose(v1) @ A @ v1 # Calculamos el autovalor
   return v1,l,nrep<maxrep

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1 * np.outer(v1,v1) # Calculamos su version deflacionada
    return deflA

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   # La funcion aplica el metodo de la potencia para buscar el segundo autovalor de A, suponiendo que sus autovectores son ortogonales
   # v1 y l1 son los primeors autovectores y autovalores de A
   deflA = A-(l1*np.outer(v1,v1)) 
   return metpot1(deflA,tol,maxrep)

def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    n = len(A)
    I = np.eye(n) 
    M = A + mu*I
    inv = inversa(M) # funcion que calcula la inversa de M

    v,l,_ = metpot1(inv,tol=tol,maxrep=maxrep) # calculo el 1er autovalor y su autovector de la matriz inversa
    l_min = 1 / l  # autovalor mas chico de M

    return v,l_min,_

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A, 
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   n = len(A)
   I = np.eye(n)

   X = A + mu*I # Calculamos la matriz A shifteada en mu
   iX = inversa(X) # La invertimos
   defliX = deflaciona(iX,tol,maxrep) # La deflacionamos
   v,l,_ = metpot1(defliX,tol,maxrep)  # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return [nombres_s]
    else:
        L = calcula_L(A) # Recalculamos el L
        v,l,_ = metpotI2(L,1,tol=1e-8,maxrep=np.inf) # Encontramos el segundo autovector de L
        
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo
        Ap = A[v>=0,:][:,v>=0] # Asociado al signo positivo
        Am = A[v<0,:][:,v<0] # Asociado al signo negativo
        
        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )        

def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = list(range(R.shape[0]))
    # Acá empieza lo bueno ...
    if R.shape[0] == 1: # Si llegamos al último nivel (si tiene un solo nodo)
        return([nombres_s])
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            # Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[v>=0,:][:,v>=0] # Parte de R asociada a los valores positivos de v
            Rm = R[v<0,:][:,v<0] # Parte de R asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm
        
            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                modularidad_iterativo(A,Rp,nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>=0]) +
                modularidad_iterativo(A,Rm,nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )

def A_simetrizada(a): # funcion para simetrizar una matriz dada
    res = 1/2 * (a+np.transpose(a))
    return np.ceil(res) #redondeo

### ANALISIS MATRIZ EJEMPLO
# Matriz A de ejemplo
A_ejemplo = np.array([
    [0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]
])

R = calcula_R(A_ejemplo)
print(laplaciano_iterativo(A_ejemplo,1,nombres_s=None))
print(laplaciano_iterativo(A_ejemplo,2,nombres_s=None))
print(modularidad_iterativo(A_ejemplo,R,nombres_s=None))

'''
En modularidad, vemos que la partición óptima es en dos grupos. Tiene sentido, dado que si observamos el bloque superior 
entre las filas 0 y 3, vemos que los nodos se conectan mayormente entre sí y únicamente los nodos 2 y 3 tienen una conexión externa.
Lo mismo vale para el bloque inferior (filas 4 a 7). Modularidad agrupa buscando maximizar las conexiones internas, y en este caso el siguiente posible corte no mejora 
Q, por lo que el método se detiene en dos comunidades.

Mientras tanto, para laplaciano en un principio elegimos que sólo haga un corte, para así poder comparar con modularidad. Vemos que los agrupa de la misma manera.
Esto también tiene sentido, dado que laplaciano busca agrupar minimizando las conexiones externas, y en el caso de esta matriz sucede que al maximizar las 
conexiones internas se minimizan las externas y viceversa. En este caso cada bloque está casi completamente conectado internamente y mínimamente hacia afuera, 
de modo que ambos métodos resultan equivalentes.

Para que el laplaciano resulte en una partición de 4 grupos, se lo debe fijar con niveles = 2. En este caso, sigue manteniendo en comunidades separadas a los nodos 0-3
de los nodos 4-7, pero ahora también divide cada una de esas comunidades.

'''
