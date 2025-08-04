# -*- coding: utf-8 -*-
"""
INTEGRANTES:
    Cristina Dorogov, Francisco Lautaro Pucciarelli, Julian Salto

GRUPO:
    N°8
"""

# %%
"""
Imports de Módulos.
"""
import numpy as np
import pandas as pd
import duckdb as dk
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path
import duckdb
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --------------------------------------------------------------------------------------------------------------------------------------- #

#%%
'''ANTES DE EJECUTAR EL CÓDIGO CAMBIAR ESTAS LINEAS QUE CONTIENEN LAS DIRECCIONES DE LOS ARCHIVOS UTILIZADOS

 40, 444, 649

'''
# %% 
""" 
Carga del DataSet.
"""

data_fashion:pd.DataFrame = pd.read_csv("C:/Users/maxim/Desktop/labo datos/TP/tp_2/archivos/Fashion-MNIST.csv", index_col=0)
# COMPLETAR CON EL PATH AL DATASET ORIGINAL.


"""
Aclaración Importante: 
Todas las instrucciones 'print()' las dejo comentadas, son verificaciones rápidas que fuí haciendo para ver si lo que iba haciendo 
era correcto o no. Lo único que hacen es imprimir mensajes o resultados intermedios.
"""


# --------------------------------------------------------------------------------------------------------------------------------------- #
# %% 
"""
Exploración Inicial: entendiendo qué tiene el DataSet.
"""

"""
Quiero ver cuáles son todas las clasificaciones de prendas distintas.
OBS -> De paso le hago 'sort', ya que si las imprimo así nomas, no me quedan ordenadas y es mas dificil saber qué numeros hay y cuáles no.
"""

clasificacion_prendas = data_fashion["label"].unique()
clasificacion_prendas = sorted(clasificacion_prendas) 
# print("Clasificaciones de Prendas: " , clasificacion_prendas)


"""
Características básicas:
    
- Imágenes de 28x28 pixeles.
- Cada columna es el valor del color que debe tener cada pixel -> Hay 28x28 = 784 columnas.
- Hay 70.000 filas; es decir, 70.000 imágenes.

- La última columna, llamada 'label', es el tipo de prenda al que representa esa imágen.
- Tenemos 10 clasificaciones de prendas distintas: 0 a 9.
"""


# --------------------------------------------------------------------------------------------------------------------------------------- #
# %%
"""
Necesito buscar patrones que me permitan identificar qué píxeles tienen información importante y qué píxeles no.

Mi primera idea es ver si ciertos píxeles tienen valores distintivos en cada tipo de prenda (para cada clasificación).
"""

"""
Voy a realizarlo con una consulta SQL. 
Lo que busco hacer es una consulta que le calcule el AVG (promedio) a cada pixel de cada tipo de prenda. Hay 784 píxeles, es inviable escribir
'AGV(columna)' para cada columna dentro de la consulta. Así que primero voy a hacer una función que me genere automaticamente esas sentencias.
"""

columnas_pixeles:list[str] = [f"pixel{i}" for i in range(0, 784)]   # Aquí tengo todos los nombres de las columnas.
# print(columnas_pixeles[0])   # A ver cómo quedó el primero.
# print(columnas_pixeles[15])  # A ver cómo quedó uno del medio.
# print(columnas_pixeles[-1])  # A ver cómo quedó el último.

"""
La sentencia la armo como un solo string, le pongo un salto de línea para que sea entendible cuando la imprima para verificar cómo quedó.
"""
sentencias_AVG:str = ", \n".join([f"AVG({col}) AS avg_{col}" for col in columnas_pixeles])

"""
Ahora sí, hago la consulta.
"""
consulta_1:str = f"""
                 SELECT label, 
                        {sentencias_AVG}
                 FROM data_fashion 
                 GROUP BY label 
                 ORDER BY label
                 """
promedio_pixel_por_clase:pd.DataFrame = dk.query(consulta_1).df()  
# print(promedio_pixel_por_clase.head(10))  

"""
Ok, ahora me gustaría graficar lo que me quedó del promedio de cada clase.
Aquí lo que voy a estar haciendo es visualizando cuál es la "imágen promedio" de cada una de las imágenes que pertenecen a cada clase.
Es decir, para cada clase, obtendé la imágen promedio.
"""
for i in range(10):
    imagen = np.array(promedio_pixel_por_clase.iloc[i, 1:]).reshape(28, 28)

    plt.figure()                
    plt.imshow(imagen, cmap="gray")
    plt.title(f"Imagen Promedio - Clase {i}")
    plt.axis('off')   # En estos gráficos, los ejes no me aportan nada. Los saco.
    

"""
Análisis de los gráficos. 

A partir de estas imágenes podemos establecer una mejor clasificación para los distintos tipos de prendas.
De forma aproximada, lo que observamos es:
    
- Clase 0 -> Prenda Superior: Mangas Cortas.
- Clase 1 -> Prenda Inferior: Pantalón Largo.
- Clase 2 -> Prenda Superior: Mangas Largas.
- Clase 3 -> Prenda Completa: Vestido Largo, Mangas Cortas.
- Clase 4 -> Prenda Superior: Mangas Largas.
- Clase 5 -> Calzado.
- Clase 6 -> Prenda Superior: Mangas Largas.
- Clase 7 -> Calzado.
- Clase 8 -> Accesorio: Bolso.
- Clase 9 -> Botas.

A partir de esto podemos suponer que cada Clase va a representar variantes del tipo de prenda que refleja su promedio.

Además de eso, notamos que hay Clases muy parecidas entre sí. Las Clases 2, 4 y 6 parecen representar variantes de la misma pieza: 
prendas de mangas largas.
Dificil diferenciar si se trata de remeras, abrigos, camisas (formales), etc; también si se trata de prendas masculinas o femeninas.
Las Clases 5, 7 y 9 queda claro que corresponden a calzados. Mas no creemos que sean lo suficientemente claras como para diferenciar 
tipos de calzados en cada clase. 

En cuanto a los píxeles con información importante, podemos notar lo siguiente.
Parece ser una constante que los píxeles de los bordes se mantienen siempre en negro, y que los píxeles del centro de cada imágen 
se mantienen siempre en blanco.
No nos parece correcto pensar en píxeles que no aportan información en todas las clases. Nos parece más apropiado ir Clase por Clase, 
diferenciando aquellos píxeles que se mantienen constantes en el promedio. Por ejemplo: En la Clase 1, podemos generalizar que los 
pixeles que aportan información relevante son aquellos que se mantienen más en el medio; los demás parecen mantener un promedio de negro.
Sin embargo, por ahora decidimos no sacar conclusiones al respecto. Vamos a esperar el análisis posterior para poder cerrar esta conclusión.
"""


# --------------------------------------------------------------------------------------------------------------------------------------- #
# %%
"""
Para poder hacer un análisis más fino sobre cuáles son aquellos píxeles que podrían descartarse (porque no aportan información), proponemos
el siguiente paso. 
Calcular la varianza de cada píxel en todo el DataSet (no separando por clase). Esto puede darnos una mejor idea de cuáles son los píxeles 
que no aportan información. Ya que podremos diferenciar más claramente cuáles son "casi siempre" negro, cuáles son "casi siempre" 
blancos y cuáles se mueven en un rango de grises (a veces más negro y a veces más blanco).
No vemos demasiado sentido separar por Clases para hacer este análisis, ya que buscamos aquellos píxeles que siempre (o casi siempre) se 
mantienen totalmente encendidos (blanco) o totalmente apagados (negro). No nos importa la Clase de la prenda que se oculta en la imágen 
(por ahora...), queremos mirar todo.
Vamos a buscar los píxeles que se mantienen siempre (o casi siempre) en blanco y aquellos que se mantienen siempre (o casi siempre) en 
negro. Estos son píxeles que, consideramos, no aportan información para diferenciar Clases (y, más adelante, buscar predecir); y que 
podríamos "descartar". 
"""


"""
Para empezar, calculo la varianza de cada píxel.
Recupero de la sección anterior 'columnas_pixeles', la lista de todos los nombres de todas las columnas de los píxeles.
"""
varianzas_pixeles = data_fashion[columnas_pixeles].var()
# print(varianzas_pixeles.head(10)) 

"""
Para poder visualizar esta informaación, construyo un histograma con la información obtenida.
La idea es ver cuáles son los valores de varianza que más cantidad de píxeles tienen.
"""
nbins = 70   # Tiré varios tamaños y este es el que me permite visualizar mejor.
f, s1 = plt.subplots()
plt.suptitle('Distribución de las Varianzas de los Pixeles', size = 'large')

sns.histplot(data = varianzas_pixeles, bins = nbins, ax = s1)
s1.set_xlabel('Varianza') 
s1.set_ylabel('Frecuencia') 

"""
A partir de este gráfico podemos visualizar mejor cómo se distribuya la variaza a lo largo de todos los píxeles.
Es fácil notar que tenemos múchos píxeles con varianza pequeña. Esto se corresponde con que el primer bin más cercano a cero,
es el que mayor altura tiene; siendo este bin el que almacena a aquellos pixeles que tienen varianza muy muy cercana a cero.
Esto se corresponde con píxeles que siempre se mantienen apagados (siempre en negro) o encendidos (siempre en blanco). 
Sin embargo, también podemos observar que en el otro extremo del histograma, también hay una buena cantidad de datos.
Aquellos bins en los valores mayores del eje horizontal, se corresponden con las varianzas de valores altos. Notamos que 
tenemos muchos bins con alturas considerables, lo que nos dice que contamos con una buena cantidad de píxeles que tienen 
varianza muy alta. Estos son los píxeles que vamos a buscar priorizar para entrenar los modelos. Consideramos que estos son 
los que van a marcar la diferencia a la hora de entrenar y mejorar la taza de acierto en las predicciones.
"""


"""
Construyo un Heatmap a partir de las varianzas. A ver si este gráfico aporta algo de 
información inetresante...
"""
heatmap_varianzas = varianzas_pixeles.values.reshape(28, 28)

plt.figure(figsize=(6, 6))
sns.heatmap(heatmap_varianzas, cmap="viridis", cbar=True)
plt.title("Mapa de Calor de la Varianza de Cada Píxel")
plt.axis('off')
plt.show()

"""
En este gráfico podemos observar mejor cuáles son aquellas zonas que, en general, menor varianza tienen. Es decir, 
cuáles son las zonas que menos cambian a lo largo de todas las imágenes. Dándonos una idea de cuáles son los
píxeles que deberíamos descartar. Podemos observar que las zonas bordes y centrales mantienen colores oscuros. 
Estos representan píxeles con varianza pequeña, candidatos a ser descartados para el entrenamiento de los modelos.

También notamos que aquellas zonas con colores más claros contienen a los píxeles de mayor varianza. Estos van a ser 
los que más tomemos en cuenta para la etapa de entrenamiento. Ya que son los que más varían imagen a imagen, y los que
mejor caracterizan los cambios de Clases.
"""


# --------------------------------------------------------------------------------------------------------------------------------------- #
# %%
"""
A partir del gráfico (heatmap de la sección anterior), voy a definir un umbral de varianza mínimo. Me voy a quedar con todos los píxeles 
que tienen varianza mayor a ese umbral.
Busco que el umbral sea lo mas ajustado posible a aquellas zonas con píxeles de muy poca varianza. 

Visualizando el gráfico, determino el UMBRAL = 8900

Este umbral parece incluír a los píxeles que pertenecen a los bordes (de las imágenes) y al centro. Coherente con esta idea de que,
en todas las ímagenes, los bordes siempre son colores muy cercanos al negro (absoluto), y el centro son colores muy cercanos al 
blanco (absoluto). 
"""
umbral_global:int = 8900


"""
Como dijo Jack, vamos por partes...
Primero tomo el DataFrame 'varianzas_pixeles' y armo una lista donde almacene los index de las filas que tienen valor mayor a mi
umbral.
"""
columnas_utiles_global:list[str] = varianzas_pixeles[varianzas_pixeles > umbral_global].index.tolist() 

"""
Ahora voy al DataFrame con los datos originales, y me quedo sólo con las columnas que rescaté.
"""
columnas_utiles_con_label = columnas_utiles_global + ['label']
data_filtrada_1 = data_fashion[columnas_utiles_con_label]

"""
Ahora sí vamos por clase. 
Resulta que, si miramos clase por clase, aquellos píxeles más representativos de cada clase son aquellos píxeles que menor varianza tienen
(dentro de esa clase). Un ejemplo: tengo 10 remeras mangas cortas, cada una con un estampado diferente. Le calculo la varinza a los píxeles
de esa clase:
    - Los píxeles con mayor varianza van a ser aquellos donde se encuentra la estampa de cada remera.
    - Los píxeles con menor varianza van a ser aquellos que moldean la remera.
De esa forma, es fácil nota que aquellos píxeles que representan y mejor caracterizan a las remeras mangas corta, son aquellos que menos 
cambian entre una imagen y otra.

Generalizando esto para todas las clases, nos vamos a estar quedando con los píxeles que mejor caracterizan la figura de cada clase.
"""

grupo_por_clase = data_filtrada_1.groupby('label')   # Agrupo por clase.

varianzas_por_clase = grupo_por_clase.var()   # Varianza de cada píxel dentro de cada clase.
# print(varianzas_por_clase.head())


# --------------------------------------------------------------------------------------------------------------------------------------- #
# %% 
"""
A partir de ahora, defino dos clasificaciónes.

Varianza Intra-Clase: ¿Cuánto varía un píxel dentro de una clase?
- Me va a permitir identificar píxeles estables para una clase dada.

Varianza Inter-Clase: ¿Cuánto varía el valor promedio de un píxel entre las diferentes clases?
- Esta varianza me va a permitir encontrar píxeles que separan bien las clases: píxeles donde el valor medio cambia significativamente de 
una clase a otra.

Un buen píxel predictivo es aquel que:
- Tiene baja Varianza Intra-Clase (es estable dentro de cada clase).
- Tiene alta Varianza Inter-Clase (distingue entre clases).

Así que, a partir de ahora, el objetivo va a ser buscar píxeles que cumplan estas dos condiciones (sobre el conjunto de píxeles filtrados
en el paso anterior).
"""

# Necesito el promedio de cada píxel dentro de cada clase.
promedio_pixel_por_clase_1 = grupo_por_clase.mean()

# Para la Varianza Inter-Clase, calculo la varianza al promedio de cada píxel dentro de cada clase.
varianza_entre_clases = promedio_pixel_por_clase_1.var(axis=0)

# Para la Varianza Intra-Clase, calculo el promedio de las varianzas dentro de cada clase.
varianza_intra_clase = varianzas_por_clase.mean(axis=0)

"""
Grafico Histogramas para poder tener la información en forma visual.
A partir de estos gráficos voy a buscar umbrales para luego filtrar píxeles.
"""

plt.figure(figsize=(14, 5))

# Histograma 1: Varianza entre Clases.
plt.subplot(1, 2, 1)
plt.hist(varianza_entre_clases, bins=40, color='skyblue', edgecolor='black')
plt.title("Varianza entre Clases por Píxel")
plt.xlabel("Varianza entre Clases")
plt.ylabel("Cantidad de Píxeles")
plt.grid(True)

# Histograma 2: Varianza Intra-Clase Promedio.
plt.subplot(1, 2, 2)
plt.hist(varianza_intra_clase, bins=40, color='lightcoral', edgecolor='black')
plt.title("Varianza Intra-Clase Promedio por Píxel")
plt.xlabel("Varianza Intra-Clase")
plt.ylabel("Cantidad de Píxeles")
plt.grid(True)

plt.tight_layout()   # Hago un solo gráfico con ambas figuras.
plt.show()


"""
Análisis de Figuras:

Como especificamos antes, buscamos:
    - Gráfico Varianza entre clases por Píxel -> Umbral alto.
    - Gráfico Varianza Intra-Clase Promedio por Píxel -> Umbral bajo.
    - Tener un volumen de datos considerado.

A partir ded estos requisitos, definimos:
    - Umbral Entre Clases -> 4000 (y tomo todos los píxeles posteriores)
    - Umbral Intra Clases -> 4400 (y tomo todos los píxeles anteriores)
"""

umbral_entre_clases:int = 4000 
umbral_intra_clase:int = 4400


"""
Contruyo la condición que van a tener que cumplir los píxeles que quiero capturar.
   - Varianza entre Clases superior a 4000.
   - Varianza Intra-Clase inferior a 4400.
"""

pixeles_utiles = (varianza_entre_clases > umbral_entre_clases) & (varianza_intra_clase < umbral_intra_clase)

# Filtro los píxeles (en este caso, me quedo con el nombre de las columnas).
columnas_utiles = varianza_entre_clases.index[pixeles_utiles]

# Ahora sí, capturo aquellos píxeles que quiero (conservando también la columna 'label' con la clasificación).
columnas_finales = ['label'] + list(columnas_utiles) if 'label' in data_fashion.columns else list(columnas_utiles)
data_filtrado_final = data_fashion[columnas_finales]


#%% 
# A partir de lo observado en los gráficos promedio de cada clase, me armo una nueva condición para identificar clases, en específico las más similares

# Por ejemplo, las clases 2,4 y 6, que contienen imágenes muy similares
# -> me busco una manera que complemente lo que ya teniamos y que ayude a predecir mejor esas clases
X = data_fashion.drop(columns="label")
y = data_fashion["label"]
mean2 = X[y==2].mean(axis=0) #calculo los valores promedio de pixeles para cada una de esas clases
mean4 = X[y==4].mean(axis=0)
mean6 = X[y==6].mean(axis=0)

# Sumo las diferencias absolutas entre cada par de pixeles para luego quedarme con los de mayor distancia (mayor diferencia entre las clases)
delta = (mean2 - mean4).abs() + (mean2 - mean6).abs() + (mean4 - mean6).abs()

k = 15  
top_k = delta.sort_values(ascending=False).index[:k]

data_filtrado_final = pd.concat(
    [data_filtrado_final, data_fashion[top_k]],
    axis=1
)

#elimino las columnas duplicadas (puedo haber tomado pixeles que ya estaban)
data_filtrado_final = data_filtrado_final.loc[:, ~data_filtrado_final.columns.duplicated()] 



#%%
#hago lo mismo pero para las clases 5 y 7
X = data_fashion.drop(columns="label")
y = data_fashion["label"]
mean5 = X[y==5].mean(axis=0)
mean7 = X[y==7].mean(axis=0)

delta = (mean5 - mean7).abs()

k = 10  
top_k = delta.sort_values(ascending=False).index[:k]

data_filtrado_final = pd.concat(
    [data_filtrado_final, data_fashion[top_k]],
    axis=1
)

data_filtrado_final = data_filtrado_final.loc[:, ~data_filtrado_final.columns.duplicated()]

#%%
#nuevamente pero para las clases 0 y 6
X = data_fashion.drop(columns="label")
y = data_fashion["label"]
mean0 = X[y==0].mean(axis=0)
mean6 = X[y==6].mean(axis=0)

delta = (mean0 - mean6).abs()

k = 10  
top_k = delta.sort_values(ascending=False).index[:k]

data_filtrado_final = pd.concat(
    [data_filtrado_final, data_fashion[top_k]],
    axis=1
)

data_filtrado_final = data_filtrado_final.loc[:, ~data_filtrado_final.columns.duplicated()]


#%%

# Lo paso a CSV.
data_filtrado_final.to_csv("C:/Users/maxim/Desktop/labo datos/TP/tp_2/archivos/Data_Filtrada.csv", index=False) # CAMBIAR

#%%

############### KNN ################ 


# el inciso A pide un nuevo dataset que solo contenga las clases 0 y 8
# para ello se realiza una consulta SQL con el fin de filtrar y seleccionar las clases correspondientes

consulta = duckdb.query(""" 
                        SELECT * 
                        FROM data_filtrado_final 
                        WHERE label = 0 OR label = 8
                        """)
archivo_filtrado = consulta.df()


# OBSERVACION: hay 13999 imágenes de clase 0 o clase 8

# paso adicional: seleccionar atributos que más diferencian a las clases 0 y 8
# esto permite mejorar la separacion de las clases ya que hay mayor diferencia de media
X_df = archivo_filtrado.drop(columns="label")
y_df = archivo_filtrado["label"]

# calculo de diferencia de medias entre clases 0 y 8
mean_0 = X_df[y_df == 0].mean(axis=0)
mean_8 = X_df[y_df == 8].mean(axis=0)

# diferencia absoluta
delta = (mean_0 - mean_8).abs()

# selecciono los 12 píxeles más diferentes, podía elegir mas o menos.
# elijo este numero ya que en el inciso C me pide que tome n atributos y repita el analisis para distintos
# subconjuntos de n atributos. al elegir 12, puedo hacer 3 subconjuntos de 4 atributos.
k = 12
top_k = delta.sort_values(ascending=False).index[:k]

# crea un nuevo DataFrame solo con esos atributos más la clase
data_filtrado = pd.concat([archivo_filtrado, archivo_filtrado[top_k]], axis=1)
data_filtrado = data_filtrado.loc[:, ~data_filtrado.columns.duplicated()]

#%%

# Imprimir los píxeles seleccionados
print("Los 12 píxeles más discriminantes entre clase 0 y 8 son:")
for i, pixel in enumerate(top_k, 1):
    print(f"{i}. {pixel}")
# luego de realizar 3 prints observo que siempre me devuelve los mismos 12 atributos. (por suerte!)

# Creo subconjunto con esos 12 atributos, quizas luego lo utilice
subconjunto_elegido = list(top_k)



# en el inciso B pide separar los datos en conjuntos de testing y training.
# se decide usar 70% para training y 30% para testing.

# atributos que le ingreso  (todos salvo la clase)
X = data_filtrado.drop(columns="label").values

# atributo que quiero predecir (la clase)
Y = data_filtrado["label"]




# El inciso C me pide realizar un modelo de KNN, para ello primero decido realizar el modelo y luego lo testeo. Posteriormente hago el ajuste
# con una cantidad reducida de atributos    

# Entreno modelo 
model = KNeighborsClassifier(n_neighbors = 3) # modelo en abstracto
model.fit(X, Y) # entreno el modelo con los datos X e Y
Y_pred = model.predict(X) # me fijo qué clases les asigna el modelo a mis datos
metrics.accuracy_score(Y, Y_pred) # evaluo que tan bien predice
metrics.confusion_matrix(Y, Y_pred)  # Vemos la matriz de confusión

# testeo modelo
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3) # 70% para train y 30% para test

model = KNeighborsClassifier(n_neighbors = 6) # modelo en abstracto
model.fit(X_train, Y_train) # entreno el modelo con los datos X_train e Y_train
Y_pred = model.predict(X_test) # me fijo qué clases les asigna el modelo a mis datos X_test
print("Exactitud del modelo:", metrics.accuracy_score(Y_test, Y_pred)) 
metrics.confusion_matrix(Y_test, Y_pred)

# aca voy a utilizar el conjunto de 12 atributos mas discriminantes que hice previamente.
# Ajusto inicialmente con 4 atributos relevantes y luego pruebo con otros subconjuntos de 4 atributos
# los subconjuntos son:
    
# 1) ["pixel554", "pixel538", "pixel582", "pixel510"]
# 2) ["pixel470", "pixel566", "pixel482", "pixel442"]
# 3) ["pixel454", "pixel39", "pixel610", "pixel426"]


# NO hago codigo para los distintos 4 subconjuntos, simplemente copio y pego en la siguiente linea los distintos grupos.
subconjunto_elegido = ["pixel454", "pixel39", "pixel610", "pixel426"]
# me armo un dataframe que solo contenga la información de estas tres columnas.
# esto me permitirá observar como se comporta el KNN pero con menos información
X_subconjunto = data_filtrado[subconjunto_elegido].values
Y = data_filtrado["label"]

# vuelvo a separar los datos entre train (70%) y test (30%) 
# sub es la abreviacion de subconjunto
X_train_sub, X_test_sub, Y_train, Y_test = train_test_split(X_subconjunto, Y, test_size=0.3)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_sub, Y_train)

# evaluación y prediccion
Y_pred_red = model.predict(X_test_sub)

print("Exactitud con atributos reducidos:", metrics.accuracy_score(Y_test, Y_pred_red))
# la matriz de confusión binaria es una matriz 2X2
# defino que la posición de lee [fila,columna] empezando a contar a partir de 1 y finalizando en 2
# en la posicion [1,1] me indica los True Negative
# en la posicion [1,2] me indica los False Positive
# en la posicion [2,1] me indica los False Negative
# en la posicion [2,2] me indica los True Positive
print("Matriz de confusión:")
print(metrics.confusion_matrix(Y_test, Y_pred_red))


# Esta parte no es obligatoria pero este split permite hacer un analisis mas profundo y conocer mejores hiperparametros y valores de k.
# para cada valor de k vecinos elegidos, hay distintas precisiones
# IMPORTANTE: no es necesario modificar esta parte del codigo, solo modifico los atributos del subconjunto
Nrep = 5
valores_k = range(1, 20)  # probamos k = 1 a 19

# matrices para guardar los resultados
resultados_test = np.zeros((Nrep, len(valores_k)))
resultados_train = np.zeros((Nrep, len(valores_k)))

# Repetimos Nrep veces con distintas particiones de train y test
for i in range(Nrep):
    X_train_red, X_test_red, Y_train, Y_test = train_test_split(X_subconjunto, Y, test_size=0.3)

    for j, k in enumerate(valores_k):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_red, Y_train)

        # Predicciones
        Y_pred_test = model.predict(X_test_red)
        Y_pred_train = model.predict(X_train_red)

        # Métricas usadas
        acc_test = metrics.accuracy_score(Y_test, Y_pred_test)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)

        # Guardo resultados
        resultados_test[i, j] = acc_test
        resultados_train[i, j] = acc_train

# le calculo el promedio tras finalizar las repeticiones definidas
promedios_test = resultados_test.mean(axis=0)
promedios_train = resultados_train.mean(axis=0)

# Mostramos resultados de una manera mas legible y fácil de entender
for j, k in enumerate(valores_k):
    print(f"k = {k:2d} | Acc Test = {promedios_test[j]:.3f} | Acc Train = {promedios_train[j]:.3f}")

# ahora quiero graficar mis resultados 
# voy a hacer un gráfico de precisión promedio en testing y training para distintos k
# promedios_test: accuracy promedio en test para cada k
# promedios_train: accuracy promedio en train para cada k

plt.figure(figsize=(10, 6))

# funcion de Testing
plt.plot(valores_k, promedios_test, marker='o', label='Accuracy Test', color='blue')

# función de Training
plt.plot(valores_k, promedios_train, marker='s', label='Accuracy Train', color='green')

# Etiquetas y título
plt.xlabel('Número de vecinos (k)')
plt.ylabel('Precisión promedio')
plt.title('Precisión promedio en Train y Test según k (k-NN)')
plt.legend()
plt.grid(True)
plt.xticks(valores_k)
plt.ylim(0.7, 1.0) # escala para mostrar el gráfico
plt.tight_layout()
plt.show()



#%%

################ ARBOL DE DECISION ####################


#%%
# METODO
'''
1-Separo los datos en 2 partes:
    .Desarrollo (dev-train): que luego utilizaré para el cross validation y probar hiperparametros
    .Evaluacion (holdout-test): para una evaluación final
2-Evaluo los arboles desarrollados con diferentes profundidades usando kfold cross-validation
3-Calculo la precision promedio de cada modelo segun cada valor de profundidad
4-Elijo la profundidad cuyo accuracy promedio sea el mayor
5-Evaluo el modelo elegido en el holdout
6-Evalúo el arbol en mi dataset original como una verificacion final
'''

#%% cargamos los datos
datos_fin = pd.read_csv("C:/Users/maxim/Desktop/labo datos/TP/tp_2/archivos/Data_Filtrada.csv")  # CAMBIAR

X = datos_fin.drop(columns="label")
y = datos_fin["label"]
#%% separamos entre dev y eval
# separo mi conjunto en desarrollo y evaluación 
# 90% para dev y 10% para el holdout
X_dev, X_eval, y_dev, y_eval = train_test_split(X,y,
                                                random_state=1,
                                                test_size=0.1,
                                                stratify=y) 

#%% experimento
'''
En esta seccion voy probando el accuracy del arbol a medida que le voy cambiando su profundidad 
Es decir, voy entrenando el arbol con distintas profundidades, usando k-fold cross validation
Calculo el accuracy promedio (segun la precision que me va dando con cada fold) para cada valor de profundidad, para luego elegir la mejor profundidad

'''
from sklearn.model_selection import StratifiedKFold 
alturas = [1,2,5,7,10] #lista de profundidades para probar
nsplits = 5 #numero de folds usados
kf = KFold(n_splits=nsplits,shuffle=True,random_state=1)

resultados = np.zeros((nsplits, len(alturas)))
# una fila por cada fold, una columna por cada modelo

for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for j, hmax in enumerate(alturas): # se entrena al arbol con cierta profundidad y se lo evalúa según el test
        
        arbol = tree.DecisionTreeClassifier(max_depth = hmax,random_state=1)
        arbol.fit(kf_X_train, kf_y_train)
        pred = arbol.predict(kf_X_test)
        
        score = accuracy_score(kf_y_test,pred)
        
        resultados[i, j] = score # matriz de 5x5 con los disintos resultados de cada profundidad
#%% promedio scores sobre los folds
scores_promedio = resultados.mean(axis = 0) # promedio obtuvido de los 5 folds para cada profundidad


#%% 
#Accuracy promedio de cada profundidad
for i,h in enumerate(alturas):
    print(f'Score promedio del modelo con hmax = {h}: {scores_promedio[i]:.4f}')

#%% entreno el modelo elegido en el conjunto dev entero
mejor_indice = np.argmax(scores_promedio) # elijo el modelo con mayor exactitud promedio en el proceso de kfold cross-validation
mejor_altura = alturas[mejor_indice] #profundidad 10

arbol_elegido = tree.DecisionTreeClassifier(max_depth = mejor_altura, random_state=1)
arbol_elegido.fit(X_dev, y_dev) # lo entreno en mi conjunto de datos total de desarrollo y lo testeo
y_pred = arbol_elegido.predict(X_dev)

print("Evaluación en conjunto de desarrollo:")
print("Accuracy:", accuracy_score(y_dev, y_pred))
print("Precision macro:", precision_score(y_dev, y_pred, average='macro'))
print("Recall macro:", recall_score(y_dev, y_pred, average='macro'))
print("F1 macro:", f1_score(y_dev, y_pred, average='macro'))
print("Matriz de confusión:\n", confusion_matrix(y_dev, y_pred))


#%% pruebo el modelo elegido y entrenado, en el conjunto eval - holdout
#calculo el accuracy final
y_pred_eval = arbol_elegido.predict(X_eval)       
score_arbol_elegido_eval = accuracy_score(y_eval,y_pred_eval)
print("Evaluación en conjunto holdout: ", score_arbol_elegido_eval)


#%%
# Finalmente, para tener un panorama completo del rendimiento del arbol, lo evaluo en mi dataset original

# Selecciono solo las columnas que use en datos_fin, para pasarle al arbol los pixeles que conoce
features_entrenamiento = datos_fin.drop(columns="label").columns
X_orig = data_fashion[features_entrenamiento] #evaluo el arbol solo en los pixeles q mi modelo ya conoce
y_orig = data_fashion["label"]

y_pred_orig = arbol_elegido.predict(X_orig)

print("### Evaluación sobre DATASET ORIGINAL ###")
print("Accuracy: ", accuracy_score(y_orig, y_pred_orig))
print("Precision (macro): ", precision_score(y_orig, y_pred_orig, average="macro"))
print("Recall (macro): ", recall_score(y_orig, y_pred_orig, average="macro"))
print("F1 (macro): ", f1_score(y_orig, y_pred_orig, average="macro"))
print("Matriz de confusión:\n", confusion_matrix(y_orig, y_pred_orig))