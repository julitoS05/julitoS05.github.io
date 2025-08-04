# Laboratorio de Datos - TP 1. 
'''
Integrantes -> Dorogov Cristina 
            -> Pucciarelli Francisco Lautaro 
            -> Salto Julian 
               
Script Principal -> Por favor, leer el ReadMe. Allí se encuentra bien detallado el contenido del archivo 
                    y detalles técnicos.
'''


# %%
'''
Módulos.
'''

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from pathlib import Path


# %%
'''
Etapa 1 - Procesamiento y Limpieza de DataSets.

Preparación de los Path a los diferentes archivos y carpeta de destino.
'''

base_path = Path(__file__).resolve().parent        # Obtener la ruta base del script actual.
originales_path = base_path / 'TablasOriginales'   # Ruta a la carpeta donde se encuentran los DataSets originales.
modelo_path = base_path / 'TablasModelo'           # Ruta a la carpeta donde se van a guardar los DataSets limpios.
modelo_path.mkdir(exist_ok=True)                   # Si no existe la carpeta de salida, crearla.


'''
ACLARACIÓN IMPORTANTE.
Todas las lineas con la instrucción 'print()' que dejo comentadas, es porque es algo intermedio que usé para ver 
si funcionaba bien lo que iba haciendo.
'''


# %%
'''
Limpieza y Formateo de DataSet -> Establecimientos Educativos.
'''

'''
Lectura del DataFrame original.
Excluyo las primeras 5 filas, que tienen el rótulo (dedicadas al título y otros datos propios del Dataset).
La columna 'Código de localidad' quiero que la lea originalmente como 'string', porque luego quiero trabajar esa 
columna en forma particular: hay valores que empiezan en cero, Pandas los lee como 'int' y elimina estos ceros; pero luego los 
necesito (en los ultimos pasos toma sentido).
'''

establecimientos_educativos = pd.read_excel(originales_path / 'Padron_Oficial_Establecimientos_Educativos_2022.xlsx', 
                                            skiprows=6,
                                            dtype={'Código de localidad': str})





# ------------------------------------------------------------------------------------------------------------------------------- # 
'''
Me molestan mucho esas columnas categóricas agrupadas bajo "Modalidad". 
Quiero eliminarlas todas y dejar su información en una sola columna llamada "Modalidad".
'''

modalidades:list[str] = ['Común', 'Especial', 'Adultos', 'Artística', 'Hospitalaria', 'Intercultural', 'Encierro']
establecimientos_educativos['Modalidad'] = establecimientos_educativos[modalidades].apply(lambda row: ', '.join([col for col in modalidades if row[col] == 1]), axis=1) 
# print(establecimientos_educativos['Modalidad']) 

'''
Elimino todas las columnas que simplifiqué.
'''

establecimientos_educativos.drop(columns=modalidades, inplace=True) 


# ------------------------------------------------------------------------------------------------------------------------------- # 
'''
Para las sub-categorías (dentro de cada modalidad específica), me gustaría tener esa información en DataFrames separados. 
Para cada modalidad, voy a crear un DataFrame con las columnas 'Cueanexo' (clave de identificación de los distintos establecimientos) 
y las columnas categóricas de la sub-catgoría a la que pertenecen. 
Todo esto con el objetivo de modularizar la información y alivianar el DataSet principal.
'''

establecimientos_comunes = establecimientos_educativos[establecimientos_educativos["Modalidad"] == "Común"][["Cueanexo", 
                                                                                                             "Nivel inicial - Jardín maternal", 
                                                                                                             "Nivel inicial - Jardín de infantes", 
                                                                                                             "Primario", 
                                                                                                             "Secundario", 
                                                                                                             "Secundario - INET", 
                                                                                                             "SNU", 
                                                                                                             "SNU - INET"]] 
# print(establecimientos_comunes) 
# print(establecimientos_comunes.columns) 

establecimientos_especiales = establecimientos_educativos[establecimientos_educativos["Modalidad"] == "Especial"][["Cueanexo", 
                                                                                                                   "Nivel inicial - Educación temprana", 
                                                                                                                   "Nivel inicial - Jardín de infantes", 
                                                                                                                   "Primario", 
                                                                                                                   "Secundario", 
                                                                                                                   "Integración a la modalidad común/ adultos"]]
# print(establecimientos_especiales) 
# print(establecimientos_especiales.columns) 

establecimientos_adultos = establecimientos_educativos[establecimientos_educativos["Modalidad"] == "Adultos"][["Cueanexo", 
                                                                                                               "Primario", 
                                                                                                               "Secundario", 
                                                                                                               "Alfabetización", 
                                                                                                               "Formación Profesional", 
                                                                                                               "Formación Profesional - INET"]]
# print(establecimientos_adultos) 
# print(establecimientos_adultos.columns)

establecimientos_artisticas = establecimientos_educativos[establecimientos_educativos["Modalidad"] == "Artística"][["Cueanexo", 
                                                                                                                    "Secundario", 
                                                                                                                    "SNU", 
                                                                                                                    "Talleres"]]
# print(establecimientos_artisticas) 
# print(establecimientos_artisticas.columns)

establecimientos_hospitalarias = establecimientos_educativos[establecimientos_educativos["Modalidad"] == "Hospitalaria"][["Cueanexo", 
                                                                                                                          "Inicial",
                                                                                                                          "Primario", 
                                                                                                                          "Secundario"]]
# print(establecimientos_hospitalarias)
# print(establecimientos_hospitalarias.columns)

'''
Es verdad que solamente vamos a trabajar con los Establecimientos Educativos de Modalidad Común, pero por las dudas tengo esta
informacion también limpia y disponible. 

La columna llamada "Servicios Complementarios" no la vamos a tomar en cuenta para nuestro análisis. Luego la excluyo.
'''


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Ahora armo un DataFrame aparte que contenga la información que vamos a utilizar (según el DER planteado) de cada una de los
Establecimientos Educativos.
'''

atributos_de_interes_1:list[str] = ["Cueanexo",
                                    "Nombre", 
                                    "Jurisdicción", 
                                    "Código de localidad", 
                                    "Localidad", 
                                    "Departamento",
                                    "Sector", 
                                    "Ámbito"]

establecimientos_educativos_clean = establecimientos_educativos[atributos_de_interes_1]
establecimientos_educativos_clean.columns = ["id_establecimiento",
                                             "nombre",
                                             "nombre_provincia",
                                             "id_localidad",
                                             "nombre_localidad",
                                             "nombre_departamento",
                                             "sector",
                                             "ambito"]


'''
La información de las que llamo "columnas categóricas" lo voy a guardar aparte.
Lo voy a hacer sólo con la Modalidad Común (que es la que vamos a usar, a priori). 
Pero antes, las columnas con valores 'vacíos' (que son muchas) quiero que tengan cero en vez de "vacío". Sobre todo para prevenir 
problemas con "Nulls" cuando vayamos a hacer consultas SQL (y para que quede más prolijo).
'''

columnas_categoricas:list[str] = ["Nivel inicial - Jardín maternal",
                                  "Nivel inicial - Jardín de infantes",
                                  "Primario",
                                  "Secundario",
                                  "Secundario - INET",
                                  "SNU",
                                  "SNU - INET"]

establecimientos_comunes[columnas_categoricas] = establecimientos_comunes[columnas_categoricas].replace(r'^\s*$', pd.NA, regex=True)
'''
Ejecutando el código, en las dos líneas siguientes me saltó un error porque muchas columnas tienen espacios vacíos o 
strings vacíos. Con las líneas de abajo busco reemplazar esas columnas por 0, pero primero las transformo a 'Nan' de pandas. 
Las rastreo usando una expresión regular.
'''

establecimientos_comunes[columnas_categoricas] = establecimientos_comunes[columnas_categoricas].fillna(0)
'''
El 'fillna()' creo que convierte todo a 'floats'. Me interesa que todo sea 'int', para evitar posibles problemas
luego (uno nunca sabe... a veces esos detalles te hacen perder toda una tarde buscando ese error pavo). Así que, 
por las dudas, lo typeo de nuevo.
'''
establecimientos_comunes[columnas_categoricas] = establecimientos_comunes[columnas_categoricas].astype(int) 

'''
Como especificamos en nuestro DER, las columnas 'Secundario' y 'Secundario - INET' las vamos a centralizar en una sola columna 
llamada 'Secundario'. Esto porque no nos interesa la sub-categoría 'INET', sólo nos interesa que sea modalidad Secundario.
Ahora, para ser prolijo, no voy a crear una nueva columna sino voy a modificar la columna "Secundario" (porque no me interesa 
preservar exactamente la información que tenía).
'''

establecimientos_comunes["Secundario"] = ((establecimientos_comunes["Secundario"] == 1) | (establecimientos_comunes["Secundario - INET"] == 1)).astype(int)

'''
Hago lo mismo que antes, pero ahora combino 'SNU' y 'SNU - INET' en la columna 'Terciario'.
Luego, elimino las columnas que combiné.
'''

establecimientos_comunes["Terciario"] = ((establecimientos_comunes['SNU'] == 1) | (establecimientos_comunes['SNU - INET'] == 1)).astype(int) 

establecimientos_comunes = establecimientos_comunes.drop(columns=["Secundario - INET", "SNU", "SNU - INET"]) 

'''
Cambio los nombres de las columnas.
'''

establecimientos_comunes.columns = ["id_establecimiento",
                                    "jardin_maternal",
                                    "jardin_infantes",
                                    "primario",
                                    "secundario",
                                    "terciario"]


'''
Los demás DataFrames (de los otros tipos de establecimientos educativos) los dejo así como están, porque creemos no los 
vamos a usar. Si llegamos a necesitarlos, les hago un tratamiento parecido; pero por ahora, no.
'''


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Para poder relacional este DataSet con los demás (a través de consultas SQL y demás), necesito que la columna 'id_localidad' 
tenga los primeros 5 dígitos del número que aparece. 
Observamos que los primeros 5 dígitos corresponden al id del departamento. Como cada departamento tiene sus propias localidades, 
los próximos digitos diferencian las localidades. Pero a nosotros nos basta con los departamentos.
Aquellos valores que empiezan con cero, voy a eliminar ese primer cero y luego tomar cinco dígitos a partir de allí.
'''

establecimientos_educativos_clean['id_localidad'] = establecimientos_educativos_clean['id_localidad'].str[:5].astype(int)
'''
Esta línea, a cada elemento de la columna 'id_localidad', lo transforma a string, toma 5 caracteres y lo vuelve a parsear a 
'int' (eliminando ceros que puedan quedar por delante). Los únicos valores que tienen ceros delante son los correspondientes a 
las provincias de Buenos Aires y los correspondientes a CABA. Pero, en el caso de los de Buenos Aires, en el DataSet de Bibliotecas
Populares tienen también 4 dígitos; y los de CABA por ahora no importan (les voy a hacer otro tratamiento después).
'''


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Hay un último tema a solucionar. En el DataSet de Bibliotecas Populares, aquellas bibliotecas que pertenecen a la Ciudad Autónoma
de Buenos Aires no están clasificadas por departamento. Es decir, tanto su localidad (provincia) como su departamento son 'Ciudad 
Autónoma de Buenos Aires'. Y en 'id_departamento' tienen el id de la Ciudad Autónoma de Buenos Aires (como provincia, que es 
'02').
Por lo tanto, vamos a tener que trabajar 'Ciudad Autónoma de Buenos Aires' sin la división en departamentos. Por lo tanto, todos
los Establecimientos Educativos que pertenecen a la 'Ciudad de Buenos Aires' voy a ponerles como 'id_departamento' el código de 
la provincia; para luego poder machear establecimientos educativos con bibliotecas populares que pertenecen a CABA.
Es una lástima, pero bueno... los datos son los datos y nos obligar a trabajrlos así (para poder mantener un orden y más coherencia
en el filtrado y limpieza de datos).
'''

establecimientos_educativos_clean.loc[establecimientos_educativos_clean['nombre_localidad'] == 'CIUDAD DE BUENOS AIRES', 'id_localidad'] = 2000
'''
El 'id_departamento' que usa el Dataset para identificar a las bibliotecas que pertenecen a la Ciudad de Buenos Aires es 2000.
'''

'''
Ahora cambio el nombre a todas las filas cuyo departamento se corresponde con alguno de CIudad Autónoma de Buenos Aires. 
Ahora todos estos establecimientos pertenece a "CIUDAD DE BUENOS AIRES".
'''

departamentos_caba = ["Comuna 1", "Comuna 2", "Comuna 3", "Comuna 4", "Comuna 5", 
                      "Comuna 6", "Comuna 7", "Comuna 8", "Comuna 9", "Comuna 10", 
                      "Comuna 11", "Comuna 12", "Comuna 13", "Comuna 14", "Comuna 15"]

for i, fila in establecimientos_educativos_clean.iterrows():
    if fila["nombre_departamento"] in departamentos_caba:
        establecimientos_educativos_clean.at[i, "nombre_departamento"] = "CIUDAD DE BUENOS AIRES"
        
        
# ------------------------------------------------------------------------------------------------------------------------------- # 
'''
Siguiendo nuestro Modelo Relacional, en el DataSet correspondiente a los Establecimientos Educativos sólo vamos a mantener la 
columna 'id_establecimiento'. Así que las columnas de 'nombre_provincia', 'nombre_departamento' y 'nombre_localidad' debo 
sacarlas. Pero primero, voy a aprovechar que tengo bastante información para poder armar la otra relación que aparece en 
nuestro DER: 'departamentos'. La construyo con SQL (para aumentar la diversidad de herramientas).
'''

consultaAux = duckdb.query(''' 
                       SELECT id_localidad AS id_departamento, ANY_VALUE(nombre_departamento) AS nombre_departamento, ANY_VALUE(nombre_provincia) AS nombre_provincia
                       FROM establecimientos_educativos_clean
                       GROUP BY id_departamento
                       ORDER BY id_departamento
                       ''')
departamentos = consultaAux.df() 

'''
Pero hay un último detalle. Hay muchos 'nombre_departamento' que dicen 'CAPITAL', sin el nombre de la provincia no queda claro 
capital de qué provincia se refiere. Para que quede mas explícito y fácil de interpretar, voy a cambiar esos valores y juntar 
'CAPITAL' con 'nombre_provincia' (para que, por ejemplo, en vez de 'CAPITAL' diga 'CATAMARCA CAPITAL').
'''
departamentos.loc[departamentos['nombre_departamento'] == 'CAPITAL', 'nombre_departamento'] = departamentos['nombre_provincia'].str.upper() + ' CAPITAL'

'''
Ahora sí, termino de acomodar 'establecimientos_educativos_clean'
'''
establecimientos_educativos_clean = establecimientos_educativos_clean.drop(columns=['nombre_provincia','nombre_departamento','nombre_localidad'])
establecimientos_educativos_clean.columns = ["id_establecimiento",
                                             "nombre_establecimiento",
                                             "id_departamento",
                                             "sector",
                                             "ambito"]


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Finalmente, paso todos los DataFrames (que vamos a usar) a archivos csv.
'''

establecimientos_educativos_clean.to_csv(modelo_path / 'establecimientos_educativos.csv', index=False)
print("Archivo guardado como 'establecimientos_educativos.csv', en la carpeta 'TablasModelo'")

establecimientos_comunes.to_csv(modelo_path / 'establecimientos_educativos_comunes.csv', index=False)
print("Archivo guardado como 'establecimientos_educativos_comunes.csv', en la carpeta 'TablasModelo'")

departamentos.to_csv(modelo_path / 'departamentos.csv', index=False)
print("Archivo guardado como 'departamentos.csv', en la carpeta 'TablasModelo'")


#%%
'''
Limpieza y Formateo de DataSet -> Bibliotecas Populares.
'''

# Lectura del DataSet original.
bibliotecas = pd.read_csv(originales_path / 'Bibliotecas_Populares.csv')


#Limpieza dataset Biblioteca
bibliotecas.drop(columns=["tipo_latitud_longitud", 
                          "telefono", 
                          "web", 
                          "domicilio", 
                          "observacion", 
                          "latitud", 
                          "longitud", 
                          "subcategoria", 
                          "fuente", 
                          "anio_actualizacion", 
                          "cod_tel", 
                          "piso", 
                          "cp", 
                          "informacion_adicional"], 
                 inplace=True)


# Me armo una tabla aparte q tome lo mas importante para las consultas.
biblioteca_2 = bibliotecas[["nro_conabip","nombre","id_departamento","departamento","provincia","mail","fecha_fundacion"]].copy()

biblioteca_2["mail"] = biblioteca_2["mail"].str.split('@').str[1]
biblioteca_2["mail"] = biblioteca_2["mail"].str.split('.').str[0]
biblioteca_2["fecha_fundacion"] = pd.to_datetime(biblioteca_2["fecha_fundacion"])
biblioteca_2["año_fundacion"] = biblioteca_2["fecha_fundacion"].dt.year
biblioteca_2.drop(columns = ["fecha_fundacion"], inplace = True) 


# Cambio el nombre de las columnas.
biblioteca_2.columns = ["id_biblioteca", 
                        "nombre_biblioteca",
                        "id_departamento",
                        "nombre_departamento",
                        "nombre_provincia",
                        "dominio_email",
                        "año_fundacion"] 


# Para respetar el MR, tomo solo las columnas que explicitamos allí. 
columnas_MR:list[str] = ["id_biblioteca", "nombre_biblioteca", "id_departamento", "año_fundacion", "dominio_email"]
biblioteca_clean = biblioteca_2[columnas_MR]

# Exporto a csv el DataSet limpio (en su carpeta correspondiente).

biblioteca_clean.to_csv(modelo_path / 'bibliotecas_populares.csv', index=False)
print("Archivo guardado como 'bibliotecas_populares.csv', en la carpeta 'TablasModelo'")


# %%
'''
Limpieza y Formateo de DataSet -> Población por Edad por Departamento.
'''

'''
Lectura del DataFrame original.
Le digo que lea a partir de la fila 13 (que se saltee las primeras 12 filas), ya que esas filas tienen el rótulo del DataSets (datos
como el título, la fecha, etc). No quiero esa información ahora porque me complica el trabajo de limpieza y parseo.
'''

poblacion_por_edad_original = pd.read_excel(originales_path / "Poblacion_por_Edad_por_Departamento.xlsx",
                   skiprows=12,
                   header=None)


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Voy a generar un Dataframe con la información que necesito.
El Dataset original está dividido en Sub-Tablas consecutivas. Eso es un problema, ya que parsear a un DataFrame que me de la 
información (de todas las filas) en columnas no es directo. 
Mi idea es ir recorriendo todas las filas, e ir completando el nuevo DataFrame con los datos de las filas que necesito. Sé de antemano
cuáles son las columnas que quiero armar, y sé cuáles son los datos de cada sub-tabla que van en cada columna.
'''

# Quito las filas completamente vacías y ajusto los índices (los reseteo).
poblacion_por_edad_aux = poblacion_por_edad_original.dropna(how='all').reset_index(drop=True)

'''
La idea es armar una lista con todas las filas del nuevo DataFrame (mi DataFrame final).
Luego, voy a armar un nuevo DataFrame a partir de esta lista.
'''
resultados = []

'''
Voy a iterar sobre todas las filas viendo qué hay en la columna dos (la columna 'B' si abro con excel el archvo original).
Depende de lo que me encuentre (un código de Area, la fila correspondiente a una edad en una sub-tabla, o cualuqier otra cosa), 
voy a hacer lo que deba hacer con la información de la columna siguiente (de esa misma fila), que es la columna que tiene: o el 
nombre del departamento, o la cantidad de habitantes de ese departamento de esa edad.
'''

i = 0
while i < len(poblacion_por_edad_aux):
    fila = poblacion_por_edad_aux.iloc[i]
    
    # Identificar inicio de una nueva sub-tabla por "AREA #".
    if isinstance(fila[1], str) and "AREA #" in fila[1]:
        # Extraigo ID y nombre del departamento.
        id_departamento = fila[1].split("AREA #")[-1].strip()
        nombre_departamento = fila[2]
        
        # Avanzar hasta encontrar la fila de encabezado de la sub-tabla ("Edad", "Casos", etc).
        i += 1
        while i < len(poblacion_por_edad_aux) and not (poblacion_por_edad_aux.iloc[i][1] == "Edad"):
            i += 1
        i += 1  # Pasar a la primera fila de datos reales (en este punto encontré el encabezado de la sub-tabla).
        
        # Inicializo los acumuladores.
        jardin_maternal = jardin_infante = primaria = secundaria = terciario = poblacion_total = 0
        
        # Leer los datos de la sub-tabla hasta llegar a la fila "Total".
        while i < len(poblacion_por_edad_aux):
            edad = poblacion_por_edad_aux.iloc[i][1]
            casos = poblacion_por_edad_aux.iloc[i][2]
            
            # Si es "Total", tomar como población total y terminar bloque.
            if isinstance(edad, str) and edad.strip().lower() == "total":
                poblacion_total = casos
                i += 1
                break
            
            # Si la edad es numérica, sumarsela al acumulador correspondiente.
            if pd.api.types.is_number(edad):
                edad = int(edad) # Si no lo parseo como 'int', a veces se vuelve loco y lo interpreta como millones.
                if edad in [0, 1, 2, 3]:
                    jardin_maternal += casos
                elif edad in [4, 5]:
                    jardin_infante += casos
                elif 6 <= edad <= 12:
                    primaria += casos
                elif 13 <= edad <= 18:
                    secundaria += casos
                elif 19 <= edad <= 50:
                    terciario += casos
            i += 1
        
        # Construyo la fila y la guardo en 'resultado'. Uso diccionarios, para poder tener bien explícito cada columna.
        resultados.append({
            "id_departamento": id_departamento,
            "nombre_departamento": nombre_departamento,
            "jardin_maternal": jardin_maternal,
            "jardin_infante": jardin_infante,
            "primaria": primaria,
            "secundaria": secundaria,
            "terciario": terciario,
            "poblacion_total": poblacion_total
        })
    else:
        i += 1  # Avanzar si no es un inicio de sub-tabla.

# Creo el DataFrame final
df_resultado = pd.DataFrame(resultados)

# Reviso a ver qué es lo que cocinó... (o sea, qué es lo que hizo).
# print(df_resultado.head())


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Como hicimos en los demás DataSets, todos los departamentos de Ciudad Autónoma de Buenos Aires vamos a trabajarlos como si fueran 
uno. Por lo tanto, voy a hacer lo siguiente. Primero, modificar 'id_departamento' de todas aquellos cuyo 'nombre_departamento' sean 
'Comuna X'; el nuevo id va a ser 2000 (id de la provincia CIUDAD DE BUENOS AIRES que usamos en los demás DataSets).
'''

df_resultado.loc[df_resultado['nombre_departamento'].str.startswith('Comuna '), 'id_departamento'] = 2000 

'''
Ahora quiero combinar esas filas en una sola, que centralice toda la información de población de Ciudad de Buenos Aires.
'''
# Tomo las filas que tienen id_departamento = 2000.
df_caba = df_resultado[df_resultado['id_departamento'] == 2000]

# Sumo todas las columnas numéricas de esas filas.
df_caba_suma_total = df_caba.drop(columns=['nombre_departamento']).sum(numeric_only=True)

# Creo una nueva fila con los valores calculados (en realidad, le añado las columnas que le faltan).
df_caba_suma_total['id_departamento'] = 2000
df_caba_suma_total['nombre_departamento'] = 'Ciudad de Buenos Aires'

# Reordeno las columnas en el orden original.
columnas_ordenadas = df_resultado.columns
df_caba_final = pd.DataFrame([df_caba_suma_total])[columnas_ordenadas]

# Elimino las filas originales (del DataFrame original) con id_departamento = 2000.
df_resultado = df_resultado[df_resultado['id_departamento'] != 2000]

# Agrego la nueva fila al DataFrame original.
df_resultado = pd.concat([df_resultado, df_caba_final], ignore_index=True) 


'''
Hay una cosa mas por solucionar aquí. Hay varios 'id_departamentos' que empiezan con cero (creo que son todos los 
'id_departamento' correspondientes a Provincia de Buenos Aires). En los otros DataSets, estos ids los dejé con cuatro dígitos
(quité el cero de delante). Aquí voy a hacer lo mismo.
Para ello, creo que parseando toda la columna a 'int' me basta para eliminar los ceros del principio.
'''
df_resultado['id_departamento'] = df_resultado['id_departamento'].astype(int)


# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Hay un problema en el DataFrame resultado. La columna 'nombre_departamento' tiene el mismo valor en todos los departamentos que 
corresponden a la capital de una provincia. Es decir, todos los departamento que son capital de una provincia, su valor en la 
columna 'nombre_departamento' es "Capital" (para todos). Teniendo en cuenta que contamos con una tabla aparte donde tenemos 
cada 'id_departamento' asociado al nombre del mismo, para que no sea redundante, voy a sacar la columna 'nombre_departamento' 
del DataFrame donde clasifico la población.
'''

df_resultado = df_resultado.drop(columns=["nombre_departamento"]) 
# print(df_resultado.head())

'''
También resulta que los 'id_departamento' correspondientes a Tierra del Fuego, tampoco son los mismos 'id_departamento'
que los que aparecen en el DataSets de Bibliotecas Populares. Para que las relaciones y uniones entre tablas se realicen correctamente,
deben coincidir. 
Como son solamente dos casos aislados y particulares, los modifico a mano. Utilizo el 'id_departamento' del DataSets de Bibliotecas 
Populares.
'''

df_resultado.iloc[511, 0] = 94014
df_resultado.iloc[509, 0] = 94009

# ------------------------------------------------------------------------------------------------------------------------------- #
'''
Con todo funcionando, estoy listo para crear el archvio CSV con los datos limpios.
'''

df_resultado.to_csv(modelo_path / 'nivel_educativo_por_departamento.csv',
                    index=False)
print("Archivo guardado como 'nivel_educativo_por_departamento.csv', en la carpeta 'TablasModelo'")


# %%
'''
Etapa 2 - Consultas SQL (sobre los DataSets limpios y procesados).
'''

'''
Preparación de los Path a los diferentes archivos y carpeta de destino.
'''

consultas_path = base_path / 'ConsultasSQL'        # Ruta a la carpeta donde se van a guardar los DataSets limpios.
consultas_path.mkdir(exist_ok=True)                # Crear la carpeta de salida para las consultas SQL. 

# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Lectura de DataSets.
'''

BP = pd.read_csv(modelo_path / "bibliotecas_populares.csv")
EE = pd.read_csv(modelo_path / "establecimientos_educativos.csv")
dept = pd.read_csv(modelo_path / "departamentos.csv")
EEcomunes = pd.read_csv(modelo_path / "establecimientos_educativos_comunes.csv")
nivel_educ_dept = pd.read_csv(modelo_path / "nivel_educativo_por_departamento.csv")


# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Consultas.
'''

### Consulta i.

# Sumo la cantidad de niveles educativos (por separado) que hay en cada departamento.

consulta_i = duckdb.query(""" 
                          SELECT
                            ANY_VALUE(EE.id_departamento) as id_dept, 
                            sum(EEcomunes.jardin_maternal) AS jardin_maternal,          --Sumo la cantidad de cada nivel educativo.
                            sum(EEcomunes.jardin_infantes) AS jardin_infantes,      
                            sum(EEcomunes.primario) AS primarias,
                            sum(EEcomunes.secundario) AS secundarios,
                            ANY_VALUE (dept.nombre_departamento) AS departamento,
                            ANY_VALUE(dept.nombre_provincia) AS Provincia
                          FROM EEcomunes 
                          INNER JOIN EE 
                            ON EE.id_establecimiento = EEcomunes.id_establecimiento     --Joineo EEcomunes con EE para quedarme solo con los establecimientos comunes.
                          INNER JOIN dept
                            ON dept.id_departamento = EE.id_departamento                --Joineo la tabla (con la que vengo trabajando) con la tabla dept para obtener el nombre del departamento y de la provincia.
  
                          GROUP BY dept.id_departamento                                 --Agrupo por id_departamento.

                         """) 
dfi = consulta_i.df()  

consultaSQL2 = duckdb.query(""" 
                            SELECT 
                              dfi.Provincia,  
                              dfi.jardin_maternal,
                              nivel_educ_dept.jardin_maternal AS Poblacion_maternal,
                              dfi.jardin_infantes,
                              nivel_educ_dept.jardin_infante AS Poblacion_infante,
                              dfi.primarias,
                              nivel_educ_dept.primaria AS poblacion_primaria,
                              dfi.secundarios,
                              nivel_educ_dept.secundaria AS poblacion_secundaria,
                              dfi.departamento,
                              nivel_educ_dept.poblacion_total AS poblacion_total       --Agrego la población total (útil luego para los gráficos)
                            FROM dfi 
                            INNER JOIN nivel_educ_dept 
                              ON dfi.id_dept = nivel_educ_dept.id_departamento

                            """) 
dfi2 = consultaSQL2.df()   # Básicamente, la misma tabla que 'dfi', pero le agrego la cantidad de población por nivel educativo (haciendo JOIN entre 'dfi' y 'nivel_educ_dept').

dfi2["Jardines"] = dfi2["jardin_infantes"] + dfi2["jardin_maternal"]    # Junto 'jardin_infantes' con 'jardin_maternal' para tener una sola columna de jardin (como se pide).
dfi2["Poblacion Jardin"] = dfi2["Poblacion_infante"] + dfi2["Poblacion_maternal"]   # Junto también sus poblaciones.
dfi2 = dfi2.drop(columns=["Poblacion_infante","Poblacion_maternal","jardin_infantes","jardin_maternal"])   # Elimino columnas innecesarias.
dfi2 = dfi2[["Provincia","departamento","Jardines","Poblacion Jardin","primarias","poblacion_primaria","secundarios","poblacion_secundaria", "poblacion_total"]] # Ordeno las columnas.


## Consulta ii.

# En esta consulta se busca obtener la cantidad de BP fundadas desde 1950, agrupadas según su departamento.

consulta_ii = duckdb.query(""" 
                          SELECT dept.nombre_provincia as provincia, dept.nombre_departamento as departamento, count(BP.año_fundacion) AS cantidad_BP_fundadas_desde_1950     --Tomo los datos necesarios.

                          FROM BP
                          
                          INNER JOIN dept                                           --Conecto la tabla BP con la de 'dept' para obtener la provincia y el nombre del departamento a partir de 'id_departamento' (clave en ambas tablas).
                          ON dept.id_departamento = BP.id_departamento

                          WHERE BP.año_fundacion >= 1950                            --Filtro según año.

                          GROUP BY dept.nombre_provincia,dept.nombre_departamento

                          ORDER BY dept.nombre_provincia, cantidad_BP_fundadas_desde_1950 DESC
                          """) 
dfii = consulta_ii.df()


## Consulta iii.

#En esta consulta se busca tomar la cantidad de BP y EE por departamento, junto con su población.

aux = duckdb.query('''
                   SELECT id_departamento, count(id_biblioteca) as cantBP 
                   FROM BP
                   GROUP BY id_departamento                   
                   ''')
cantBPdept = aux.df()   # Toma la cantidad de bibliotecas que hay por departamento.

aux2 = duckdb.query('''
                    SELECT EE.*
                    FROM EE
                    INNER JOIN EEcomunes 
                        ON EE.id_establecimiento = EEcomunes.id_establecimiento                 
                   ''')
a2 = aux2.df()   # Me quedo con los EE comunes (como se pide) para luego contarlos según el departamento.

aux2 = duckdb.query('''
                    SELECT id_departamento, count(id_establecimiento) as cantEE
                    FROM a2      
                    GROUP BY id_departamento
                   ''')
cantEEdept = aux2.df()


# Me voy armando tablas temporales para mantener el código encapsulado y organizado.

consulta_iii = duckdb.query(""" 

WITH tabla_temp AS(                                                            --Me quedo sólo con los establecimientos comunes.
    SELECT *
    FROM EE
    INNER JOIN EEcomunes 
        ON EE.id_establecimiento = EEcomunes.id_establecimiento
),
tabla_temp2 AS(                                                                --Conecto la tabla temporal anterior con la de 'bibliotecas', y me quedo con las columnas necesarias.         
    SELECT 
        tabla_temp.id_establecimiento,
        tabla_temp.id_departamento,
        BP.id_biblioteca
    
    FROM tabla_temp
    
    INNER JOIN BP
        ON BP.id_departamento = tabla_temp.id_departamento
)
SELECT                                                                         --Conecto 'tabla_temp2' con las de 'dept', 'nivel_educ_dept', 'cantBPdept' y 'cantEEdept' para obtener los datos necesarios.

    ANY_VALUE(dept.nombre_provincia) as Provincia,
    dept.nombre_departamento as Departamento,
    ANY_VALUE(cantBPdept.cantBP) as Cant_BP,
    ANY_VALUE(cantEEdept.cantEE) as Cant_EE,
    ANY_VALUE(nivel_educ_dept.poblacion_total) as Poblacion
FROM tabla_temp2

INNER JOIN dept
ON dept.id_departamento = tabla_temp2.id_departamento

INNER JOIN nivel_educ_dept
    ON tabla_temp2.id_departamento = nivel_educ_dept.id_departamento

INNER JOIN cantBPdept
    ON cantBPdept.id_departamento = tabla_temp2.id_departamento
    
INNER JOIN cantEEdept
    ON cantEEdept.id_departamento = tabla_temp2.id_departamento

GROUP BY Departamento

""") 

dfiii = consulta_iii.df()


## Consulta iv.

# Lo que se busca en esta consulta es quedarte con el dominio de mail más frecuente para cada departamento según sus bibliotecas.
# Me voy armando tablas temporales para mantener el código encapsulado y organizado.

consulta_iv = duckdb.query(""" 

WITH tabla_temporal AS (                                --Me armo una tabla temporal con los distintos dominios de mail no nulos que hay en cada departamento.
  SELECT 
    dept.nombre_provincia as provincia,
    dept.nombre_departamento as departamento,
    BP.dominio_email as mail 
  FROM BP
  INNER JOIN dept
  ON dept.id_departamento = BP.id_departamento
  WHERE mail IS NOT NULL
),
tabla_final AS (                                       --Tabla temporal para contar (y ordenar según) la cantidad de veces que aparece cada dominio de mail en cada departamento a partir de la tabla anterior.
  SELECT 
    ANY_VALUE(provincia) as provincia,
    departamento,
    mail,
    COUNT(*) AS cantidad
  FROM tabla_temporal
  GROUP BY departamento, mail
  ORDER BY cantidad DESC
)
SELECT                                                 --Consulta final que toma lo necesario de la tabla temporal anterior, el dominio de mail que más veces aparece en cada departamento, y agrupa por departamento.
    ANY_VALUE(provincia) as Provincia, 
    departamento as Departamento, 
    ANY_VALUE(mail) as Dominio_mas_frecuente, 
    MAX(cantidad) as max
FROM tabla_final
GROUP BY Departamento
ORDER BY Departamento

""") 
dfiv = consulta_iv.df()

dfiv = dfiv.drop(columns=["max"])   # Elimino la columna 'max' de la tabla final (me decía la cantidad de veces que aparecía cada dominio de mail por departamento).


# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Paso los 'df' a tipo CSV, y los guardo para usarlos luego al hacer los graficos.
'''

dfi2.to_csv(consultas_path / "dfi.csv", index=False)
dfii.to_csv(consultas_path / "dfii.csv", index=False)
dfiii.to_csv(consultas_path / "dfiii.csv", index=False)
dfiv.to_csv(consultas_path / "dfiv.csv", index=False)

print("Todas los resultados de las consultas exportados exitosamente (en formato CSV) en la carpeta 'Consultas_SQL'") 


# %%
'''
Etapa 3 - Visualización y Gráficos.
'''

'''
Preparación de los Path a los diferentes archivos y carpeta de destino.
'''

graficos_path = base_path / 'Graficos_y_Visualizaciones'   # Ruta a la carpeta donde se guardan gráficos generados.
graficos_path.mkdir(exist_ok=True)                         # Si no existe la carpeta de salida, crearla.

# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Lectura de DataSets.
'''

EE_dept = pd.read_csv(consultas_path / "dfi.csv")
BP_1950 = pd.read_csv(consultas_path / "dfii.csv")
cantidad = pd.read_csv(consultas_path / "dfiii.csv")


# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Tarea Preliminar.
'''

# Tomo el promedio de CABA (agrupa las 15 comunas).

EE_dept_copia = EE_dept.copy()
cantidad_copia = cantidad.copy()

index = EE_dept[EE_dept['departamento'] == 'CIUDAD DE BUENOS AIRES'].index

EE_dept_copia.iloc[index,2] = EE_dept_copia.iloc[index,2] / 15
EE_dept_copia.iloc[index,3] = EE_dept_copia.iloc[index,3] / 15
EE_dept_copia.iloc[index,4] = EE_dept_copia.iloc[index,4] / 15
EE_dept_copia.iloc[index,5] = EE_dept_copia.iloc[index,5] / 15 
EE_dept_copia.iloc[index,6] = EE_dept_copia.iloc[index,6] / 15
EE_dept_copia.iloc[index,7] = EE_dept_copia.iloc[index,7] / 15
EE_dept_copia.iloc[index,8] = EE_dept_copia.iloc[index,8] / 15


index2 = cantidad[cantidad['Departamento'] == 'CIUDAD DE BUENOS AIRES'].index

cantidad_copia.iloc[index2,2] = cantidad_copia.iloc[index2,2] / 15 
cantidad_copia.iloc[index2,3] = cantidad_copia.iloc[index2,3] / 15 
cantidad_copia.iloc[index2,4] = cantidad_copia.iloc[index2,4] / 15 


# ----------------------------------------------------------------------------------------------------------------------------------- #
'''
Gráficos.
'''

######### 1 ##########

# Cant_BP por Provincia.
consulta = duckdb.query(""" 
                        SELECT Provincia, sum(Cant_BP) as Cant_BP
                        FROM cantidad 
                        GROUP BY Provincia
                        ORDER BY Cant_BP
                        """) 
cantidad_prov = consulta.df()   # Sumo la cantidad de BP por provincia.

# Cant_EE por Provincia.
consulta2 = duckdb.query(""" 
                         SELECT Provincia, sum(Cant_EE) as Cant_EE
                         FROM cantidad 
                         GROUP BY Provincia
                         ORDER BY Cant_EE
                         """) 
cantidad_prov2 = consulta2.df()   # Sumo la cantidad de EE por provincia.

# Grafico con esas tablas.
fig, ax = plt.subplots(figsize=(12,6))
plt.rcParams['font.family'] = 'sans-serif'

cantidad_prov.plot.bar(x = 'Provincia',
                       y = 'Cant_BP',
                       ax = ax,
                       color = 'b',
                       legend = False)

# Ajustes.
ax.set_title('Cantidad Total de BP por Provincia')
ax.set_xlabel('Provincias', fontsize='medium')
ax.set_ylabel('BP', fontsize='medium')
ax.set_ylim(0, 550)

# Coloco los valores arriba de cada barra.
ax.bar_label(ax.containers[0], fontsize=6, padding=2)

# Roto etiquetas en 'x' para que los nombres de las provincias queden claros.
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(graficos_path / "Grafico_A.png", dpi=300, bbox_inches='tight') 
plt.show()


## Mismo grafico que el anterior pero con EE.
fig, ax = plt.subplots(figsize=(12,6))
plt.rcParams['font.family'] = 'sans-serif'

cantidad_prov2.plot.bar(x = 'Provincia',
                        y = 'Cant_EE',
                        ax = ax,
                        color = 'b',
                        legend = False)

# Ajustes.
ax.set_title('Cantidad total de EE por Provincia')
ax.set_xlabel('Provincias', fontsize='medium')
ax.set_ylabel('EE', fontsize='medium')
ax.set_ylim(0, 17000)

# Coloco los valores arriba de cada barra.
ax.bar_label(ax.containers[0], fontsize=6, padding=2)

# Roto etiquetas en 'x'.
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig(graficos_path / "Grafico_B.png", dpi=300, bbox_inches='tight')
plt.show()


## Grafico extra -> Junto ambos gráficos anteriores para poder realizar una comparación mas eficiente.

merge = duckdb.query(""" 
                     SELECT cantidad_prov.Provincia, cantidad_prov.Cant_BP, cantidad_prov2.Cant_EE
                     FROM cantidad_prov
                     INNER JOIN cantidad_prov2
                     ON cantidad_prov.Provincia = cantidad_prov2.Provincia 
                     ORDER BY cantidad_prov2.Cant_EE
                     """) 
cantidad_merge = merge.df()   # Mergeo las 2 tablas que usé para los gráficos anteriores.

# Grafico.
fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

# Aplico dos ejes Y, que tome cada uno un rango diferente (uno para BP y otro para EE).
cantidad_merge.set_index('Provincia')[['Cant_EE', 'Cant_BP']].plot(kind='bar',
                                                                   ax=ax,
                                                                   secondary_y='Cant_BP',   # 2do eje.
                                                                   color=['red', 'blue'])

# Ajustes.
ax.set_title('EE y BP por Provincia')
ax.set_xlabel('Provincias', fontsize='medium')
ax.set_ylabel('cantidad EE', fontsize='medium')
ax.right_ax.set_ylabel('cantidad BP', fontsize='medium')
ax.set_ylim(0, 17500)

# Rotar etiquetas en 'x'.
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout() 
plt.savefig(graficos_path / "Grafico_C.png", dpi=300, bbox_inches='tight')
plt.show()


### Extras -> Graficos iguales a los 2 primeros pero normalizados.

nuevo = duckdb.query(""" 
                     SELECT Provincia, sum(Cant_BP) as Cant_BP, sum(Cant_EE) as Cant_EE, sum(Poblacion) as Poblacion
                     FROM cantidad
                     GROUP BY Provincia 
                     """) 
cantidad_nuevo = nuevo.df()   # Sumo la cantidad de BP, EE y población por provincia, para luego normalizar.

cantidad_nuevo["Cant_EE"] = cantidad_nuevo["Cant_EE"]/cantidad_nuevo["Poblacion"] *1000   # Normalizo (cantidad de BP y EE cada 1000 habitantes).
cantidad_nuevo["Cant_BP"] = cantidad_nuevo["Cant_BP"]/cantidad_nuevo["Poblacion"] *1000
cantidad_nuevo = cantidad_nuevo.sort_values(by='Cant_EE')

## Grafico con EE.
fig, ax = plt.subplots(figsize=(12,6))
plt.rcParams['font.family'] = 'sans-serif'

cantidad_nuevo.plot.bar(x = 'Provincia',
                        y = 'Cant_EE',
                        ax = ax,
                        color = 'b',
                        legend = False)

# Ajustes.
ax.set_title('Cantidad total de EE por Provincia (Normalizado)')
ax.set_xlabel('Provincias', fontsize='medium')
ax.set_ylabel('EE', fontsize='medium')
ax.set_ylim(0, 3)

# Coloco los valores arriba de cada barra.
ax.bar_label(ax.containers[0], fmt = '%.3f', fontsize = 6, padding = 2)   # El fmt='%.3f' sirve para tomar solo 3 decimales. 

# Roto etiquetas en 'x'.
plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
plt.tight_layout()
plt.savefig(graficos_path / "Grafico_D.png", dpi=300, bbox_inches='tight')
plt.show()

## Ahora grafico pero con BP.
cantidad_nuevo = cantidad_nuevo.sort_values(by = 'Cant_BP')

fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

cantidad_nuevo.plot.bar(x = 'Provincia',
                        y = 'Cant_BP',
                        ax = ax,
                        color = 'b',
                        legend = False)

# Ajustes.
ax.set_title('Cantidad total de BP por Provincia (Normalizado)')
ax.set_xlabel('Provincias', fontsize = 'medium')
ax.set_ylabel('BP', fontsize = 'medium')
ax.set_ylim(0, 0.35)

# Coloco los valores arriba de cada barra.
ax.bar_label(ax.containers[0], fmt = '%.3f', fontsize = 6, padding = 2)

# Roto etiquetas en 'x'.
plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
plt.tight_layout()
plt.savefig(graficos_path / "Grafico_E.png", dpi=300, bbox_inches='tight')
plt.show()


############# 2 ############### 

# Grafico cada nivel educativo en graficos separados para que se puedan comprender mejor (sacar mejores conclusiones).

## JARDINES.
fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

EE_dept_copia.plot.scatter(x = 'poblacion_total',
                     y = 'Jardines',
                     ax = ax,
                     color = 'b',
                     legend = False)

ax.set_xscale('log')   # Aplico escala logarítmica para mayor separación entre los puntos -> Gráfico más comprensible.

# Ajustes.
ax.set_title('Cantidad de Jardines vs Población')
ax.set_xlabel('Poblacion (escala log)', fontsize = 'medium')
ax.set_ylabel('Jardines', fontsize = 'medium')
ax.set_ylim(0, 400)
plt.savefig(graficos_path / "Grafico_F.png", dpi=300, bbox_inches='tight')
plt.show() 

## PRIMARIAS.

fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

EE_dept_copia.plot.scatter(x = 'poblacion_total',
                     y = 'primarias',
                     ax = ax,
                     color = 'b',
                     legend = False)
ax.set_xscale('log')

# Ajustes.
ax.set_title('Cantidad de Primarias vs Población')
ax.set_xlabel('Poblacion (escala log)', fontsize = 'medium')
ax.set_ylabel('Primarias', fontsize = 'medium')
ax.set_ylim(0, 450)
plt.savefig(graficos_path / "Grafico_G.png", dpi=300, bbox_inches='tight')
plt.show() 


## SECUNDARIOS.

fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

EE_dept_copia.plot.scatter(x = 'poblacion_total',
                     y = 'secundarios',
                     ax = ax,
                     color = 'b',
                     legend = False)
ax.set_xscale('log')

# Ajustes.
ax.set_title('Cantidad de Secundarias vs Población')
ax.set_xlabel('Poblacion (escala log)', fontsize = 'medium')
ax.set_ylabel('Secundarios', fontsize = 'medium')
ax.set_ylim(0, 500) 
plt.savefig(graficos_path / "Grafico_H.png", dpi=300, bbox_inches='tight')
plt.show()


## Bibliotecas por Poblacion.
# Grafico la evolucion de la cantidad de BP a medida que aumenta la población (cada punto representa un departamento).

fig, ax = plt.subplots(figsize = (12,6))
plt.rcParams['font.family'] = 'sans-serif'

cantidad_copia.plot.scatter(x = 'Poblacion',
                      y = 'Cant_BP',
                      ax = ax,
                      color = 'b',
                      legend = False)
ax.set_xscale('log')
ax.set_yscale('log')

# Ajustes.
ax.set_title('Cantidad de Bibliotecas vs Población')
ax.set_xlabel('Poblacion (escala log)', fontsize = 'medium')
ax.set_ylabel('Bibliotecas Públicas', fontsize = 'medium')
ax.set_ylim(0, 60)
plt.savefig(graficos_path / "Grafico_I.png", dpi=300, bbox_inches='tight')
plt.show()


############## 3 ###############

medianas = cantidad.groupby('Provincia')['Cant_EE'].median().sort_values()   # Calcula la mediana de cada provincia (agrupadas) con respecto a la cantidad de EE por departamento.
orden_provincias = medianas.index.tolist()   # Ordeno la tabla segun la mediana de cada provincia.

fig, ax = plt.subplots(figsize = (10, 8))

plt.figure(figsize = (10, 8))
sns.boxplot(data = cantidad,
            x = 'Cant_EE',
            y = 'Provincia',
            order = orden_provincias,
            palette = 'Set2',
            ax = ax)

# Títulos y ejes.
ax.set_title('Cantidad de EE por Departamento segun Provincia')
ax.set_xlabel('Cantidad de EE', fontsize = 'medium')
ax.set_ylabel('Provincias', fontsize = 'medium')
ax.set_xlim(0, 1800) 

plt.tight_layout()
plt.savefig(graficos_path / "Grafico_J.png", dpi=300, bbox_inches='tight')
plt.show()


## Mismo grafico pero con BP.

medianas2 = cantidad.groupby('Provincia')['Cant_BP'].median().sort_values()   # Calcula la mediana de cada provincia (agrupadas) con respecto a la cantidad de BP por departamento.
orden_provincias2 = medianas2.index.tolist()

fig, ax = plt.subplots(figsize = (10, 8))

plt.figure(figsize = (10, 8))
sns.boxplot(data = cantidad,
            x = 'Cant_BP',
            y = 'Provincia',
            order = orden_provincias2,
            palette = 'Set2',
            ax = ax)

# Títulos y ejes.
ax.set_title('Cantidad de BP por Departamento segun Provincia')
ax.set_xlabel('Cantidad de BP', fontsize = 'medium')
ax.set_ylabel('Provincias', fontsize = 'medium')
ax.set_xlim(0, 60) 

plt.tight_layout()
plt.savefig(graficos_path / "Grafico_K.png", dpi=300, bbox_inches='tight')
plt.show()


############## 4 ###############

cantidad_copia['BP_per_1000'] = cantidad_copia['Cant_BP'] / cantidad_copia['Poblacion'] * 1000   # Calculo la cantidad de BP y EE cada 1000 habitantes, por departamento (para normalizar).
cantidad_copia['EE_per_1000'] = cantidad_copia['Cant_EE'] / cantidad_copia['Poblacion'] * 1000

plt.figure(figsize = (8, 6))
ax = sns.scatterplot(data = cantidad_copia,
                     x = 'BP_per_1000',
                     y = 'EE_per_1000',
                     hue = 'Provincia',   # Color de cada punto según a la provincia a la que pretenece ese departamento.
                     palette = 'tab10',
                     s = 60,   # Tamaño de los puntos.
                     alpha = 0.8,   # Transparencia de cada punto (para poder hacer visible la superposición de valores).
                     legend = 'brief')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Cantidad de EE (escala log)', fontsize='medium')
ax.set_xlabel('Cantidad de BP (escala log)', fontsize='medium')
ax.legend(title = 'Provincia', bbox_to_anchor = (1, 1), loc = 'upper left')   # Agrego la leyenda fuera del gráfico con las provincias y sus respectivos colores.
plt.title('Cantidad de BP y EE cada 1000 habitantes por departamento', fontsize = 13)
plt.tight_layout()
plt.savefig(graficos_path / "Grafico_L.png", dpi=300, bbox_inches='tight')
plt.show()


############# GRÁFICOS EXTRAS ###############

#### E-1.

# Grafico un HEATMAP que nos pueda ayudar a relacionar EE con BP.

extra_2 = duckdb.query(""" 
                       SELECT
                           Provincia,
                           sum(Cant_BP) as Cant_BP,
                           sum(Cant_EE) as Cant_EE,
                           sum(Poblacion) as poblacionTot
                       FROM cantidad
                       GROUP BY Provincia
                       ORDER BY Cant_EE
                       """) 
extra2 = extra_2.df()   # Tomo la cantidad de BP y EE por provincia, y su poblacion total (para despues normalizar).

extra2["Cant_BP"] = extra2["Cant_BP"]/extra2["poblacionTot"] *1000   # Normalizo la tabla.
extra2["Cant_EE"] = extra2["Cant_EE"]/extra2["poblacionTot"] *1000
extra2 = extra2.drop(columns="poblacionTot")

# Me aseguro de tener solo columnas numericas (de cantidad de BP y EE) y a la provincia como índice (para realizar el heatmap).
extra2_clust = extra2.set_index("Provincia")[["Cant_BP", "Cant_EE"]] 

a = sns.clustermap(extra2_clust,
                   col_cluster = False,   # Para que no se agrupen las columnas, si no las filas (provincias).
                   method = "single",   # Impone un orden a las provincias (relaciona provincias para formar cadenas).      
                   cmap = "Blues",
                   standard_scale = 1   # Para comparar a las provincias entre sí.
                   )

a.fig.suptitle('Relación entre provincias\n(sobre la cantidad de EE y BP)', y = 0.90, fontsize = 16)
plt.savefig(graficos_path / "Grafico_M.png", dpi=300, bbox_inches='tight')
plt.show()


# %%
'''
Etapa 4 - Cierre y saludos.
'''

print("Fin del Script.")
print("Saludos Cordiales.")

