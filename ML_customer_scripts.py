import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Estandarizar la informacion, preproceso
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import OPTICS

# Reduccion de dimensionalidad: PCA
from sklearn.decomposition import PCA


import os
os.chdir("D:/ML_Spyder/Class4")

df = pd.read_csv("https://raw.githubusercontent.com/robintux/Datasets4StackOverFlowQuestions/master/CreditCard_Customer.csv")


#%%

df.info()
df.isna().sum().sort_values(ascending = False)
#MINIMUM_PAYMENTS                    313
# CREDIT_LIMIT                          1
# CUST_ID                               0

df.MINIMUM_PAYMENTS.isna().sort_values(ascending = False)
# 1       1072.340217
# 2        627.284787
# 3               NaN

df.CREDIT_LIMIT.head()
# 0    1000.0
# 1    7000.0
# 2    7500.0
# 3    7500.0
# 4    1200.0


sns.histplot(df, x= df.MINIMUM_PAYMENTS.index,
             y= df.MINIMUM_PAYMENTS)
# Se ve que no hay mucha discrepancia en los datos y una solucion valida es añadir con el
# promedio, los valores faltantes

df.MINIMUM_PAYMENTS = df.MINIMUM_PAYMENTS.fillna(df.MINIMUM_PAYMENTS.mean())
df.CREDIT_LIMIT = df.CREDIT_LIMIT.fillna(df.CREDIT_LIMIT.mean())

df_Num = df.drop(columns=["CUST_ID"])
df_Num.MINIMUM_PAYMENTS = df_Num.MINIMUM_PAYMENTS.fillna(df_Num.MINIMUM_PAYMENTS.mean())
df_Num.CREDIT_LIMIT = df_Num.CREDIT_LIMIT.fillna(df_Num.CREDIT_LIMIT.mean())

# Boxplot e histogramas
fig = plt.figure(num=1,
                 figsize=(12,12))
####################### MATRIZ DE BOXPLOTS ############################
ContadorGraf = 0
for col in df_Num.columns:
    print(col)
    fig.add_subplot(6, 3, ContadorGraf +1)
    plt.boxplot(x = df_Num[col])
    plt.xlabel("variable: " + col)
    ContadorGraf= ContadorGraf+1
    plt.tight_layout()
    # Cuando el tight_layout no funciona hay que recurrir al subplots_adjust
    # plt.subplots_adjust(bottom = 0.05,
    #                     top = 0.1,
    #                     right = 0.1)

plt.show()



####################### MATRIZ DE histogram<ss ############################

fig2 = plt.figure(num=2,
                 figsize=(12,12))

ContadorGraf = 0
for col in df_Num.columns:
    #print(col)
    fig2.add_subplot(6, 3, ContadorGraf +1)
    plt.hist(x = df_Num[col])
    plt.xlabel("variable: " + col)
    ContadorGraf= ContadorGraf+1
    plt.tight_layout()
    # Cuando el tight_layout no funciona hay que recurrir al subplots_adjust
    # plt.subplots_adjust(bottom = 0.05,
    #                     top = 0.1,
    #                     right = 0.1)

plt.show()


# %% Boxplot e histogramas juntos : Uno al lado del otro
# Generaremos una matriz de graficos para cada columna
# 1 fila x 2 columna
for col in df_Num:
    f = plt.figure(num = 1)
    # Primer eje para el boxplot
    ax1 = f.add_subplot(1,2,1)
    plt.boxplot(df_Num[col])
    # segundo eje para el histograma 
    ax2 = f.add_subplot(1,2,2)
    plt.hist(df_Num[col])
    # Movemos los yticks (ytickslabels) al margen derecho
    ax2.yaxis.tick_right()
    # 
    # Guardamos la imagen en disco duro 
    figName = "Boxplot_histograma" + col + ".png"
    plt.savefig(figName , dpi = 300)
    # del f
    plt.clf()
    
    
    
    
# Necesitamos colocar los yticks del histograma al lado derecho
type(ax2)
dir(ax2)
# Veamos la documentacion de yaxis 
help(ax2.yaxis)
# Listemos los metodos del objeto ax2.yaxis
dir(ax2.yaxis)
# Veamos la documenacion de tick_right
help(ax2.yaxis.tick_right)

# %% 


plt.figure(figsize = (14,14))
sns.heatmap(df_Num.corr(),
            annot = True, # para que agregue numeros,
            vmin=-1,
            vmax =1)# Es una correlacion que va de -1 a 1
plt.tight_layout()
plt.show()

#COnclusiones
# Hay mucha diferencia en las escalas
# Hay muchos outliers (valores atiícos)
# Entonces debemos escalar los datos: al (0,1)
# Usemos MinMaxScaler
# help(MinMaxScaler)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_Num)

type(X_scaled)
# Es tipo array entonces lo pasamos a dataframe, y como las columnas son nuevas 
# copiamos con las de df_Num

# Construyamos un dataframe 
df_sc = pd.DataFrame(X_scaled)
# 
# Asignemos a este nuevo dataframa los nombres de las columnas 
df_sc.columns = df_Num.columns
df_sc.isna().sum()
# %%Kmeans
help(KMeans)

# Number of times the k-means algorithm is run with different centroid seeds.
#  The final results is the best output of n_init consecutive runs in terms
#  of inertia. Several runs are recommended for sparse high-dimensional problems
# (see kmeans_sparse_high_dim).

sse = {}
for k in range(1,12):
    kmeans = KMeans(n_clusters=k, 
                    n_init = 20,# COmo 10 por defecto es poco, usaremos 20
                    # combinaciones
                    max_iter=1000).fit(df_sc)
    sse[k] = kmeans.inertia_

# una vez que tengo quiero graficar llaves del diccionario vs valores del dicc
plt.plot(sse.keys(), sse.values())
plt.ylabel("SSE")
plt.xlabel("Num_clusters")
plt.title("Metodo del codo")


# Concluyamos k=3 es el numero adecuado de clusters que esta el codo
# Visualicemos clusters para 3 dimensiones de las 17 columnas
# %% Modelo reducido

ModelPCA = PCA(n_components=2)
ModelPCA.fit(df_sc)
df_PCA = ModelPCA.transform(df_sc)
# %%
ModelKmeans= KMeans(n_clusters = 3,
                    max_iter = 1000,
                    n_init = 20)

ModelKmeans.fit(df_PCA)
# Visualicemos como quedo nuestro Kmeans 
plt.scatter(df_PCA[:, 0], df_PCA[:, 1],
            c = ModelKmeans.labels_)
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.show()

# Usemos estas etiquetas para nuestro dataset
df_sc["cluster"] = ModelKmeans.labels_
# Esto es basado en los centroides, ahora usando densidades


# %% DBSCAN
# Identifica regiones de alta densidad que estan rodeadas de regiones de
# baja densidad
# Este hecho nos dice que a diferencia del algoritmo KMeans, este algoritmo 
# no necesita que le especifique el numero de clusters a construir, por
# otro lado cuando hay outliers este modelo suele considerarlos como bajas 
# densidades.

# Usemos el PCA con dos componentes: df_PCA
# Para este dataframe construyamos el DBSCAN

# ModelDB = DBSCAN(eps = 0.1, min_samples=20).fit(df_PCA)
# Sigue agrupando solo 1
ModelDB = DBSCAN(eps = 0.1, min_samples=20).fit(df_PCA)

etiquetas = ModelDB.labels_
np.unique(etiquetas)

# Observemos el etiquetado 
plt.scatter(df_PCA[:, 0], df_PCA[:, 1],
            c = etiquetas)
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.show()

# %% Busquemos un modelo de tipo DBSCAN que no obtenga una sola etiqueta
# help(DBSCAN)


Valores_eps = np.linspace(0.01, 0.99, num = 30)
# Dicc para almacenar los valores epese y sus etiquetas
DiccEPS = {}
for epese in Valores_eps:
    ModelDB = DBSCAN(eps = epese).fit(df_PCA)
    etiquetas =  ModelDB.labels_
    if len(np.unique(etiquetas)) > 1 and len(np.unique(etiquetas))<3:
        DiccEPS[epese] = etiquetas
DiccEPS

# Mostremos los posibles clusters

for k in DiccEPS.keys():
    print(np.unique(DiccEPS[k]))

# Para los eps siguientes 
   # {0.07758620689655171 
   #  0.11137931034482758
   #  0.14517241379310344
# Hay estas 2 unicas posibles etiquetas respectivamente
    # [-1  0]
    # [-1  0]
    # [-1  0]
    
plt.scatter(df_PCA[:, 0], df_PCA[:, 1],
            c = DiccEPS[list(DiccEPS.keys())[0]])
plt.xlabel("PCA_1")
plt.ylabel("PCA_2")
plt.show()

# Para este dataset no sirve hacer un algoritmo de densidad

# %% 
# Agrupemos las observaciones df_sc en funcion de la columna "cluster"

df_sc_CLUSTER = df_sc.groupby("cluster")
df_sc_CLUSTER.describe()
# Puedo ir separando la variable cluster como variable cualitativa

# Observemos que ocurre con cada variable en cada cluster
# Para eso creemos un dataset nuevo a partir de Num agregando los labels del 
# modelo KMeans
df_Num_cl = df_Num.copy()
df_Num_cl["cluster"] = ModelKmeans.labels_
for col in df_Num_cl.columns:
    plt.scatter(np.arange(0,df_Num_cl.shape[0]),
                df_Num_cl[col],
             c = df_Num_cl["cluster"])
    figName = "Variable" + col + ".png"
    plt.savefig(figName)

# %% Probemos con OPTICS

ModelOptics = OPTICS().fit(df_PCA)
etiquetOptics = Model








