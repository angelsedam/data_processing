import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

os.chdir('C:/Users/USER/Documents/python/github/data_processing/data')
df= pd.read_csv('data.csv',sep=',')
print(df.info())

##Discretizacion por intervalos de igual amplitud
from sklearn.preprocessing import KBinsDiscretizer
import math

n=len(df)
k = 1 + math.log2(n)
k = round(k,0)

xmin=df.edad.min()
xmax=df.edad.max()
c=(xmax-xmin)/k
print(c)

##uniforme
C =KBinsDiscretizer(n_bins=15,encode='ordinal',strategy='uniform')
edad_disc_sturges =C.fit_transform(df[['edad']])
xs = pd.DataFrame(edad_disc_sturges)
xs.columns= ['edad_disc_amplitud']
xs.groupby('edad_disc_amplitud').size().plot(kind='bar',rot=0)
##plt.show()

edad_disc_sturges=pd.DataFrame(edad_disc_sturges,columns=["Edad_igual_amplitud"])
edad_disc_sturges.head()
df2=pd.concat([df,edad_disc_sturges],axis=1)
df2.head()


##Discretizacion cuantil
from sklearn.preprocessing import KBinsDiscretizer
##Quantil
C_q =KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
edad_disc_cuant=C_q.fit_transform(df[['edad']])
xq=pd.DataFrame(edad_disc_cuant)
xq.columns=['edad_disc_q']
xq.groupby('edad_disc_q').size().plot(kind='bar',rot=0)
##plt.show()

#indice y valor del cuantil con base a la edad
edad_disc_cuant =pd.DataFrame(edad_disc_cuant, columns=["edad_disc_cuant"])
edad_disc_cuant.head()

df3=pd.concat([df2,edad_disc_cuant],axis=1)
print(df3)

###Discretizacion por Kmeans
from sklearn.preprocessing import KBinsDiscretizer
##kmeans
C_Kmeans = KBinsDiscretizer(n_bins=4,encode='ordinal',strategy='kmeans')

edad_disc_km =C_Kmeans.fit_transform(df['edad'])
xkmeans=pd.DataFrame(edad_disc_km)
xkmeans.columns=['edad_disc_km']
xkmeans.groupby('edad_disc_km').size().plot(kind='bar',rot=0)
#plt.show()

edad_disc_km =pd.DataFrame(edad_disc_km, columns=["edad_disc_km"])
edad_disc_km.head()

df4=pd.concat([df3,edad_disc_km],axis=1)
print(df4)

###Discretizacion por entropia
from MLDP import MDLP_Discretizer

x= df.iloc[:, df.columns != 'clase'].values        #Selecciona menos la variable clase
y = df.iloc[:, -1].values

edad=[4] #Especificar la posición de la variable
numeric_features=np.array(edad) #Convertirlos a un array de numpy

#Defino la funcion de discretizacion con las posiciones que debe convertir
discretizer = MDLP_Discretizer(features=numeric_features)

#aplicar la discretización
discretizer.fit(x, y)

#Transforma la discretizacion
x_train_discretized = discretizer.transform(x)

Ed_disc_entrop=x_train_discretized[:,4]#Selecciono los valores de la columna "Age" de la data de entrenamiento
df_age = pd.DataFrame(Ed_disc_entrop,columns=['edad_discretizada'])#Genero un dataframe a partir del los valores de "Age"

df_age.edad_discretizada.value_counts() #Agrupa los diferentes grupos

#Defino la dimensionalidad del grafico
f, ax = plt.subplots(1, 1, figsize=(4, 4))
#Coloco la data la misma que imprimi arriba
df_age.edad_discretizada.value_counts().plot(kind='bar', title="Transaction type", ax=ax, figsize=(8,8))
plt.show()#Muestra el grafico

#ver cómo se discretizó la variable en la posicion 37 "age"
print ('Interval cut-points:%s' % str(discretizer._cuts[4]))#Los puntos de corte
print ('Bin descriptions: %s' % str(discretizer._bin_descriptions[4])) #Muestra el encode que se asigno a cada uno

df5=pd.concat([df4,df_age],axis=1)
df5.head()
