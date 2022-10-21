import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/USER/Documents/python/github/data_processing/data')
df = pd.read_csv('datatumores.csv', sep=',')
#print(df.head())

df1 = df.drop(['Unnamed: 32','id'], axis=1)
print(df1.describe())

##primer parametro es filas segundo columnas
target = df.iloc[:,0:1]
df_mean = df.iloc[:,0:11]
df_se=pd.concat([target, df1.iloc[:,11:21]], axis=1)
df_worst =pd.concat([target, df1.iloc[:,21:31]], axis=1)

##visualizacion con seaborn
##df_mean
sns.set_theme(style="ticks")
sns.pairplot(df_mean, hue="diagnosis")

##df_se
#sns.set_theme(style="ticks")
#sns.pairplot(df_se, hue="diagnosis")

##df_worst
#sns.set_theme(style="ticks")
#sns.pairplot(df_worst, hue="diagnosis")

from sklearn.model_selection import train_test_split
x,y = df1.iloc[:,1:31], df1.iloc[:,0:1]

x_train, x_test, y_train, y_test =train_test_split(x,y,test_size =0.3, stratify=y,random_state=0)
    #x,  # valor de predictores
    #y,  # valor del target
    #test_size=0.3,  # proporción de datos para datos de test, el complemento datos de entrenamiento
    #stratify=y,  # estratificacion
    #random_state=0)  # semilla

print(x_train.shape, x_test.shape)
corr_general = x_train.corr()
print(corr_general)

#Matriz de correlacion de promedios
corr_mean =x_train.iloc[:,0:11].corr()
cmap =sns.diverging_palette(220,20,n=7, as_cmap=True)
sns.heatmap(corr_mean, annot=True, linewidths=1, cmap=cmap)
##plt.show()

##Matriz de correlacion de se
corr_se = x_train.iloc[:,10:20].corr()
cmap=sns.diverging_palette(220,20,n=7,as_cmap=True)
sns.heatmap(corr_se, annot = True,linewidths=1,cmap=cmap)
##plt.show()

##Matriz de correlacion de worst
corr_worst = x_train.iloc[:,20:30].corr()
cmap=sns.diverging_palette(220,20,n=7,as_cmap=True)
sns.heatmap(corr_worst,linewidths=1,annot = True,cmap=cmap)
##plt.show()

import scipy as st
import math as math

n=x_train.shape[0]
p=x_train.shape[1]
chi2=round(-(n-1-(2*p+5)/6)*math.log(np.linalg.det(corr_general)),2)
print(chi2)

gl=p*(p-1)/2
print(gl)

p_value=st.stats.chi2.pdf(chi2,gl)
print(p_value)

# Primero tipificamos las variables , es decir las escalamos.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)
print(x_train.info())

x_train_sc=pd.DataFrame(x_train_sc,columns=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                                           "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                                           "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                                           "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                                           "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                                           "concave points_worst","symmetry_worst","fractal_dimension_worst"])

from sklearn.decomposition import PCA
pca = PCA(n_components=30)
pca=pca.fit(x_train_sc)

varianza_explicada= pca.explained_variance_
print("Varianza explicada por cada componente principal:" , varianza_explicada)

prop_var_explic= pca.explained_variance_ratio_
print("Proporción de varianza explicada por cada componente principal:" , prop_var_explic)

#varianza acumulada
np.cumsum(pca.explained_variance_ratio_)

# Graficamos la explicación de la varianza por cada CP.
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], pca.explained_variance_ratio_, '-o')
plt.ylabel('Proporción de Varianza Explicada')
plt.xlabel('Componente Principal')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.show()

# Graficamos la explicación de la varianza acumulada por cada CP.
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30], np.cumsum(pca.explained_variance_ratio_), '-s')
plt.ylabel('Proporción Acumulada de Varianza Explicada')
plt.xlabel('Componente Principal')
plt.xlim(0.75,4.25)
plt.ylim(0,1.05)
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])
plt.show()

pca1=PCA(n_components=6) #indicamos el número de componentes= al # de variables originales.
pca1=pca1.fit(x_train_sc) #ejecutamos el pca

#revisemos las cargas factoriales.
#explicación de cada variable original a cada componente
pca_cargas_fact= pca1.components_

print(pca_cargas_fact)

pd.DataFrame(pca_cargas_fact,columns=x_train_sc.columns,index=["PC1","PC2","PC3","PC4","PC5","PC6"]).head(6).T

#puntuaciones factoriales, valor que toma cada fila en los componentes principales.
pca_scores=pca1.transform(x_train_sc)
pca_scores

df_pca= pd.DataFrame(pca_scores,columns=["PC1","PC2","PC3","PC4","PC5","PC6"])

#reseteo de index
x_train_reset= x_train.reset_index(drop=True)
y_train_reset=y_train.reset_index(drop=True)


df_pca_completo= pd.concat([x_train_reset,df_pca,y_train_reset],axis=1)
df_pca_completo.to_csv("df_pca_completo.csv")