import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('C:/Users/USER/Documents/python/github/data_processing/data')
df = pd.read_csv('datatumores.csv', sep=',')
#print(df.head())

df1=df.drop(["Unnamed: 32","id"],axis=1)
print(df1.describe())

from sklearn.model_selection import train_test_split
x,y= df1.iloc[:,1:31], df1.iloc[:,0:1]

x_train,x_test,y_train,y_test=\
        train_test_split(x,#valor de predictores
                        y,#valor del target
                        test_size=0.3,#proporción de datos para datos de test, el complemento datos de entrenamiento
                        stratify=y,#estratificacion
                        random_state=0)#semilla



print(x_train.shape , x_test.shape)
corr_general = x_train.corr()
print(corr_general)

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(x_train)
kmo_model

print(kmo_all )

# Primero tipificamos las variables , es decir las escalamos.
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()# definimos la función a utilizar.

x_train_sc=sc.fit_transform(x_train)
x_test_sc=sc.fit_transform(x_test)

x_train_sc=pd.DataFrame(x_train_sc,columns=["radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
                                           "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
                                           "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se",
                                           "concave points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst",
                                           "perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst",
                                           "concave points_worst","symmetry_worst","fractal_dimension_worst"])
x_train_sc.info()

x_train_sc.head() #valores tipificados


from sklearn.decomposition import FactorAnalysis
#FactorAnalysis realiza una estimación de máxima verosimilitud de la llamada matriz de carga, la transformación de las variables latentes a las observadas,
#utilizando la maximización de expectativas (expectation-maximization - EM).

fac=FactorAnalysis() #definimos la función - inicializamos la clase
fact_df=fac.fit(x_train_sc) #ejecutamos el fa

fact_df.components_  #cargas factoriales

pd.DataFrame(fact_df.components_ ,columns=x_train_sc.columns).T

#extraemos los 10 factores

fac10=FactorAnalysis(n_components=10,# de componentes
                    random_state=0)
fact10=fac10.fit_transform(x_train_sc) #puntuaciones
fact10_df=pd.DataFrame(fact10,columns=["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"])

fact10_df.head()
#integramos la data

x_train_reset= x_train.reset_index(drop=True)
y_train_reset=y_train.reset_index(drop=True)

df_fact_completo=pd.concat([x_train_reset,fact10_df,y_train_reset],axis=1)

df_fact_completo.to_csv("df_fact_completo.csv")
