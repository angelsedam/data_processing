import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

os.chdir('/Users/angelseda/Documents/angel/data_processing/data')


###Identificacion


df = pd.read_csv('Marketing.csv', sep=';')
print(df.head())

#Matriz
plt.figure(figsize=(4,4))
msno.matrix(df)
#plt.show()

#Barras
plt.figure(figsize=(4,4))
msno.bar(df)
#plt.show()

#Agrupar variables entre si por su correlacion de nulidad
plt.figure(figsize=(4,4))
msno.dendrogram(df)
#plt.show()

#df.isnull()
print(df.isnull().any())

print(df.isnull().sum())

print(df.info())

print(df.Historial.notnull().value_counts())

print('Dimension Real', df.shape)

print('Dimension Real', df.dropna().shape)

print(df.isnull().sum()/df.shape[0]*100)

###Tratamiento

#Eliminacion
df_drop=df.copy()
print(df_drop.head())

#Elimina el registro donde haya algun nulo en cualquier columna
df_drop1 = df_drop.dropna()
print(df_drop1.info())

#eliminamos por registro, donde Historial presente valores nulos
#en sus filas....eliminación específica por variables.
df_drop2=df_drop.dropna(subset='Historial')
print(df_drop2.info())

df_drop3=df_drop.dropna(subset=['Historial','Monto'])
print((df_drop3.info()))

# si requerimos eliminar aquellas variables donde se tenga valores vacios,
#se deberá usar axis=1, por defecto toma axis=0
#Elimina las columnas que tengan registros vacios
df_drop4 = df_drop.dropna(axis=1)

# para eliminar solo las filas que contiene todas las columnas con NAN
df_drop5 = df_drop.dropna(how='all')
# es el valor que usa por defecto en la eliminación por fila.
df_drop6 = df_drop.dropna(how='any')

###Imputar
df_impt1 = df.copy()

# con fillna, indicamos por que falores queremos que rellene los vacios,
#en este caso por el valor cero
df_impt1['Monto1'] = df_impt1.Monto.fillna(0)
print(df_impt1)
print(df_impt1[['Monto','Monto1']].describe())

df_impt1['Monto2']=df_impt1.Monto.fillna(df_impt1.Monto.median())
print(df_impt1[['Monto','Monto1','Monto2']].describe())

df_impt1['Monto3']=df_impt1.Monto.fillna(df_impt1.Monto.mean())
print(df_impt1[['Monto','Monto1','Monto2','Monto3']].describe())

#metodo que rellena el valor faltante con su valor anterior.
# existe también el método bfill(backfill), que reemplaza el valor faltante por el valor posterior
df_impt1['Monto4']=df_impt1.Monto.fillna(method='pad')
df_impt1[['Monto','Monto1','Monto2','Monto3','Monto4']].describe()

#Grafica de monto
plt.figure(figsize=(4,4))
plt.title(df_impt1['Monto'].name)
plt.hist(df_impt1['Monto'],bins=60)
#plt.show()

#REvision de la distribucion de las diferentes imputaciones.
for i in ['Monto','Monto1','Monto2','Monto3','Monto4']:
    plt.title(i)
    plt.hist(df_impt1[i], bins=60)
    plt.rcParams['figure.figsize'] = (4,4)
    #plt.show()

#Estimador Scikit Learn
#Es una de las principales API implementadas por Scikit-learn. Proporciona una interfaz coherente
#para una amplia gama de aplicaciones de aprendizaje automático, por eso todos los algoritmos de
#aprendizaje automático en Scikit-Learn se implementan a través de la API de Estimator.
#El objeto que aprende de los datos (ajustando los datos) es un estimador. Se puede utilizar con
#cualquiera de los algoritmos como clasificación, regresión, agrupación o incluso con un
#transformador, que extrae características útiles de los datos sin procesar.

#1 - Datos de entrenamiento -> est.fit(x_train)
#2 - Datos de entrenamiento transformados -> est.transform(x_train)
#3 - Datos de prueba  a datos de prueba transformador -> est.transform(x_test)

#SimpleImputer proporciona estrategias básicas para imputar valores perdidos.
# Los valores perdidos pueden ser imputados con un valor constante proporcionado, o
# utilizando las estadísticas (mean, median o most_frequent) de cada columna en la que
# se encuentran los valores perdidos. SimpleImputer también permite diferentes codificaciones
# de valores faltantes.

#La libreria SimpleImputer para el parametro "strategy"
#Utiliza las siguientes estrategias de imputación:
#Si es "mean", reemplace los valores faltantes usando la media a lo largo de cada columna. Solo se puede usar con datos numéricos.
#Si es "median", reemplace los valores faltantes usando la mediana a lo largo de cada columna. Solo se puede usar con datos numéricos.
#Si es "most_frequent",reemplace los que faltan usando el valor más frecuente en cada columna. Puede usarse con cadenas o datos numérico. Si hay más de uno de esos valores, solo se devuelve el más pequeño.
#Si es "constant", reemplace los valores faltantes con fill_value. Puede usarse con cadenas o datos numéricos.

df_impt2= df.copy()
print(df_impt2.isnull().sum())

#Imputacion por moda
imp_moda=SimpleImputer(strategy ='most_frequent')
df_imp_moda=imp_moda.fit_transform(df_impt2)
columns=["Edad","Genero","Vivienda","Ecivil","Ubicacion","Salario","Hijos","Historial","Catalogos","Monto"]
#Regresa un array
df_imp_moda=pd.DataFrame(df_imp_moda,columns=columns)
print(df_imp_moda.head())

#imputacion por media
imp_media = SimpleImputer(strategy='mean')
df_imp_media = imp_media.fit_transform(df_impt2[['Monto']])
df_imp_media=np.round(df_imp_media, decimals=2)
df_imp_media = pd.DataFrame(df_imp_media,columns=['MontoImp'])
print(df_imp_media, df_imp_media.info())

df_montoimputado =pd.concat([df_impt2,df_imp_media], axis=1)
print(df_montoimputado.head())

df_hist_moda = imp_moda.fit_transform(df_impt2[['Historial']])
df_hist_moda=pd.DataFrame(df_hist_moda,columns=['HistorialImpt'])
df_hist_imputado = pd.concat([df_montoimputado,df_hist_moda], axis=1)
print(df_hist_imputado.head())

####Imputacion por modelos
sns.pairplot(df)
#plt.show()

###Regrecion Lineal
#Creamos una copia de df con el cual estaremos trabajando
df_reg = df.copy()

#Creamos un df eliminando los valores nulos tomando como base la columna monto
#la data de entrenamiento no deberá tener valores nulos en monto.
#Creacion de la data de entrenamiento
df_reg_nonull = df_reg.dropna(subset=['Monto'])

#data de entrenamiento con variables a usar en el modelo
x_train = df_reg_nonull[['Salario']]
y_train = df_reg_nonull[['Monto']]

#Localizamos los registros con valores nullos
isnull = pd.isna(df_reg.loc[:,'Monto'])
#Seleccionamos solo los registros con valores nulos
df_reg_null = df_reg.loc[isnull]

x_test = df_reg_null[['Salario']]
imp_regression = LinearRegression()
imp_regression.fit(x_train,y_train)

ypred = imp_regression.predict(x_test)
ypred = np.round(ypred)
print(ypred)

ypred=pd.DataFrame(ypred,columns=['Monto'])
df_reg_null = df_reg_null.drop(['Monto'], axis=1)
df_reg_null = df_reg_null.reset_index(drop=True)
ypred= ypred.reset_index(drop=True)

df_imput_monto = pd.concat([df_reg_null,ypred], axis=1)
print(df_imput_monto.head())

df_reg_nonull =df_reg_nonull.reset_index(drop=True)
df_imput_monto= df_imput_monto.reset_index(drop=True)

df_integrada = pd.concat([df_reg_nonull,df_imput_monto], axis= 0)
print(df_integrada.info())

##Modelo Abrbol
df_arb = df_integrada.copy()
df_arb_train = df_arb.dropna(subset=['Historial'])
x_train = df_arb_train[['Edad','Genero','Ecivil']]
y_train = df_arb_train[['Historial']]

is_null_hist= pd.isna(df_arb.loc[:,"Historial"])
print(is_null_hist)
print(df_arb.columns)

df_arb_null= df_arb.loc[is_null_hist]
x_test=df_arb_null[['Edad','Genero','Ecivil']]

le=defaultdict(LabelEncoder)

# ajusto, entreno o entiendo lo que necesito
fit=x_train.apply(lambda x: le[x.name].fit_transform(x))

#aplico, transformar o predecir
x_train2 = x_train.apply(lambda x: le[x.name].transform(x))


model_tree = tree.DecisionTreeClassifier(criterion='entropy', # criterio por defecto Gini ().
                                         min_samples_split=20,#por defecto es 2, número minimo de muestras necesarias para dividir un nodo
                                         min_samples_leaf=5,# por defecto es 1, número mínimo de muestras necesarias por hoja
                                         max_depth=4)# por defecto en ninguno, profundidad máxima del arbol
model_tree.fit(x_train2,y_train)
# Gini, tiende a aislar la clase más frecuente en su propia rama del árbol, mientras que la entropía tiende a producir
#árboles ligeramente más equilibrados.

x_test2 = x_test.apply(lambda x: le[x.name].transform(x))
y_pred = model_tree.predict(x_test2)
df_pred = pd.DataFrame(data=y_pred, columns=['Historial'])

df_arb_null = df_arb_null.drop(['Historial'], axis=1)
df_arb_null = df_arb_null.reset_index(drop=True)
df_pred =df_pred.reset_index(drop=True)
df_imput_arb = pd.concat([df_arb_null,df_pred], axis=1)

df_imput_arb= df_imput_arb.reset_index(drop=True)
df_arb_train = df_arb_train.reset_index(drop=True)
df_final_imp = pd.concat([df_imput_arb,df_arb_train], axis=0)
print(df_final_imp.info)

df_final_imp.to_csv('MarketingLimpio.csv')
