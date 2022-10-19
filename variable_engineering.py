import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.model_selection import train_test_split

os.chdir('C:/Users/USER/Documents/python/github/data_processing/data')
df = pd.read_csv('dataBasePrestDigital.csv',sep=";")
print(df.head(10))

print(df.ventaPrestDig.value_counts())
print(df.ventaPrestDig.value_counts()/df.shape[0]*100)
print(df.isnull().sum())
print(df.info())

# separamos el dataset en data numérica y data categórica.

ColCateg = ['estadoCliente','rngEdad','genero','rngSueldo','procedencia','operDigital','tenTarjeta']
ColNum = ['trxDigitalUm','promTrxDig3Um','recCamp','frecCamp','promConsBanco3Um','promSaldoBanco3Um','promSaldoTc3Um','promSaldoPrest3Um','sowTcUm','sowPrestUm']

df_categ=df[ColCateg]
df_num=df[ColNum]

#recodificamos con labelencoder
le=defaultdict(LabelEncoder) #definimos la función

fit=df_categ.apply(lambda x: le[x.name].fit_transform(x))# ajusto, entreno o entiendo lo que necesito
df_categ_cod=df_categ.apply(lambda x: le[x.name].transform(x))#aplico, transformar o predecir

#recodificamos con OneHotEncoder , función está implementada en pandas.
#obtendremos las variables dummys, dicotómicas.
#generemos una copia del dataset categórico.
df_categ2=df_categ.copy()
df_categ2.head()
df_categ2_ohe=pd.get_dummies(df_categ2)
df_categ2_ohe.head() #de 7 variables, ahora se tiene 25 variables.

#para fines prácticos, usaremos la dataset del labelenconder.
df2= pd.concat([df_num,df_categ_cod,df.ventaPrestDig],axis=1)

# Generación de nuevas variables usando criterios experto o desiciones del negocio.**

df2['promConsSaldoBanco3Um'] = df2['promConsBanco3Um'] / (
            df2['promSaldoBanco3Um'] + 1)  # colocamos +1 si la variable denominador
# puede llegar tomar el valor de cero

df2["promSaldoCamp"] = df2["promSaldoBanco3Um"] / (df2["frecCamp"] + 1)

df2["promSaldoTc3mIngreso"] = df2["promSaldoTc3Um"] / (df2["rngSueldo"] + 1)

#Transformaciones no Lineales

df2["log_promTrxDig3Um"]= np.log1p(df2["promTrxDig3Um"]+1)

#log1p, por la distribución asimetrica positiva de la variable

# Usando polinomios - Polinomial Features
# Logica: 1,2,3,  1^2,1*2,1*3, 2^2, 2*3, 3^2

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2,interaction_only=False,include_bias= False) # Definimos el grado del polinomio
to_cross = ['promTrxDig3Um', 'promConsBanco3Um', 'promSaldoBanco3Um'] # Definimos las variables a usar

new_feats = poly.fit_transform(df2[to_cross].values)  # Aplicamos la transformacion polinomica.

#convertimos a un dataframe
new_feats =pd.DataFrame(new_feats)
new_feats.head()

# como las primeras tres variables son las variables con las que se cuenta en el dataset, no la consideraremos

new_feats = pd.DataFrame(new_feats.iloc[:,3:9].to_numpy(),columns=['promTrxDig3Um_2','promTrxDig3Um_promConsBanco3Um','promTrxDig3Um_promSaldoBanco3Um','promConsBanco3Um_2','promConsBanco3Um_promSaldoBanco3Um','promSaldoBanco3Um_2'])

#Ahora concatemos las nuevas variables polinómicas, con el df2

df3=pd.concat([df2,new_feats],axis=1)
df3.head()

# Anexos : WOE, función para poder calcular la tabla de resultados WOE.
import woe
from woe.eval import plot_ks
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
from pylab import rcParams

rcParams['figure.figsize'] = 14, 8
import warnings

warnings.filterwarnings('ignore')
max_bin = 20
force_bin = 3

# Creamos las Woes - IV
max_bin = 20
force_bin = 3


# Variables numéricas

def mono_bin(Y, X, n=max_bin):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return (d3)


# variables categóricas

def char_bin(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return (d3)


def data_vars(df1, target):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)

final_iv, IV = data_vars(df3,df3.ventaPrestDig) # aplicamos la función data_vars para encontrar el woe y el iv
IV.sort_values('IV',ascending=False) #ordenamos el valor de información

x= df3.drop("ventaPrestDig",axis=1)
y=df3.ventaPrestDig

# Seleccion por Random Forest
from sklearn.ensemble import RandomForestClassifier # Instancio el algoritmo
forest = RandomForestClassifier()                   # Configuro el algoritmo
forest.fit(x,y)                                     # Ajuste el algoritmo
importances = forest.feature_importances_           # Variables importantes

print(importances )

# veamos en un cuadro la importancia de cada variable
TablaImportancia = pd.concat([pd.DataFrame({'VariableName':list(x.columns)}),
                              pd.DataFrame({'Importancia':list(forest.feature_importances_)})], axis = 1)
ImportanciaVariables = TablaImportancia[['VariableName','Importancia']].sort_values('Importancia', ascending = False).reset_index(drop = True)
print(ImportanciaVariables)










