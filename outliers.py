import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('/Users/angelseda/Documents/angel/data_processing/data')
df = pd.read_csv('Marketing.csv', sep=';')
print(df.shape)
print(df.info)

##Identificacion de variables numericas
plt.figure(figsize=(4,4))
df.boxplot()
#plt.show()

#Revisar variable monto
plt.figure(figsize=(4,4))
df.boxplot(['Monto'])
#plt.show()

q1= df['Monto'].quantile(0.25)
q3= df['Monto'].quantile(0.75)
q2= df['Monto'].median()
valor_min=df['Monto'].min()
valor_max=df['Monto'].max()
iqr = q3-q1
print(q1)
print(q2)
print(q3)
print(valor_min)
print(valor_max)
print('Rango intercuartil:',iqr)

binf = (q1 -1.5*iqr)
print('Valor bigote inferior: ', binf)

bsup =(q3 + 1.5*iqr)
print('Valor bigote superior: ', bsup)

ubic_out = (df['Monto'] <binf) | (df['Monto']>bsup)
print('Ubicacion de outliers', ubic_out)

out = df[ubic_out]
ordenados = out.sort_values('Monto', ascending=False)
print(ordenados)
print(ordenados.shape[0])

ubic_sinout = (df['Monto']> binf) & (df['Monto']<bsup)
sin_out = df[ubic_sinout]
ordenados_sinout = sin_out.sort_values('Monto', ascending=False)
print(ordenados_sinout.head(15))
print(ordenados_sinout.shape)

#Revisar monto
plt.figure(figsize=(4,4))
ordenados_sinout.boxplot(['Monto'])
#plt.show()

percent = df.describe(percentiles=list(np.arange(0.1,0.9,0.1))+list(np.arange(0.9,1.0,0.01)))
print(percent)

#la función percentil es sensible a NAN , mientras que la función quantile no.
df2=df.dropna(subset=['Monto'])
p95 = np.percentile(df2['Monto'], 95)
print(p95)

q95 = df['Monto'].quantile(0.95)
print(q95)

df_final =df.loc[df['Monto']<=q95]
print(df_final.head())

df_out =df.loc[df['Monto']>q95]
print(df_out.head())

print(df_out.shape)

#observamos que se ha omitido todos los outliers...df de 961 a 913 registros
plt.figure(figsize=(4,4))
df_final.boxplot(['Monto'])
#plt.show()

###Reemplazo de outliers "capear outliers"
df3 = df.copy()
df3_max =df3['Monto'].max()
print(df3_max)

df_q95 = df3[df3.Monto >= q95]
df3.loc[df3.Monto>=q95,"Monto"]=q95
df3_max = df3['Monto'].max()
print(df3_max)

plt.figure(figsize=(4,4))
df3.boxplot(['Monto'])
plt.show()