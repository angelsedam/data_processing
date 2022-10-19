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
df = pd.read_csv('HR-Employee-Attrition.csv', sep=',')
##print(df.head())
print(pd.value_counts(df['Attrition']))

#codificacion de variables utilizando label enconder
from sklearn.preprocessing import LabelEncoder
#definicion de funcion
le=defaultdict(LabelEncoder)

att=df[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','OverTime','Attrition']]
#print(att.head())
fit =att.apply(lambda x: le[x.name].fit_transform(x))
att2=att.apply(lambda x: le[x.name].transform(x))
#print(att2.head())

df1=df.copy()
df1=df1.drop(["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime","Attrition"], axis=1)

att2=att2.reset_index(drop=True)
df1=df1.reset_index(drop=True)
df_att=pd.concat([df1,att2], axis=1)

clas_att = pd.value_counts(df_att['Attrition'])
print(clas_att)

plt.figure(figsize=(4,4))
clas_att.plot(kind='bar', rot=0)
plt.title('Frecuencia de numero de empleado')
plt.xlabel('Estado de Attrition')
plt.ylabel('Numero de empleados')
#plt.show()

df_clean = df_att.drop(["Over18","EmployeeCount","EmployeeNumber"], axis=1)
df_clean.head()

x= df_clean.iloc[:,0:31].values
y= df_clean.iloc[:,31].values
print(x.shape,y.shape)

x_train, x_test, y_train, y_test =train_test_split(x, #valores de los predictores
                                                   y, #los valores del target
                                                   test_size=0.3, #proporción para datos de testeo
                                                   random_state=1, #semilla
                                                   stratify=y) #la variable de estratificación
#print(x_train.shape,y_train.shape)
#número de empleados por clase en nuestra muestra de entrenamiento
print("Attrition:",np.sum(y_train==1, axis=0)) , print("No Attrition:",np.sum(y_train==0, axis=0))

####UnderSampling

'''El submuestreo aleatorio funciona con la clase mayoritaria y consiste en reducir el número de observaciones de la clase mayoritaria para equilibrar el conjunto de datos.'''

from imblearn.under_sampling import NearMiss

unds = NearMiss(sampling_strategy=0.8,
                n_neighbors=5,
                version=3) # semilla

x_t_unds, y_t_unds= unds.fit_resample(x_train,y_train) #aplicamos undersapling
X_t_unds1=pd.DataFrame(x_t_unds,columns=["Age","DailyRate","DistanceFromHome","Education",
                                        "EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel",
                                        "JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
                                        "PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours",
                                        "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance",
                                        "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
                                        "BusinessTravel","Department","EducationField","Gender",
                                        "JobRole","MaritalStatus","OverTime"])

Y_t_unds1=pd.DataFrame(y_t_unds,columns=["Attrition"])
df_training=pd.concat([X_t_unds1,Y_t_unds1],axis=1)
df_training.head()


count_clases=pd.value_counts(df_training["Attrition"])
count_clases

#fig, ax=plt.subplots(1,1)
plt.figure(figsize=(4,4))
count_clases.plot(kind="bar",rot=0)
plt.title("Frecuencia de número de empleados - Undersampling")
plt.xlabel("Estado de Attrition")
plt.ylabel("Número de empleados")
plt.show()


###Oversampling
#número de empleados por clase en nuestra muestra de entrenamiento
print("Attrition:",np.sum(y_train==1, axis=0)) , print("No Attrition:",np.sum(y_train==0, axis=0))

os =  RandomOverSampler(sampling_strategy=0.8) #llevamoa la clase minorista a un equivalente del 80% de la clase mayoritaria
X_t_overs, Y_t_overs = os.fit_resample(x_train, y_train)#aplicando oversampling

X_t_overs1=pd.DataFrame(X_t_overs,columns=["Age","DailyRate","DistanceFromHome","Education",
                                        "EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel",
                                        "JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
                                        "PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours",
                                        "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance",
                                        "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
                                        "BusinessTravel","Department","EducationField","Gender",
                                        "JobRole","MaritalStatus","OverTime"])

Y_t_overs1=pd.DataFrame(Y_t_overs,columns=["Attrition"])
df_training_over=pd.concat([X_t_overs1,Y_t_overs1],axis=1)
df_training_over.head()

count_clasover=pd.value_counts(df_training_over["Attrition"])
print(count_clasover)

#fig, ax=plt.subplots(1,1)
plt.figure(figsize=(4,4))
count_clasover.plot(kind="bar",rot=0)
plt.title("Frecuencia de número de empleados - Oversampling")
plt.xlabel("Estado de Attrition")
plt.ylabel("Número de empleados")
plt.show()


### Resampling con Smote-Tomek

#número de empleados por clase en nuestra muestra de entrenamiento
print("Attrition:",np.sum(y_train==1, axis=0)) , print("No Attrition:",np.sum(y_train==0, axis=0))

res_over_und= SMOTETomek(sampling_strategy=0.8) #definimos la estrategia

X_t_res, Y_t_res = res_over_und.fit_resample(x_train, y_train)#aplicando oversampling

X_t_res1=pd.DataFrame(X_t_res,columns=["Age","DailyRate","DistanceFromHome","Education",
                                        "EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel",
                                        "JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
                                        "PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours",
                                        "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance",
                                        "YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager",
                                        "BusinessTravel","Department","EducationField","Gender",
                                        "JobRole","MaritalStatus","OverTime"])

Y_t_res1=pd.DataFrame(Y_t_res,columns=["Attrition"])

df_training_res=pd.concat([X_t_res1,Y_t_res1],axis=1)
df_training_res.head()

count_clasres=pd.value_counts(df_training_res["Attrition"])
print(count_clasres)

#fig, ax=plt.subplots(1,1)
plt.figure(figsize=(4,4))
count_clasres.plot(kind="bar",rot=0)
plt.title("Frecuencia de número de empleados - Resampling")
plt.xlabel("Estado de Attrition")
plt.ylabel("Número de empleados")
plt.show()

