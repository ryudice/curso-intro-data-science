#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas
import os
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
"""
Obtención del set de datos
"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)
"""
Exploración de datos
"""
print(dataset.shape)#cuantas filas y columnas vienen
print(dataset.head(30)) #mostrar los primeros 20 registros
print(dataset.describe()) #muestra valor maximos, minimos y media
print(dataset.groupby('class').size()) #agrupamos por cada instancia o llave
"""
La salida en pantalla nos muestra en este orden:
(150,5) = primer parametro nos muestra la cantidad de registros, el segundo los 5 atributos o columnas que tiene el archivo
El resultado de la muestra 30 filas.
Número de registros que hay en el dataset, la media, la desviación estándar, 
los valores máximo y mínimo de cada atributo y algunos porcentajes.
"""


# In[13]:


#mostrando la data en un gráfico
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[14]:


#Mostrar un histograma
dataset.hist()
plt.show()

