#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
#Separamos los datos en conjuntos de entrenamiento y validación
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
"""
Validación cruzada, es vital para validar modelos en proyectos de IA
Consiste en repetir y calcular la media aritmética de las medidas de evaluación que obtenemos sobre diferentes particiones
"""
seed = 7
scoring = 'accuracy'#precisión la librería solo acepta terminos en inglés :(
"""
Construcción de modelos, ¿qué modelo utilizar?

Regresión logística (LR)
Análisis del Discriminante lineal (LDA)
K- Vecinos más cercanos (KNN)
Árboles de clasificación y regresión (CART)
Gaussiana Naive Bayes (NB)
Máquinas de vectores de soporte (SVM)

Como no sabemos la mejor opción, vamos a probar y que nos muestre el mejor desempeño.
"""
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
"""
Como observamos, el modelo que da un mayor valor de precisión es KNN (98%)
Pero siempre es mejor mostrar en gráficos.
En el diagrama de caja y bigotes se aprecia claramente cómo la precisión para muchas
de las muestras en los modelos KNN, NB y SVM llega a ser del 100%, 
mientras que el modelo que ofrece menor precisión es la regresión lineal LR.
"""
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""
Aplicamos el modelo para realizar las predicciones
El resultado es la precisión es 0.93, un 93%, un dato muy bueno.
La matriz de confusión, por su parte, nos indica el número de puntos 
para los cuales la predicción del modelo ha sido correcta 
(valores en la diagonal: 7+10+11=28)
y los elementos fuera la diagonal son los errores de predicción (2).
Por tanto, podemos concluir que es un buen modelo que podemos aplicar con tranquilidad a un nuevo dataset.
"""
SVM = SVC()
SVM.fit(X_train, Y_train)
predictions = SVM.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

