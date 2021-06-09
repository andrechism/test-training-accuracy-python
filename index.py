# -*- coding: utf-8 -*-


# Importando DATASET
import pandas as pd

dataset = pd.read_csv('dataset.csv')

del dataset['clientid']
del dataset['income']

dataset.isnull().sum()

media_idade = dataset.age.mean()
dataset.fillna(media_idade, inplace=True)

X = dataset.drop('default', axis=1)
y = dataset['default']

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=0)




# criando models / classificadores

from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier # MLP
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.tree import ExtraTreeClassifier # ExtraTreeClassifier
from sklearn.linear_model import LogisticRegressionCV #LogisticRegression - duvida se usa o CV ou não

classificadores = []

KNN_model = KNeighborsClassifier()
SVM_model = SVC()
MLP_model = MLPClassifier()
RF_model = RandomForestClassifier()
ETC_model = ExtraTreeClassifier()
LR_model = LogisticRegressionCV()


classificadores.append((' KNN', KNN_model))
classificadores.append((' SVM', SVM_model))
classificadores.append((' MLP', MLP_model))
classificadores.append((' RF', RF_model))
classificadores.append((' ETC', ETC_model))
classificadores.append((' LR', LR_model))




# Criando KFold do dataset
from numpy import mean
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, shuffle=False)


print('\nAcurácias de Treino')
for classificador in classificadores:
    # Validação Cruzada
    scores = cross_val_score(classificador[1], X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    acuracia_de_treino = mean(scores)
    print(classificador[0], acuracia_de_treino)


# Acurácia de Testes
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
    
resultado_teste = []

print('\nAcurácias de Teste')
for classificador in classificadores:
    current_model = classificador[1]
    current_model.fit(x_treino, y_treino)
    
    predicao = current_model.predict(x_teste)
    
    print(classificador[0], accuracy_score(y_teste, predicao))
   
    resultado_teste.append(predicao)
   
    
   
# Ensemble

from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(classificadores)
ensemble_model.fit(x_treino, y_treino)

predicao_voting = ensemble_model.predict(x_teste)

print('\nEnsemble voting: ', accuracy_score(y_teste, predicao_voting))


# Matrizes de Confusão

print('\nMatrizes de Confusão')
    
for predicao in resultado_teste:
    ax = plt.axes()
    
    matriz_de_confusao = confusion_matrix(y_teste, predicao)
    matriz_de_confusao = pd.crosstab(y_teste, predicao, rownames=['Atual'], colnames=['Previsto'])
    # print('matriz de confusão: ', classificador[0], predicao, matriz_de_confusao)
    
    sns.heatmap(matriz_de_confusao, annot=True, ax=ax, fmt='g')
    ax.set_title(classificador[0])
    plt.show()

    

    

