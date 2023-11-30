# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:26:59 2023

@author: Usuario
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from scipy.stats import ks_2samp

#C:\Projetos\Python\ModeloCredito\cs-training.csv
#Carregando os dados
arquivo = 'cs-training.csv'

df = pd.read_csv(arquivo, index_col=0)

#Mostra a quantidade de linhas e colunas da base de dados.
df.shape

#Analise univariada
df['SeriousDlqin2yrs'].value_counts(normalize='True').plot.bar()

pd.DataFrame({'contagens':df['SeriousDlqin2yrs'].value_counts(),
              'pct':df['SeriousDlqin2yrs'].value_counts(normalize='True')},
             index=df['SeriousDlqin2yrs'].value_counts().index,
             ).style.format(precision=2, decimal=',', thousands='.',
                            formatter={'pct':'{:.1%}'})
                   
# Analise por idade
variavel = 'age'                            

sns.displot(df
            , x = variavel
            , bins = 110 # colunas dos gráficos
            , alpha = .2 # transparência
            , kde = True # linha suavizada
            , element = 'step'
            )

plt.show()

# Analise por saldos em rotativos sobre limite total.(Coluna RevolvingUtilizationOfUnsecuredLines)
variavel = 'RevolvingUtilizationOfUnsecuredLines'

sns.displot(df
            , x = variavel
            , bins = 50
            , aspect= 3
            , height= 3)



print('teste')




