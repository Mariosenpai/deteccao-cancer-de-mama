import tensorflow as tf
import numpy as np
import shutil
import random
import sklearn
import json
import matplotlib.pyplot as plt
import os
import re
import math
import keras
from glob import glob
from pathlib import Path
from random import shuffle
from PIL import Image
from tensorflow.keras import layers 
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
import sklearn


SEED = 1
'''
Preparando as imagens para o MIL
'''

from Preparando_Para_MIL import lista_caminhos
from Preparando_Para_MIL import transforma_np_array
from Preparando_Para_MIL import criar_instancias
from Preparando_Para_MIL import detencao_de_ruido


from hyperopt import hp, tpe, Trials, fmin, STATUS_OK

#------------------------------------------------------------------------------#

#separa os dados

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def evaluate(params):
    print("\n\nExecutando: " + str(params))
    #------------------------------------------------------------------------------#

    #treino

    caminho_100 = os.path.join('dataset_100X', 'treino' )
    caminho_200 = os.path.join('dataset_200X', 'treino' )
    caminho_400 = os.path.join('dataset_400X', 'treino' )
    caminho_40 = os.path.join('dataset_40X', 'treino' )

    data_treino_100 , labels_treino_100 = lista_caminhos(caminho_100)
    data_treino_200 , labels_treino_200 = lista_caminhos(caminho_200)
    data_treino_400 , labels_treino_400 = lista_caminhos(caminho_400)
    data_treino_40 , labels_treino_40 = lista_caminhos(caminho_40)

    #------------------------------------------------------------------------------#
    #teste 
    caminho_teste_100 = os.path.join('dataset_100X', 'teste' )
    caminho_teste_200 = os.path.join('dataset_200X', 'teste' )
    caminho_teste_400 = os.path.join('dataset_400X', 'teste' )
    caminho_teste_40 = os.path.join('dataset_40X', 'teste' )

    data_teste_100 , labels_teste_100 = lista_caminhos(caminho_teste_100)
    data_teste_200 , labels_teste_200 = lista_caminhos(caminho_teste_200)
    data_teste_400 , labels_teste_400 = lista_caminhos(caminho_teste_400)
    data_teste_40 , labels_teste_40 = lista_caminhos(caminho_teste_40)

    #------------------------------------------------------------------------------#

    data = data_treino_100
    labels = labels_treino_100
    data_teste = data_teste_100
    labels_teste = labels_teste_100

    #------------------------------------------------------------------------------#



    data = data + data_teste
    labels = list(labels) + list(labels_teste)
    labels = np.array(labels)

    X = data
    y = labels

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    skf.get_n_splits(X, y)

    X = np.array(X)
    y = np.array(y)

    cont = 1
    for train_index, test_index in skf.split(X,y):

        TAM_INSTANCIA = params['Tam_instancia']
        BAG_SIZE = params['Instancias']
        EPOCAS = params['Epocas']
        BATCH_SIZE = params['Batch_size']
        
        print('#---------------------------------------------------------------------#')
        print(f"Validacao cruzada, foram {cont} de {n_splits}")

        #----------------------------------------------------------------------------#
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        data_teste = X_test 
        labels_teste = y_test

        #tamanho da base de teste em porcetagem
        base_teste = 0.25

        data_treino , val_treino, labels_treino,labels_val = train_test_split(X_train , y_train,
                                                                test_size = base_teste, 
                                                                random_state = SEED)
        print(f'treino = {len(data_treino)}\nValidacao = {len(val_treino)}\nTeste = {len(data_teste)}')


        '''Tranformando em np.array as imagens'''
        #----------------------------------------------------------------------------#
        print('\nTransformando os caminhos em imagens e depois e np.array:')
        print('Dataset treino')
        data_treino_np = transforma_np_array(data_treino)
        print('Dataset validacao')
        data_val_np = transforma_np_array(val_treino)
        print('Dataset teste')
        data_teste_np = transforma_np_array(data_teste)


        '''Transformando as imagens em instancias '''
        #------------------------------------- -----------------------------------------#



        print('\ninstancias por bag')
        print(params['Instancias'])
        print('Criando Instancias para treino')
        data_treino_bag = criar_instancias(data_treino_np, TAM_INSTANCIA, TAM_INSTANCIA, params['Instancias'])
        print('Criando Instancias para validacao')
        val_treino_bag = criar_instancias(data_val_np, TAM_INSTANCIA, TAM_INSTANCIA,params['Instancias'])
        print('Criando Instancias para teste')
        data_teste_bag = criar_instancias(data_teste_np, TAM_INSTANCIA, TAM_INSTANCIA,params['Instancias'])

        tam_treino = len(data_treino_bag[0])
        
        #------------------------------------------------------------------------------#
        
        porcentagem_ruido = 0.60
        print('Incobrindo os ruidos do treino')
        treino = detencao_de_ruido(data_treino_bag, tam_treino,porcentagem_ruido )
        validacao = val_treino_bag
        teste = data_teste_bag

        #------------------------------------------------------------------------------#

        '''
            MIL
        '''

        from MIL import create_model
        from MIL import compute_class_weights
        from MIL import plot
        from MIL import predict




        instance_shape = treino[0][0].shape
        caminho_load = 'modelo_pre_treinado_16'
        # models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]
        print(instance_shape)
        print(np.array(treino).shape)
        neuronios = params['Neuronios']
        model = create_model(instance_shape,BAG_SIZE,neuronios)

        # Show single model architecture.
        # print(model.summary())
        

        # Training model(s).
        steps = math.ceil(tam_treino/BATCH_SIZE)

        

        print(f'Shape = {instance_shape}\nEpocas = {EPOCAS}\nBatch Size = {BATCH_SIZE}\n')
        # model.trainable = True
        model.fit(
            treino,
            labels_treino,
            steps_per_epoch =steps ,
            validation_data=(validacao, labels_val),
            epochs=int(EPOCAS),
            class_weight=compute_class_weights(labels_val),
            batch_size=BATCH_SIZE,
            # callbacks = [TensorBoard(log_dir = (os.path.join('logs' , str(cont+1))))],
        )
        
        model.save(caminho_load+'_128')


        y_pred = model.predict(teste)
        y_pred = np.argmax(y_pred, axis=1)
        #print(y_pred.shape)

        acc = sklearn.metrics.accuracy_score(labels_teste, y_pred)
        f1 = sklearn.metrics.f1_score(labels_teste, y_pred)
        precision = sklearn.metrics.precision_score(labels_teste, y_pred)
        recall = sklearn.metrics.recall_score(labels_teste, y_pred)

        print("Acurácia: " + str(acc))    
        #------------------------------------------------------------------------------#
        '''salva o log'''

        
        info = 'Validacao Cruzada - ' +str(cont) + ' de 5\n'+ str(EPOCAS)+' Epocas, bags com '+str(BAG_SIZE)+' instancias\n'
        classificacao_geral = classification_report(labels_teste, y_pred)
        print(classificacao_geral,'\n')
        print('matriz de confusao:')
        mc = sklearn.metrics.confusion_matrix(labels_teste, y_pred)
        print(mc)
        linha = '\n#-----------------------------------------------------#\n'
        log = linha + info + classificacao_geral
        log = log +'\nMatriz de confussao:\n' + str(mc)
        log = log + '\n\n\nAcuracia: ' + str(acc)  
        log = log + '\n\nF1_Score: ' + str(f1) 
        log = log + '\n\nPrecision: ' + str(precision)
        log = log + '\n\nRecall: ' + str(recall)
        log = log +'\n'+ linha

        nome_log = 'log_'+str(EPOCAS)+'_'+str(neuronios) +'_'+str(BAG_SIZE)+'_'+str(BATCH_SIZE)+'.txt'
        caminho_log = os.path.join('Logs', nome_log)
        with open(caminho_log , 'a') as arquivo:
            arquivo.write(log);
        cont=cont + 1

    return {'loss': 1-acc, 'status': STATUS_OK}

space = {
    'Epocas': hp.choice('Epocas', [10,16,20]),
    'Batch_size' : hp.choice('Batchsize', [4 ,8 ,16]),
    'Neuronios' : hp.choice('Neuronios', [512, 1024]),
    'Instancias': hp.choice('Instancias', [16]),
    'Tam_instancia': hp.choice('Tam_instancia', [64])
}

best = fmin(evaluate, 
            space,
            algo=tpe.suggest,
            max_evals=15)

print(best)    

#definição de espaço de busca


# salva_pesos = {
#     'Epocas': 16,
#     # 'Batchsize' : hp.choice('Batchsize', [16, 32, 48]),
#     'Neuronios' : 32,
#     'Instancias': 16,
# }

# evaluate(salva_pesos)