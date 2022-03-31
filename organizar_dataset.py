import tensorflow as tf
import numpy as np
import shutil
import os
import re
import math
from glob import glob
from pathlib import Path
from random import shuffle

def remove_element(list_,index_):
  clipboard = []
  for i in range(len(list_)):
      if i is not index_:
          clipboard.append(list_[i])
  return clipboard


def pegar_dir(tamanho_string, padrao_pasta, diretorio):
  lista_caminhos = []

  for i in range(len(diretorio)):
    for pastaRaiz_, dirs_, arquivos_ in os.walk(diretorio[i]):
      for dir in dirs_:
        if dir[:tamanho_string] == padrao_pasta:
          lista_caminhos.append(diretorio[i]+"/"+dir)
  return lista_caminhos

def pegar_tipos(numeros_tipos, dataset):
  dir_tipos = []
  num = 0
  for pastaRaiz, dirs, arquivos in os.walk(dataset):
    for dir in dirs:
      if num == numeros_tipos:
        break
      dir_tipos.append(pastaRaiz +"/"+ dir)
      num += 1
  return dir_tipos

def criar_dataset(tipo_imagem,porcentagem_teste):
  # --------------------------------------------------------------------------#
  dataset_benign = os.path.join("BreaKHis_v1", "BreaKHis_v1", "histology_slides", "breast","benign","SOB")
  dataset_malignant = os.path.join("BreaKHis_v1", "BreaKHis_v1", "histology_slides", "breast","malignant","SOB")
  num_tipos = 4
  tipo_image = str(tipo_imagem) #100 ou 200 ou 400 ou 40x


  if tipo_image[2] == '0':
    tipo_imagem = tipo_image+'X'
  else:
    pass

  print('Pegas as imagens do tipo '+tipo_imagem)
  #imagens benign
  dir_tipos_benign = pegar_tipos(num_tipos, dataset_benign)

  dir_parcientes_benign = pegar_dir(3,"SOB",dir_tipos_benign)
  dir_tipo_imagem_paciente_benign = pegar_dir(3,tipo_image, dir_parcientes_benign)


  #imagens malignant
  dir_tipos_malignant = pegar_tipos(num_tipos, dataset_malignant)

  dir_parcientes_malignant = pegar_dir(3,"SOB",dir_tipos_malignant)
  dir_tipo_imagem_paciente_malignant = pegar_dir(3,tipo_image, dir_parcientes_malignant)

  pacientes_benign = []
  pacientes_malignant = []
  paciente = []

  #faz uma lista de pacientes que tem uma lista de imagem de cada um

  for pasta in dir_tipo_imagem_paciente_benign:
    for img in glob(os.path.join(pasta,"*")):
      paciente.append(img)
    pacientes_benign.append(paciente)
    paciente = []

  for pasta in dir_tipo_imagem_paciente_malignant:
    for img in glob(os.path.join(pasta,"*")):
      paciente.append(img)
    pacientes_malignant.append(paciente)
    paciente = []

  # --------------------------------------------------------------------------#
  #separa 30 porcente pra o teste

  #mistura os tipos de cancer 
  shuffle(pacientes_benign)
  shuffle(pacientes_malignant)

  total_pacientes = pacientes_malignant

  for paciente in pacientes_benign:
    total_pacientes.append(paciente)

  tam_total = len(total_pacientes)

  shuffle(total_pacientes)

  tam_teste_pacientes = math.ceil(tam_total * porcentagem_teste)
  tam_treino_pacientes = tam_total - tam_teste_pacientes

  teste_pacientes = []
  treino_pacientes = []

  cont = 0
  for i in range(tam_teste_pacientes):
    teste_pacientes.append(total_pacientes[i])
    cont = i

  for i in range(tam_treino_pacientes):
    cont+=1
    treino_pacientes.append(total_pacientes[cont])

  print('teste',tam_teste_pacientes,'treino',tam_treino_pacientes)


  # --------------------------------------------------------------------------#
  
  #pega so as imagens e criar uma nova pasta com um dataset de benign e malignant
  dataset = os.path.join("BreaKHis_v1", "BreaKHis_v1", "histology_slides", "breast")

  classes = ["benign", "malignant"] #pastas

  #criando dataset
  dataset_nome = "dataset_"+tipo_imagem
  for c in classes:
    os.makedirs(os.path.join(dataset_nome,"treino",c),exist_ok=True)
    os.makedirs(os.path.join(dataset_nome,"teste",c),exist_ok=True)

  
  print("Copiando as fotos para o treino...")
  padrao = re.compile('benig.')
  for paciente in treino_pacientes:
    for img in paciente:
      vazio = padrao.search(img)

      if vazio == None:
        shutil.copy(img,os.path.join(dataset_nome,"treino", "malignant",Path(img).name))
      else:
        shutil.copy(img,os.path.join(dataset_nome,"treino", "benign",Path(img).name))
      
  print('Copiando as fotos para o teste...')
  for paciente in teste_pacientes:
    for img in paciente:
      vazio = padrao.search(img)

      if vazio == None:
        shutil.copy(img,os.path.join(dataset_nome,"teste", "malignant",Path(img).name))
      else:
        shutil.copy(img,os.path.join(dataset_nome,"teste", "benign",Path(img).name))

  #shutil.rmtree("BreaKHis_v1") apaga a pasta original

  print('')
  pass


criar_dataset(100,0.3)

criar_dataset(200,0.3)

criar_dataset(400,0.3)

criar_dataset('40X',0.3)