import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import random
import cv2
from glob import glob
from PIL import Image

def transforma_np_array(lista):
  '''
  transformando a imagem em uma lista de np.array
  '''
  treino = []
  for img in lista:  
    image = Image.open(img)
    img_np = np.array(image)
    treino.append(img_np)
  return treino


def lista_caminhos(caminho):

  labels = []
  data = sorted(glob(os.path.join(caminho, 'benign','*') ))

  for _ in data:
    labels.append(np.array([0]))

  data = data + sorted(glob(os.path.join(caminho, 'malignant','*')))

  tam = len(data[len(labels):])
  for _ in range(tam):
    labels.append(np.array([1]))

  return data , np.array(labels)

def redimencionalizar(instancia, altura, largura):
  img_instancia = Image.fromarray(instancia)
  res_instancia = img_instancia.resize((altura, largura))
  instancia = np.array(res_instancia)
  return instancia

def instancia(img, altura: int = None, largura: int = None):
  
  instancia = []
  tam_x = len(img)
  tam_y = len(img[0])

  metade_x = math.ceil(tam_x/2)
  metade_y = math.ceil(tam_y/2)

  instancia_1 = img[:metade_x , :metade_y]
  instancia_2 = img[metade_x: , metade_y:]
  instancia_3 = img[:metade_x , metade_y:]
  instancia_4 = img[metade_x: , :metade_y] 

  #redimencionlizar as instancia
  if altura != None or largura != None:
    instancia_1 = redimencionalizar(instancia_1 ,altura , largura)
    instancia_2 = redimencionalizar(instancia_2 ,altura , largura)
    instancia_3 = redimencionalizar(instancia_3 ,altura , largura)
    instancia_4 = redimencionalizar(instancia_4 ,altura , largura)

  instancia.append(instancia_1)
  instancia.append(instancia_2)
  instancia.append(instancia_3)
  instancia.append(instancia_4)

  return instancia 

def criar_instancias(lista_imagens,altura_instancia, largura_instancia,num_instancias):

  lista_bags = []
  if num_instancias == 16:
    for img in lista_imagens:
      bag = []
      instancias = instancia(img)

      for i in instancias:
        sub_instancia = instancia(i,altura_instancia , largura_instancia)
        bag = bag + sub_instancia
      lista_bags.append(bag)
    
    lista_bags = np.moveaxis(lista_bags, [1], [0])
    lista_bags = list(lista_bags)

    return lista_bags
  if num_instancias == 64:

    for img in lista_imagens:
      bag = []
      instancias = instancia(img)

      for i in instancias:
        sub_instancia_16 = instancia(i)
        for j in sub_instancia_16:
          sub_instancia_64 = instancia(j,altura_instancia,largura_instancia )
          bag = bag + sub_instancia_64
      lista_bags.append(bag)
    
    lista_bags = np.moveaxis(lista_bags, [1], [0])
    lista_bags = list(lista_bags)

    return lista_bags




def grayscale(colored):
    w, h = colored.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = colored.getpixel((x,y))
            # média ponderada das coordenadas RGB
            lum = int(0.3*pxl[0] + 0.59*pxl[1] + 0.11*pxl[2])
            img.putpixel((x,y), (lum, lum, lum))
    return img

def media_grayscale(colored):
    w, h = colored.size
    img = Image.new("RGB", (w, h))

    for x in range(w):
        for y in range(h):
            pxl = colored.getpixel((x,y))
            # média das coordenadas RGB
            lum = (pxl[0] + pxl[1] + pxl[2])//3
            img.putpixel((x,y), (lum, lum, lum))
    return img

def transforme_imagem_binario(imagem):
  image = Image.fromarray(imagem)
  img_cinza = grayscale(image)
  img_cinza = np.array(img_cinza)
  a,img_binario = cv2.threshold(img_cinza, 220, 255, cv2.THRESH_BINARY)
  # plt.imshow(img_binario)
  return img_binario

def porcentagem_zeros(i):  
  vazio = np.array([255,255,255])

  zeros = 0
  uns = 0
  for x in i:
    for y in x:
      if y in vazio:
        zeros+=1
      else:
        uns+=1

  tam = len(i)*len(i)
  zeros = zeros/tam
  uns = uns/tam

  # print(f'quantidade de zeros {zeros},  quantidade de uns {uns}\nTamanho total {tam}')
  return zeros

def detencao_de_ruido(data,tam_bag,porcentagem_ruido,tam_instancia: int =16):
  """
  dataset
  tamanho da bag
  porcentagem de ruido maximo na imagem
  tamanho da instancia padrao 16
  """
  data_novo = data
  data_auxilar = data
  for bag in range(tam_bag):

    vazio =[]
    for index in range(tam_instancia):
      # print(f'bag {bag}; instancia {index}')
      i = transforme_imagem_binario(data_auxilar[index][bag])
      por_zeros = porcentagem_zeros(i)

      if por_zeros >= porcentagem_ruido:
        # print(f'Porcentagem de banco {por_zeros*100}')
        vazio.append(index)
      else:
        # print(f'instacia {index} com baixo ruido')
        pass

    index_imagens = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    
    if len(vazio) == len(index_imagens):
      continue 
    
    for v in vazio:
      index_imagens.remove(v) 

    for v in vazio:
      index_random = random.choice(index_imagens)
      duplicada = data[index_random][bag] 
      data_novo[v][bag] = cv2.flip(duplicada, 1)

    print('\n')
  return data_novo