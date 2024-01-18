# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:26:57 2023

@author: IVAN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

df = pd.read_csv('/home/and/python_poned/lab_2_AI/data.csv')
# возьмем перые 100 строк, 4-й столбец 
y = df.iloc[0:100, 4].values
# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1) # reshape нужен для матричных операций

# 4 признака
X = df.iloc[0:100, [0, 2]].values

# добавим фиктивный признак для удобства матричных вычислений
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

# зададим функцию активации - сигмоида
def sigmoid(y):
    return 1 / (1 + np.exp(-y))

# нам понадобится производная от сигмоиды при вычислении градиента
def derivative_sigmoid(y):
    return sigmoid(y) * (1 - sigmoid(y))

# инициализируем нейронную сеть 
inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 5 # задаем число нейронов скрытого слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи

# веса инициализируем случайными числами, но теперь будем хранить их списком
weights = [
    np.random.uniform(-2, 2, size=(inputSize,hiddenSizes)),  # веса скрытого слоя
    np.random.uniform(-2, 2, size=(hiddenSizes,outputSize))  # веса выходного слоя
]

#Прямой проход
def feed_forward(x):
    input_ = x # входные сигналы
    hidden_ = sigmoid(np.dot(input_, weights[0])) # выход скрытого слоя = сигмоида(входные сигналы*веса скрытого слоя)
    output_ = sigmoid(np.dot(hidden_, weights[1]))# выход сети (последнего слоя) = сигмоида(выход скрытого слоя*веса выходного слоя)

    # возвращаем все выходы, они нам понадобятся при обратном проходе
    return [input_, hidden_, output_]

#обратный проход
def backward(learning_rate, target, net_output, layers):

    # считаем производную ошибки сети
    err = (target - net_output)

    # прогоняем производную ошибки обратно ко входу, считая градиенты и корректируя веса
    # для этого используем chain rule
    # цикл перебирает слои от последнего к первому
    for i in range(len(layers)-1, 0, -1):
        # градиент слоя = ошибка слоя * производную функции активации * на входные сигналы слоя
        
        # ошибка слоя * производную функции активации
        err_delta = err * derivative_sigmoid(layers[i])       
        
        # пробрасываем ошибку на предыдущий слой
        err = np.dot(err_delta, weights[i - 1].T)
        
        # ошибка слоя * производную функции активации * на входные сигналы слоя
        dw = np.dot(layers[i - 1].T, err_delta)
        
        # обновляем веса слоя
        weights[i - 1] += learning_rate * dw

#функция обучения GD
def train(x_values, target, learning_rate):
    output = feed_forward(x_values)
    backward(learning_rate, target, output[2], output)
    return None

#функция обучения SGD
def train(x_values, target, learning_rate, batch_size):
    # Проходим по обучающим примерам пакетами размера batch_size
    for _ in range(0, len(x_values), batch_size):
        # Берем случайную строку из выборки
        j = random.randint(0, len(x_values) - batch_size)
        x_batch = x_values[j:j + batch_size]
        target_batch = target[j:j + batch_size]
        for x, t in zip(x_batch, target_batch):
            output = feed_forward(x_values)
            backward(learning_rate, target, output[2], output)
    return None        

#функия предсказания
def predict(x_values):
    return feed_forward(x_values)[-1]

#параметры обучения
iterations = 50
learning_rate = 0.01
batch_size = 20

#Стохастический градиентный спуск (SGD)
#Процесс обучения

for i in range(iterations):
    train(X, y, learning_rate, batch_size)

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' + "Средняя ошибка: " + str(np.mean(np.square(y - predict(X)))))
        
#Проверка
y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1) 
X = df.iloc[:, [0, 1, 2, 3]].values
X = np.concatenate([np.ones((len(X),1)), X], axis = 1)

pr = predict(X)
#print(sum(abs(y-(pr>0.5))))