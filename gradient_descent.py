from random import Random

import numpy as np
points = [([1,1], 5), ([2,1],7), ([3,1],9), ([4,1],11), ([5,1],13)]
#points = [([2,1], 4), ([4,1], 2)]
n = len(points)
d = len(points[0][0])

# Produto escalar entre vetores w (pesos) e x (vetor de features)
def dot(w, x):
    if len(w) != len(x):
        return

    # Retorna a score (predicao) do modelo
    return sum(w[i] * x[i] for i in range(len(w)))

# subtração entre dois vetores
def sub(w, x):
    if len(w) != len(x):
        return

    return [w[i] - x[i] for i in range(len(w))]

# função de perda (diz o quanto o algoritmo está errando)
def mse(w):
    return sum((dot(w, x) - y) **2 for (x, y) in points)/n

def sMSE(w, i):
    x, y = points[i]
    return sum((dot(w,x) - y)**2)

# produto de um vetor por um escalar N
def dotN(w, n):
    return [w[i] * n for i in range(len(w))]

def dMSE(w):
    # ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z,…)
    gradient = []
    for i in range(len(w)):
        gradient.append(2 * sum(x[i] * (dot(w, x) - y) for (x, y) in points)/n)
    return gradient

def sdMSE(w, i):
    # ∇f = (∂f/∂x, ∂f/∂y, ∂f/∂z, …)
    gradient = []
    x, y = points[i]
    for i in range(len(w)):
        gradient.append(2 * x[i] * (dot(w, x) - y))
    return gradient

def gradientDescent(d):
    w = [0] * d # vetor de pesos (m, b)
    iterations = 3750
    learning_rate = 0.05
    for i in range(iterations):
        gradient = dMSE(w)
        error = mse(w)
        # nova posicao = posicao atual − η.∇J
        w = sub(w, dotN(gradient, learning_rate))
        print(f"w: {w}, mse: {error}, i:{i}")

def sGradientDescent(d):
    w = [0] * d # vetor de pesos (m, b)
    iterations = 100000
    learning_rate = 0.05
    for i in range(iterations):
        random = Random()
        gradient = sdMSE(w, random.randint(0, len(points)-1))
        error = mse(w)
        # nova posicao = posicao atual − η.∇J
        w = sub(w, dotN(gradient, learning_rate))
        print(f"w: {w}, mse: {error}, i:{i}")

# Gradiente descendente para aproximar a funcao y = 2x + 3
gradientDescent(d)