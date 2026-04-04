

import numpy as np
#points = [(1,5), (2,7), (3,9), (4,11), (5,13)]
points = [([2,1], 4), ([4,1], 2)]
n = len(points)
d = len(points[0][0])

def dot(w, x):
    return sum(w[i] * x[i] for i in range(len(w)))

def sub(w, x):
    return [w[i] - x[i] for i in range(len(w))]

def mse(w): # mean squared error = cost function
    return sum((dot(w, x) - y) ** 2 for (x, y) in points)/n

def productByScalar(w, n):
    return [w[i] * n for i in range(len(w))]

def dMSE(w):
    gradient = []
    for i in range(len(w)):
        gradient.append(2 * sum(x[i] * (dot(w, x) - y) for (x, y) in points)/n)
    return gradient

def gradient_descent(d):
    w = [0] * d
    iterations = 10000
    learning_rate = 0.01
    for i in range(iterations):
        gradient = dMSE(w)
        error = mse(w)
        w = sub(w, productByScalar(gradient, learning_rate))
        print(f"w: {w}, mse: {error}, i:{i}")

gradient_descent(d)