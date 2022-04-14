#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      arthu
#
# Created:     14/04/2022
# Copyright:   (c) arthu 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import math
import copy
import random
import matplotlib.pyplot as plt

def CholeskyAlternative (A):
    n = A.shape[0]
    M = copy.deepcopy(A)
    L = np.eye(n)   # création de la matrice triangulaire inférieure avec des coefficients diagonaux égaux à 1
    D = np.zeros((n, n))    # création de la matrice diagonale
    s=0
    Q = 0   # compteur de calculs matriciels
    for k in range(n):
        s = 0
        for j in range(k):
            s += (L[k,j]**2) * D[j,j]
            Q = Q + 1
        D[k,k] = M[k,k] - s
        Q = Q + 1

        for i in range(k+1,n):
            s = 0
            for j in range(1,k):
                s += L[i,j] * L[k,j] * D[j,j]
                Q = Q + 1
            L[i,k] = (M[i,k] - s) / D[k,k]
            Q = Q + 1
    return L, D, Q

def ResolCholeskyAlternative(A, B) :
    n = A.shape[0]
    L = CholeskyAlternative(A)[0]
    T = np.transpose(L)
    D = CholeskyAlternative(A)[1]
    Q = CholeskyAlternative(A)[2]
    Y = np.zeros(n)     # vecteur colonne Y
    X = np.zeros(n)     # vecteur colonne X
    P = np.dot(L, D)    # il s'agit d'une matrcice triangulaire inférieure, ce qui facilite les calculs
    Y[0] = B[0] / P[0,0]
    k = 1   # permet de rester sur la même ligne dans la boucle j dans la double boucle sur Y
    l = n-2   # permet de rester sur la même ligne dans la boucle j dans la double boucle sur X
    for i in range (1, n) :     # cette double boucle permet de calculer Y
        s = 0
        for j in range (1, n) :
            s = s + P[k, j-1]*Y[j-1]
            Q = Q + 1
        Y[i] = (B[i] - s) / P[i, i]
        Q = Q + 1
        k = k + 1
    X[n-1] = Y[n-1] / T[n-1, n-1]
    Q = Q + 1
    for i in range (n-2, -1, -1) :  # cette boucle permet de calculer X
        s = 0
        for j in range (n-2, -1, -1) :
            s = s + T[l, j+1]*X[j+1]
            Q = Q + 1
        X[i] = (Y[i] - s) / T[i, i]
        Q = Q + 1
        l = l - 1
    return X, Q

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)



y1 = np.zeros(20)   # matrice ligne représentant le nombre de calculs matriciels en fonction du nombre de lignes
z1 = []             # matrice ligne représentant les erreurs en fonction du nombre de lignes avec la méthode de Cholesky alternative
z2 = []             # matrice ligne représentant les erreurs en fonction du nombre de lignes avec la méthode Solve
for i in range (1, 21) :
    y2 = 0
    y3 = 0
    B = np.random.rand(i)
    A = 2*np.eye(i) - np.diag(np.ones(i-1),1) - np.diag(np.ones(i-1), -1)
    C = ResolCholeskyAlternative(A, B)
    D = np.linalg.solve(A, B)
    y1[i-1] = C[1]
    for j in range (0, i) :
        y2 = y2 + abs(np.dot(A, C[0])[j] - B[j])  # chaque élément de y3 est multiplié par 10^16 afin de rendre les erreurs plus visibles
        y3 = y3 + abs(np.dot(A, D)[j] - B[j])
    z1.append(y2)
    z2.append(y3)

x = range (0, 20)

ydata1 = []
for i in range (0, 20) :
    ydata1.append(y1[i])
plt.plot(x, ydata1, label = "Méthode de Cholesky alternative")
plt.title("Evolution du nombre de calculs matriciels en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Nombre d'opérations")
plt.grid()
plt.legend()
plt.draw()
plt.show()

zdata1 =[]
for i in range (0, 20) :
    zdata1.append(z1[i])
zdata2 = []
for i in range (0, 20) :
    zdata2.append(z2[i])
plt.plot(x, zdata1, label = "Méthode de Cholesky alternative")
plt.plot(x, zdata2, label = "Méthode Solve")
plt.title("Evolution des erreurs en fonction de la taille de la matrice")
plt.xlabel("Taille de la matrice")
plt.ylabel("Erreur")
plt.grid()
plt.legend()
plt.draw()
plt.show()



