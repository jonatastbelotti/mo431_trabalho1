# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


print("Python", sys.version.split(" ")[0])
print("Numpy", np.__version__)
print("Matplotlib", matplotlib.__version__)



# Lendo arquivo com a matriz X e a primeira imagem
print("\nLeia o arquivo X.npy")
X = np.load("/home/jonatastbelotti/Documentos/Doutorado/3° Semestre/Algebra/Trabalhos/mo431_trabalho1/X.npy")


# Exibindo imagem
print("\nImprima a imagem da primeira pessoa.")
imagem_1 = X[0].reshape((50, 37))
plt.imshow(imagem_1, cmap=cm.gray)
# plt.show()


# Fatoração SVD da matriz X
print("\nFaça a fatoração svd da matriz X.")
u1, s1, vh1 = np.linalg.svd(X, full_matrices=True)
d1 = np.zeros((u1.shape[0], s1.shape[0]), dtype=complex)
print("Fatoração full matix")
print(u1.shape, d1.shape, vh1.shape)

u2, s2, vh2 = np.linalg.svd(X, full_matrices=False)
d2 = np.diag(s2)
print("Fatoração compacta")
print(u2.shape, d2.shape, vh2.shape)


# Verificando formulação
print("\nVerifique a formulação compacta do SVD")
media_X = np.mean(X)
print("Média X = %f" % media_X)

dif2 = X - np.linalg.multi_dot([u2, d2, vh2])
dif_max2 = np.max(np.absolute(dif2))
print("Erro compacta = %f" % dif_max2)


