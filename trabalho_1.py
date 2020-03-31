# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.decomposition import TruncatedSVD


print("Python", sys.version.split(" ")[0])
print("Numpy", np.__version__)
print("Sklearn", sklearn.__version__)
print("Matplotlib", matplotlib.__version__)



# Lendo arquivo com a matriz X e a primeira imagem
print("\nLeia o arquivo X.npy")
X = np.load("/home/jonatastbelotti/Documentos/Doutorado/3° Semestre/Algebra/Trabalhos/mo431_trabalho1/X.npy")


# Exibindo imagem
print("\nImprima a imagem da primeira pessoa.")
imagem_original = X[0].reshape((50, 37))
plt.title("Imagem original")
plt.imshow(imagem_original, cmap=cm.gray)
plt.savefig('Imagem original.png')
plt.show()


# Fatoração SVD da matriz X
print("\nFaça a fatoração svd da matriz X.")
u1, s1, vh1 = np.linalg.svd(X, full_matrices=True)
d1 = np.zeros((u1.shape[0], s1.shape[0]), dtype=np.float)
d1[:s1.shape[0], :s1.shape[0]] = np.diag(s1)
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


# Compute a matriz reduzida e a matriz reconstruída
print("\nMatriz reduzida")
u_k1 = u1[:,:100]
d_k1 = d1[:100,:100]
vh_k1 = vh1[:100,:]
reduzida1 = u_k1 @ d_k1
reconstruida1 = u_k1 @ d_k1 @ vh_k1

imagem_reconstruida1 = reconstruida1[0].reshape((50, 37))
plt.title("Reconstruida full matix")
plt.imshow(imagem_reconstruida1, cmap=cm.gray)
plt.savefig('Reconstruida full matix.png')
plt.show()

u_k2 = u2[:,:100]
d_k2 = d2[:100,:100]
vh_k2 = vh2[:100,:]
reduzida2 = u_k2 @ d_k2
reconstruida2 = u_k2 @ d_k2 @ vh_k2

imagem_reconstruida2 = reconstruida2[0].reshape((50, 37))
plt.title("Reconstruida compacta")
plt.imshow(imagem_reconstruida2, cmap=cm.gray)
plt.savefig('Reconstruida compacta.png')
plt.show()


svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
reduzida_sklearn = svd.fit_transform(X)
reconstruida_sklearn = svd.inverse_transform(reduzida_sklearn)
imagem_reconstruida_sklearn = reconstruida_sklearn[0].reshape((50, 37))
plt.title("Reconstruida TuncatedSVD Sklearn")
plt.imshow(imagem_reconstruida_sklearn, cmap=cm.gray)
plt.savefig('Reconstruida TuncatedSVD Sklearn.png')
plt.show()



