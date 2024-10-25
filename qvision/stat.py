import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat



# Carica i dati
Train = pd.read_csv('mnist_test.csv')
Full = Train.values
del Train
zero = Full[Full[:, 0] == 0, :]
one = Full[Full[:, 0] == 1, :]
del Full

N = 50
N_tot = min(zero.shape[0], one.shape[0])
qN = np.random.permutation(N_tot)[:N]
zero = zero[qN, 1:]
one = one[qN, 1:]

# Carica GS_train.mat
data = loadmat('GS_train.mat')
zerone_modulated = data['zerone_modulated']
del data

GS_0 = zerone_modulated[zerone_modulated[:, 0] == 0, :]
GS_1 = zerone_modulated[zerone_modulated[:, 0] == 1, :]
GS_0 = GS_0[qN, 1:]
GS_1 = GS_1[qN, 1:]
del zerone_modulated

# Visualizzazione delle immagini
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i + 1)
    if i < 8 and (i % 4 == 0 or i % 4 == 2):
        fig = GS_0[i - 1].reshape(28, 28)
        plt.imshow(np.abs(fig) ** 2 / np.max(np.abs(fig) ** 2), cmap='gray')
    elif i < 8 and (i % 4 == 1 or i % 4 == 3):
        fig = zero[i].reshape(28, 28).T
        plt.imshow(fig / np.max(fig), cmap='gray')
    elif i >= 8 and (i % 4 == 0 or i % 4 == 2):
        fig = GS_1[i - 1].reshape(28, 28)
        plt.imshow(np.abs(fig) ** 2 / np.max(np.abs(fig) ** 2), cmap='gray')
    elif i >= 8 and (i % 4 == 1 or i % 4 == 3):
        fig = one[i].reshape(28, 28).T
        plt.imshow(fig / np.max(fig), cmap='gray')
plt.show()

# Calcolo delle distanze
dist0 = np.zeros((N, N))
dist1 = np.zeros((N, N))
dist01 = np.zeros((N, N))
dist10 = np.zeros((N, N))
adist0 = np.zeros((N, N))
adist1 = np.zeros((N, N))
adist01 = np.zeros((N, N))
adist10 = np.zeros((N, N))
odist0 = np.zeros((N, N))
odist1 = np.zeros((N, N))
odist01 = np.zeros((N, N))
odist10 = np.zeros((N, N))

for i in range(N):
    vg0 = GS_0[i]
    vg1 = GS_1[i]
    zg0 = zero[i]
    zg1 = one[i]
    for j in range(i):
        vg0b = GS_0[j]
        vg1b = GS_1[j]
        zg0b = zero[j]
        zg1b = one[j]
        dist0[i, j] = np.abs(np.dot(vg0, vg0b)) ** 2
        dist1[i, j] = np.abs(np.dot(vg1, vg1b)) ** 2
        dist01[i, j] = np.abs(np.dot(vg0, vg1b)) ** 2
        dist10[i, j] = np.abs(np.dot(vg1, vg0b)) ** 2
        adist0[i, j] = (np.abs(vg0) * np.abs(vg0b)).sum() ** 2
        adist1[i, j] = (np.abs(vg1) * np.abs(vg1b)).sum() ** 2
        adist01[i, j] = (np.abs(vg0) * np.abs(vg1b)).sum() ** 2
        adist10[i, j] = (np.abs(vg1) * np.abs(vg0b)).sum() ** 2
        odist0[i, j] = (np.sqrt(zg0) @ np.sqrt(zg0b)) ** 2 / (np.linalg.norm(np.sqrt(zg0)) * np.linalg.norm(np.sqrt(zg0b))) ** 2
        odist1[i, j] = (np.sqrt(zg1) @ np.sqrt(zg1b)) ** 2 / (np.linalg.norm(np.sqrt(zg1)) * np.linalg.norm(np.sqrt(zg1b))) ** 2
        odist01[i, j] = (np.sqrt(zg0) @ np.sqrt(zg1b)) ** 2 / (np.linalg.norm(np.sqrt(zg0)) * np.linalg.norm(np.sqrt(zg1b))) ** 2
        odist10[i, j] = (np.sqrt(zg1) @ np.sqrt(zg0b)) ** 2 / (np.linalg.norm(np.sqrt(zg1)) * np.linalg.norm(np.sqrt(zg0b))) ** 2

ndist0 = np.diag(dist0)
dist0 = dist0[~np.eye(N, dtype=bool)]
ndist1 = np.diag(dist1)
dist1 = dist1[~np.eye(N, dtype=bool)]
dist_u = np.concatenate((dist0, dist1))
dist10 = dist10[~np.eye(N, dtype=bool)]
dist_d = np.concatenate((dist01.ravel(), dist10.ravel()))

# Istogrammi
plt.figure()
plt.hist(dist_u, density=True, alpha=0.5, label='dist_u')
plt.hist(dist_d, density=True, alpha=0.5, label='dist_d')
plt.legend()
plt.show()

nadist0 = np.diag(adist0)
adist0 = adist0[~np.eye(N, dtype=bool)]
nadist1 = np.diag(adist1)
adist1 = adist1[~np.eye(N, dtype=bool)]
adist_u = np.concatenate((adist0, adist1))
adist10 = adist10[~np.eye(N, dtype=bool)]
adist_d = np.concatenate((adist01.ravel(), adist10.ravel()))

plt.figure()
plt.hist(adist_u, density=True, alpha=0.5, label='adist_u')
plt.hist(adist_d, density=True, alpha=0.5, label='adist_d')
plt.legend()
plt.show()

nodist0 = np.diag(odist0)
odist0 = odist0[~np.eye(N, dtype=bool)]
nodist1 = np.diag(odist1)
odist1 = odist1[~np.eye(N, dtype=bool)]
odist_u = np.concatenate((odist0, odist1))
odist10 = odist10[~np.eye(N, dtype=bool)]
odist_d = np.concatenate((odist01.ravel(), odist10.ravel()))

plt.figure()
plt.hist(odist_u, density=True, alpha=0.5, label='odist_u')
plt.hist(odist_d, density=True, alpha=0.5, label='odist_d')
plt.legend()
plt.show()

# Ultimo istogramma con scaling
plt.figure()
dist_u2 = 40 * dist_u
plt.hist(dist_u2, density=True, alpha=0.5, label='dist_u2')
dist_d2 = 40 * dist_d
plt.hist(dist_d2, density=True, alpha=0.5, label='dist_d2')
plt.legend()
plt.show()
