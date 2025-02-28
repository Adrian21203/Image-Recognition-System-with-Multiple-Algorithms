import numpy as np
import cv2
from numpy import linalg as la
import statistics as st
import matplotlib
from matplotlib import pyplot as plt
import os
import time
import sys

# Matricea de antrenare
def Matrice_antr(path, nr_poze):
    A = np.zeros([10304, nr_poze * 40])
    for i in range(1, 41):
        caleFolderPers = path + '\\s' + str(i) + '\\'
        for j in range(1, nr_poze + 1):
            calePozaAntrenare = caleFolderPers + str(j) + '.pgm'
            pozaAntrenare = np.array(cv2.imread(calePozaAntrenare, 0))
            pozaVect = pozaAntrenare.reshape(10304,)
            A[:, nr_poze * (i - 1) + j - 1] = pozaVect
    return A

# Algoritmul kNN (NN este k=1)
def kNN(k, nr_poze, norma, path, A, poza_cautata):
    z = np.zeros(len(A[0]))
    for i in range(len(A[0])):
        if norma == '1':
            z[i] = la.norm(A[:, i] - poza_cautata, 1)  # Norma 1 (distanța Manhattan)
        elif norma == '2':
            z[i] = la.norm(A[:, i] - poza_cautata, 2)  # Norma 2 (distanța Euclidiană)
        elif norma == 'inf':
            z[i] = la.norm(A[:, i] - poza_cautata, np.inf)  # Norma infinit (distanța supremum)
        elif norma == 'cos':
            z[i] = 1 - np.dot(A[:, i], poza_cautata) / (la.norm(A[:, i]) * la.norm(poza_cautata))
        else:
            raise ValueError("Norma necunoscută. Alege dintre: 1, 2, 'inf' sau 'cos'.")

    pozitii = np.argsort(z)[:k]
    persoane = pozitii // nr_poze + 1
    persoana_gasita = st.mode(persoane)
    return persoana_gasita

def RR(k, nr_poze, norma, path, A):
    recunoasteri = 0
    timpTotal = 0
    for i in range(1, 41):
        caleFolderPers = path + '\\s' + str(i) + '\\'
        for j in range(nr_poze + 1, 11):
            calePozaTest = caleFolderPers + str(j) + '.pgm'
            pozaTest = np.array(cv2.imread(calePozaTest, 0))

            if pozaTest is None:
                raise ValueError("poza de test nu a fost gasita!")
            else:
                pozaTest = pozaTest.reshape(10304,)

            t0 = time.perf_counter()
            i0 = kNN(k, nr_poze, norma, path, A, pozaTest)
            t1 = time.perf_counter()
            timpTotal += t1 - t0

            if i0 == i:
                recunoasteri += 1

    return recunoasteri, timpTotal

def save_stats(path, nr_poze, A):

    rr_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\kNN\\ORL_kNN_{nr_poze}_RR.txt"
    tmi_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\kNN\\ORL_kNN_{nr_poze}_TMI.txt"

    with open(rr_filename, "a") as f, open(tmi_filename, "a") as f1:
        for k in range(1, 10, 2):
            for norma in ('1', '2', 'inf', 'cos'):
                recunoasteri, timpTotal = RR(k, nr_poze, norma, path, A)
                rr = recunoasteri / (40 * (10 - nr_poze))
                tmi = timpTotal / (40 * (10 - nr_poze))
                f.write(f"{rr:.8f} ")
                f1.write(f"{tmi:.8f} ")
                print('k= ', k, ' norma=', norma)
            f.write("\n")
            f1.write("\n")
    print("Datele au fost salvate")

def read_data_from_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            values = [float(x) for x in line.split()]
            data.append(values)
    return data

def create_plots():
    rr_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\kNN\\ORL_kNN_{nr_poze}_RR.txt"
    tmi_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\kNN\\ORL_kNN_{nr_poze}_TMI.txt"
    rr_data = read_data_from_file(rr_filename)
    tmi_data = read_data_from_file(tmi_filename)

    valori_k = list(range(1, 10, 2))
    norme = ['1', '2', 'inf', 'cos']

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Statistici kNN - {nr_poze} poze de antrenament")

    # plot RR
    plt.subplot(1, 2, 1)
    for idx, k in enumerate(valori_k):
        plt.plot(norme, rr_data[idx], marker='o', linestyle='-', label=f"k={k}")
    plt.title("RR (Rata recognitie)")
    plt.ylim(0, 1)
    plt.ylabel("RR (%)")
    plt.xlabel("Norme")
    plt.legend()

    # plot TMI
    plt.subplot(1, 2, 2)
    for idx, k in enumerate(valori_k):
        plt.plot(norme, tmi_data[idx], marker='o', linestyle='-', label=f"k={k}")
    plt.title("TMI (Timp mediu interogare)")
    plt.ylabel("TMI (s)")
    plt.xlabel("Norme")
    plt.legend()

    plt.tight_layout()
    plt.show()

def searchImage(path, nr_poze, k, norma, A):
    poza_cautata = np.array(cv2.imread(path, 0))
    poza_cautata = poza_cautata.reshape(10304,)
    persoana_gasita = kNN(k, nr_poze, norma, path, A, poza_cautata)
    print(f"Persoana gasita: {persoana_gasita}")
    return persoana_gasita

if __name__ == "__main__":
    path = "C:\\Users\\win\\.spyder-py3\\att_faces"
    nr_poze = int(sys.argv[1])
    print(sys.argv)
    A = Matrice_antr(path, nr_poze)
    if len(sys.argv) > 2:
        if sys.argv[2] == "searchImage":
            image_path = sys.argv[3]
            k = int(sys.argv[4])
            norma = sys.argv[5]
            persoana_gasita = searchImage(image_path, nr_poze, k, norma, A)
        else:
            create_plots()
    else:
        create_plots()