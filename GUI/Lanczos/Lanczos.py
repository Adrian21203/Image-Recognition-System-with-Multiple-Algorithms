import numpy as np
import cv2
import time
from numpy import linalg as la
from matplotlib import pyplot as plt
from numpy.linalg import norm
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

def NN(A, norma, poza_cautata):
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
    pozitia = np.argmin(z)
    return pozitia

# Algoritmul Lanczos pentru calculul vectorilor proprii (preprocesarea)
def lanczos(A, k):
    m,n = A.shape
    q = np.zeros((m, k + 2))  # Matrice pentru vectorii ortonormali
    q[:, 1] = np.ones(m) 
    q[:, 1] = q[:, 1] / norm(q[:, 1])
    
    alpha = 0  # Diagală principală
    beta = 0   # Diagonală secundară
    for i in range(1, k + 1):
        w = np.dot(A, np.dot(A.T, q[:, i])) - beta * q[:, i - 1] 
        alpha = np.dot(q[:, i].T, w) 
        w = w - alpha * q[:, i] 
        beta = norm(w)  # Calculul lui beta
        
        if beta < 1e-10:  # Verificare pentru convergență (evit diviziunea cu zero)
            break
        q[:, i + 1] = w / beta  # Normalizare și stocare q_{i+1}
    
    # Eliminăm primii doi vectori din q pentru HQPB
    HQPB = q[:, 2:k+2]  # Selectăm doar vectorii ortonormali necesari
    
    # Calculăm media pe coloane (pentru imagini reprezentate ca vectori-coloană)
    media = np.mean(A, axis=1)
    
    # Centrăm matricea A
    A_centrata = A - media[:, np.newaxis]
    
    # Calculăm proiecțiile
    proiectii = np.dot(A_centrata.T, HQPB)
    
    return HQPB, media, proiectii

def interogare(calePozaTest, media, HQPB, proiectii, norma):
    poza_test = np.array(cv2.imread(calePozaTest, 0))
    poza_test = poza_test.reshape(10304,)
    poza_test_centrata = poza_test - media
    pr_test = np.dot(poza_test_centrata, HQPB)
    pozitia = NN(proiectii.T, norma, pr_test)
    return int(pozitia / nr_poze)

def timpPreprocesare(k):
    start_preprocesare = time.time()
    HQPB, media, proiectii = lanczos(A, k)
    end_preprocesare = time.time()
    timp_preprocesare = end_preprocesare - start_preprocesare
    print(f"Timpul de preprocesare: {timp_preprocesare:.8f} secunde")
    return timp_preprocesare, HQPB, media, proiectii

def statistici(path, nr_poze, media, HQPB, proiectii, norma):
    corecte = 0
    timp_total_interogare = 0

    for i in range(1, 41):
        for j in range(nr_poze + 1, 11):
            calePozaTest = path + f'\\s{i}\\{j}.pgm'
            start_interogare = time.time()
            pozitia_identificata = interogare(calePozaTest, media, HQPB, proiectii, norma)
            end_interogare = time.time()
            timp_interogare = end_interogare - start_interogare
            timp_total_interogare += timp_interogare

            if pozitia_identificata == i - 1:
                corecte += 1

    return corecte, timp_total_interogare

def save_stats(path, nr_poze):
    rr_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_RR.txt"
    tmi_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_TMI.txt"
    tpp_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_TPP.txt"
    with open(rr_filename, "a") as f, open(tmi_filename,"a") as f1, open(tpp_filename,"a") as f2:
        for k in range(20, 101, 20):
            timp_preprocesare, HQPB, media, proiectii = timpPreprocesare(k)
            for norma in ('1', '2', 'inf', 'cos'):
                recunoasteri, timpTotal = statistici(path, nr_poze, media, HQPB, proiectii, norma)
                rr = recunoasteri / (40 * (10 - nr_poze))
                tmi = timpTotal / (40 * (10 - nr_poze))
                f.write(f"{rr:.8f} ")
                f1.write(f"{tmi:.8f} ")
                print('k= ', k, ' norma=', norma)
            f.write("\n")
            f1.write("\n")
            f2.write(f"{timp_preprocesare:.8f}\n")
    print("Datele au fost salvate")

def read_data_from_file(filename):
    data = []
    with open(filename, "r") as file:
        for line in file:
            values = [float(x) for x in line.split()]
            data.append(values)
    return data

def create_plots():
    rr_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_RR.txt"
    tmi_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_TMI.txt"
    tpp_filename = f"C:\\Users\\win\\OneDrive\\Desktop\\GUI\\Lanczos\\Lanczos_{nr_poze}_TPP.txt"
    rr_data = read_data_from_file(rr_filename)
    tmi_data = read_data_from_file(tmi_filename)
    tpp_data = read_data_from_file(tpp_filename)

    norme = ['1', '2', 'inf', 'cos']
    valori_k = list(range(20, 101, 20))

    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Statistici Lanczos- {nr_poze} poze de antrenament")

    # plot RR
    plt.subplot(1, 3, 1)
    for idx, k in enumerate(valori_k):
        plt.plot(norme, rr_data[idx], marker='o', linestyle='-', label=f"k={k}")
    plt.title("RR (Rata recognitie)")
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1)
    plt.ylabel("RR (%)")
    plt.xlabel("Norme")
    plt.legend()

    # plot TMI
    plt.subplot(1, 3, 2)
    for idx, k in enumerate(valori_k):
        plt.plot(norme, tmi_data[idx], marker='o', linestyle='-', label=f"k={k}")
    plt.title("TMI (Timp mediu interogare)")
    plt.ylabel("TMI (s)")
    plt.xlabel("Norme")
    plt.legend()

     # plot TPP
    plt.subplot(1, 3, 3)
    for idx, k in enumerate(valori_k):
        plt.plot(valori_k ,tpp_data, marker='o', linestyle='-')
    plt.title("TPP (Timp preprocesare)")
    plt.ylabel("TPP (s)")
    plt.xticks(np.arange(20, 101, 20))
    plt.xlabel("k")
    plt.legend()

    plt.tight_layout()
    plt.show()

def vizualizeaza_fantome(HQPB, numar_fantome=40):
    plt.figure(figsize=(15, 5))
    for i in range(min(numar_fantome, HQPB.shape[1])):
        plt.subplot(1, numar_fantome, i + 1)
        plt.imshow(HQPB[:, i].reshape(112, 92), cmap='gray')
        plt.title(f"Fantoma {i + 1}")
        plt.axis('off')
    plt.show()

def searchImage(path, nr_poze, k, norma, A):
    HQPB, media, proiectii = lanczos(A, k)
    poza_cautata = np.array(cv2.imread(path, 0))
    poza_cautata = poza_cautata.reshape(10304,)
    persoana_gasita = interogare(path, media, HQPB, proiectii, norma)+1
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