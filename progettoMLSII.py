import cv2 as cv
import skimage.measure
import numpy as np
import glob
import imageio
import shutil

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from scipy import stats

#inizializzazione delle variabili

lista = []
imgs_path = []


#Carico la cartella contenente le immagini
for image_path in glob.glob("C:\\Users\\liguo\\Downloads\\PsychoFlickrImages1\\*\\*.jpg"):
    img = imageio.imread(image_path)

#Calcolo del rapporto delle dimensioni dell'immagine
#Alcune immagini hanno componente channels nulla
    if(len(img.shape)==3):
        height, width, channels = img.shape
    elif(len(img.shape)==2):
        height, width = img.shape
    aspect_ratio = width/height
        
# Estrazione delle componenti RGB e calcolo della loro distribuzione percentuale
# Alcune immagini hanno stranamente componenti RGB nulla
    try:
        red, green, blue = img[:, :, 0], img[:, :, 1], img[:,:,2]
        intensity = img.sum(axis=2)
        
        def colour_frac(color):
            return np.sum(color)/np.sum(intensity)
        
        red_fraction = colour_frac(red)
        green_fraction = colour_frac(green)
        blue_fraction = colour_frac(blue)
    except Exception:
        continue
    
# Estrazione di Hue, Saturation e Value con calcolo media e deviazione standard
    try:
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        media_h = np.mean(h)
        deviazione_h = np.std(h)
        media_s = np.mean(s)
        deviazione_s = np.std(s)
        media_v = np.mean(v)
        deviazione_v = np.std(v)
    except Exception:
        continue

#Calcolo dell'entropia complessiva dell'immagine
    entropy = skimage.measure.shannon_entropy(img)
    
#Copio i risultati in una lista di appoggio
    lista.append([aspect_ratio, red_fraction, green_fraction, blue_fraction, media_h, media_s, media_v, deviazione_h, deviazione_s, deviazione_v, entropy])
    imgs_path.append(image_path)

#Normalizzo ed elimino gli outlier
z = np.abs(stats.zscore(lista))
z = list(z)
cont = 0
for i in z:
    cbool = 0
    for y in i:
        if(y>3):
            cbool = 1
    if(cbool==0):
        cont += 1
    else:
        lista.pop(cont)
        imgs_path.pop(cont)

#Salvataggio dei valori in un CSV
with open('C:\\Users\\liguo\\Downloads\\features.csv', mode='w') as file:
    for line in lista:
        file.write(str(line))
        file.write('\n')
        
# Clustering e generazione del grafico con i risultati
data = PCA(n_components=2).fit_transform(lista)
kmeans = KMeans(n_clusters=6)
kmeans = kmeans.fit(data)
labels = kmeans.predict(data)
for label, color in zip(range(6), ['chocolate', 'indigo', 'pink', 'dimgrey','k','gold']):
    plt.scatter(data[labels == label, 0], data[labels == label, 1], c=color)
plt.show()
#individuazione dei centroidi
centroids = kmeans.cluster_centers_
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, data)


#Salvataggio delle immagini nei diversi cluster
cont = 0
for i in labels:
    if(i==0):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k1\\'+str(cont)+'.jpg')
        cont += 1
    elif(i==1):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k2\\'+str(cont)+'.jpg')
        cont += 1
    elif(i==2):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k3\\'+str(cont)+'.jpg')
        cont += 1
    elif(i==3):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k4\\'+str(cont)+'.jpg')
        cont += 1
    elif(i==4):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k5\\'+str(cont)+'.jpg')
        cont += 1
    elif(i==5):
        shutil.copy(imgs_path[cont], 'C:\\Users\\liguo\\Downloads\\outputoutlier\\k6\\'+str(cont)+'.jpg')
        cont += 1