#BIBLIOTECAS
import numpy as np
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
from glob import glob

#IMPORTANDO IMAGENS
paths=pd.DataFrame(glob('*.tiff'))
paths.columns=['location']

#FILTRO MEDIANA E REMOÇÃO SALTO DE FASE
for x in range(len(paths)):
    img=paths['location'].iloc[x]
    im = cv.imread(img, 0)
    
    imcos=np.cos(np.array(im,'float32')*2*np.pi/255)
    imsen=np.sin(np.array(im,'float32')*2*np.pi/255)
    icos=cv.medianBlur(imcos,5)
    isen=cv.medianBlur(imsen,5)
    phase=np.arctan2(icos,isen)
    # plt.imshow(phase, cmap='gray')
    
    image_unwrapped = unwrap_phase(phase)
    ImgRem=cv.GaussianBlur(image_unwrapped,(25,25),0)
    # plt.imshow(ImgRem, cmap='gray')
    
    norm = cv.normalize(ImgRem, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F) #normalizar entre 0 e 255
    # norm = norm.astype(np.uint8)
    
    # norm[0:500,:]=np.mean(norm)
    
    #DUPLO THRESHOLD
    ret,thresh1 = cv.threshold(norm,np.mean(norm)*0.8,255,cv.THRESH_BINARY) #threshold lado direito
    # plt.imshow(thresh1, cmap='gray')
    norm=norm*-1
    norm=norm+255
    
    ret,thresh2 = cv.threshold(norm,np.mean(norm)*0.8,255,cv.THRESH_BINARY) #threshold lado esquerdo
    # plt.imshow(thresh2, cmap='gray')
    
    thresh=thresh1+thresh2 #somatório ambos lados
    
    kernel = np.ones((5,5),np.uint8) #kernel 
    
    #OPERAÇÃO ABERTURA
    erosion = cv.erode(thresh,kernel,iterations = 20) #erosão
    dilation = cv.dilate(erosion,kernel,iterations = 20) #dilatação
    
    dilation=255-dilation
    
    plt.figure(x)
    plt.imshow(dilation, cmap='gray')
    # plt.imsave(paths['location'].iloc[x]+'.png', dilation, cmap='gray')


# histogram, bin_edges = np.histogram(norm, bins=256, range=(0, 255))
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("pixel count")
# plt.xlim([0.0, 255.0])

# plt.plot(bin_edges[0:-1], histogram)