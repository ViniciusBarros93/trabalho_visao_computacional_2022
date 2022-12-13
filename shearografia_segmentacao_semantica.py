#BIBLIOTECAS
import pandas as pd
from glob import glob
from simple_unet_model import simple_unet_model   #Use normal unet model
from keras.utils import normalize
import cv2
import numpy as np
from matplotlib import pyplot as plt

#IMPORTANDO IMAGENS
paths=pd.DataFrame(glob('*.tiff'))
paths.columns=['location']

image_dataset = []

for x in range(len(paths)):
    img = paths['location'].iloc[x]
    im = cv2.imread(img, 0)
    image_dataset.append(im)
    
paths2=pd.DataFrame(glob('*.png'))
paths2.columns=['location']

mask_dataset = []

for x in range(len(paths2)):
    img2 = paths2['location'].iloc[x]
    im2 = cv2.imread(img2, 0)
    mask_dataset.append(im2)


#NORMALIZANDO IMAGENS
image_dataset = np.expand_dims(normalize(np.array(image_dataset), axis=1),3)
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

#DIVIDINDO CONJUNTOS DE TREINO E TESTE
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#DEFININDO PARAMETROS DE REDE
# ###############################################################
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

#CHAMANDO O MODELO DE UNET
def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()

#TREINAMENTO
history = model.fit(X_train, y_train, 
                    batch_size = 30, 
                    verbose=1, 
                    epochs=70, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('shearograph_segmentation2.hdf5')

############################################################

#AVALIANDO O MODELO

##################################
#IOU
y_pred=model.predict(X_test)
y_pred_thresholded = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#SALVAR MODELO
model = get_model()
model.load_weights('shearograph_segmentation.hdf5')

#AVALIANDO RESULTADOS DO CONJUNTO DE TESTE
# for x in range(5):
#     plt.figure(x)
#     plt.imshow(y_pred[x], cmap='gray')

#plt.imsave('unet_1.png', y_pred[:,:,0], cmap='gray')










