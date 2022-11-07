#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!conda install --yes tensorflow-datasets
#!conda install --yes matplotlib
#!conda install --yes seaborn
#!conda install --yes scipy
#!conda install -c conda-forge opencv
#!conda install --yes glob
#!conda install --yes scikit-image
#!pip install opencv-python
#!pip install imutils
#!pip install kaggle --upgrade
#!pip install tqdm


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance as dist
import os
import cv2
import glob as gb
from skimage.filters import gaussian
from skimage.morphology import dilation,erosion
from skimage.feature import canny
from skimage.measure import find_contours
import imutils
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

print("pass all package")


# In[3]:


os.environ['KAGGLE_USERNAME'] = 'hongyielsuh'
os.environ['KAGGLE_KEY'] = '6cc8b1fd26df2be653b31439180c58d7'
get_ipython().system('kaggle -h')


# In[4]:


def findedges(image):
    gray = cv2.GaussianBlur(image, (1, 1), 0)
    edged = cv2.Canny(gray, 100, 400)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


# In[5]:


def getimageconturs(edged):
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    return contours


# In[6]:


def getboxes(contours,orig):
    boxes = []
    centers = []
    for contour in contours:
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        if (dist.euclidean(tl, bl)) > 0 and (dist.euclidean(tl, tr)) > 0:
            boxes.append(box)
    return boxes


# In[7]:


#!kaggle datasets download -d paultimothymooney/blood-cells


# In[8]:


get_ipython().system('ls')


# In[9]:


#!unzip '*.zip'
#!pwd


# In[10]:


image_size=(120,120)
code={"EOSINOPHIL":0,"LYMPHOCYTE":1,"MONOCYTE":2,"NEUTROPHIL":3}
def getcode(n):
    if type(n)==str:
        for x,y in code.items():
            if n==x:
                return y 
    else:
        for x,y in code.items():
            if n==y:
                return x


# In[26]:


get_ipython().system('pwd')
import os
import time
import glob as gb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance as dist
import os
import cv2
import glob as gb
from skimage.filters import gaussian
from skimage.morphology import dilation,erosion
from skimage.feature import canny
from skimage.measure import find_contours
import imutils
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tqdm import tqdm
def getimageconturs(edged):
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    return contours

def findedges(image):
    gray = cv2.GaussianBlur(image, (1, 1), 0)
    edged = cv2.Canny(gray, 100, 400)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def getboxes(contours,orig):
    boxes = []
    centers = []
    for contour in contours:
        box = cv2.minAreaRect(contour)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        (tl, tr, br, bl) = box
        if (dist.euclidean(tl, bl)) > 0 and (dist.euclidean(tl, tr)) > 0:
            boxes.append(box)
    return boxes
image_size=(120,120)
code={"EOSINOPHIL":0,"LYMPHOCYTE":1,"MONOCYTE":2,"NEUTROPHIL":3}
def getcode(n):
    if type(n)==str:
        for x,y in code.items():
            if n==x:
                return y 
    else:
        for x,y in code.items():
            if n==y:
                return x
current_directory = os.getcwd()
def loaddata():
    datasets=[current_directory + "/dataset2-master/dataset2-master/images/TRAIN/",
          current_directory + "/dataset2-master/dataset2-master/images/TEST/",]
    
    images=[]
    labels=[]
    count=0
    
    for dataset in datasets:        
        for folder in os.listdir(dataset):
            

            files=gb.glob(pathname=str(dataset+folder+"/*.jpeg"))
            label=getcode(folder)
            for file in tqdm(files):
                time.sleep(0.1)
                
                
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # add padding to the image to better detect cell at the edge
                image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[198, 203, 208])

                #thresholding the image to get the target cell
                image1 = cv2.inRange(image,(80, 80, 180),(180, 170, 245))
                
                
                
                
                
                
                # openning errosion then dilation
                kernel = np.ones((3, 3), np.uint8)
                kernel1 = np.ones((5, 5), np.uint8)
                img_erosion = cv2.erode(image1, kernel, iterations=2)
                image1 = cv2.dilate(img_erosion, kernel1, iterations=5)
                
                
                

                
                #detecting the blood cell
                edgedImage = findedges(image1)
                edgedContours = getimageconturs(edgedImage)
                edgedBoxes =  getboxes(edgedContours, image.copy())
                if len(edgedBoxes)==0:
                    count +=1
                    continue
                # get the large box and get its cordinate
                last = edgedBoxes[-1]
                max_x = int(max(last[:,0]))
                min_x = int( min(last[:,0]))
                max_y = int(max(last[:,1]))
                min_y = int(min(last[:,1]))
                
                
                

                # draw the contour and fill it 
                mask = np.zeros_like(image)
                cv2.drawContours(mask, edgedContours, len(edgedContours)-1, (255,255,255), -1) 
                
                # any pixel but the pixels inside the contour is zero
                image[mask==0] = 0
                
                # extract th blood cell
                image = image[min_y:max_y, min_x:max_x]

                if (np.size(image)==0):
                    count +=1
                    continue
                # resize th image
                image = cv2.resize(image, image_size)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)
                
                break
        print(images)
                
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')
        
    return images,labels

print(loaddata())


# In[11]:


images,labels=loaddata()


# In[13]:


images,labels=shuffle(images,labels,random_state=10)


# In[14]:


images=images/255
train_image,test_image,train_label,test_label=train_test_split(images,labels,test_size=.2)
test_image,val_image,test_label,val_label=train_test_split(test_image,test_label,test_size=.5)


# In[15]:


def displayrandomimage(image,label,typeofimage):
    plt.figure(figsize=(15,15))
    plt.suptitle("some random image of "+typeofimage,fontsize=17)
    for n,i in  enumerate(list(np.random.randint(0,len(image),36))):
        plt.subplot(6,6,n+1)
        plt.imshow(image[i])
        plt.axis("off")
        plt.title(getcode(label[i]))


# In[16]:


displayrandomimage(train_image,train_label,"train image")


# In[17]:


displayrandomimage(test_image,test_label,"test image")


# In[18]:


displayrandomimage(val_image,val_label,"val image")


# In[19]:


import keras
from keras import models
from tensorflow.keras.applications import MobileNetV2

# mobilenetv2
transfer_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    classifier_activation="softmax",
    input_shape=(120,120,3)
)

transfer_model.trainable=False

model_tr = keras.models.Sequential([
    transfer_model, 
    keras.layers.Flatten(),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    
    keras.layers.Dense(4, activation='softmax')
])
model_tr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_tr.summary()

history=model_tr.fit(train_image,
                  train_label,
                  epochs=30,
                  batch_size=32,
                  validation_data=(val_image,val_label))
# End
loss, accuracy = model_tr.evaluate(test_image, test_label)
print("the acc of test image is : ", accuracy)


# In[20]:


def plot_acc_and_loss_of_train_and_val(history):
    #plt.figure(figsize=(15,15))
    #plt.suptitle("acc,loss of train VS acc,loss of val")
    epochs=[i for i in range(30)]
    train_acc=history.history['accuracy']
    train_loss=history.history['loss']
    val_acc=history.history['val_accuracy']
    val_loss=history.history['val_loss']
    fig , ax=plt.subplots(1,2)
    fig.set_size_inches(20,10)
    ax[0].plot(epochs,train_acc,'go-',label='training accuracy')
    ax[0].plot(epochs,val_acc,'ro-',label='validation accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    ax[1].plot(epochs,train_loss,'g-o',label='training loss')
    ax[1].plot(epochs,val_loss,'r-o',label='validation loss')
    ax[1].set_title('Training & Validation loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Training & Validation Loss")

    
plot_acc_and_loss_of_train_and_val(history)


# In[ ]:




