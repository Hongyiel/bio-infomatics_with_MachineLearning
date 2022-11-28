import keras
from keras import models
from tensorflow.keras.applications import VGG19
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import MaxPool2D

# VGG 16
# transfer_model = VGG19(weights='imagenet', include_top=False, input_shape=(120, 120, 3))
# transfer_model.trainable=False
# model_tr = keras.models.Sequential([
#     transfer_model, 
#     keras.layers.Flatten(),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(32, activation='relu'), 
#     keras.layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(16, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(8, activation='relu'),
#     keras.layers.Dropout(0.2),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dense(4, activation='softmax')
# ])

# model = Sequential()

# VGG16
from tensorflow.keras import layers, models
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Model, Sequential

model = Sequential()
kernel_initializer = keras.initializers.he_normal(seed=None)
# conv 1
model.add(Conv2D(filters=64,     
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_size = (3,3), strides = 1, padding="same", activation="relu", input_shape=(120, 120, 3)))
model.add(Conv2D(filters=64, 
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_size = (3,3), strides = 1, padding="same", activation="relu"))

# MaxPooling 1
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))

# Conv2-1, 2-2
model.add(Conv2D(filters=128,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_size = (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size = (3,3), strides = 1, padding="same",activation = "relu"))

#MaxPooling 2
model.add(MaxPooling2D(pool_size=(2,2), strides = 2))

#Conv3-1, 3-2, 3-3
model.add(Conv2D(filters=256, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))

#MaxPooling 3
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

#Conv4-1, 4-2, 4-3
model.add(Conv2D(filters=512, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size = (3,3), strides = 1, padding="same", activation="relu"))

#MaxPooling 4
model.add(MaxPooling2D(pool_size= (2, 2), strides = 2))

#Conv5-1, 5-2, 5-3
model.add(Conv2D(filters=512, kernel_size= (3,3), strides = 1, padding="same", activation ="relu"))
model.add(Conv2D(filters=512, kernel_size= (3,3), strides = 1, padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size= (3,3), strides = 1, padding="same", activation="relu"))

#MaxPooling 5
model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))

model.add(Flatten())

# FC 6 7 8 
# model.add(Dense(4096, activation="relu"))
# model.add(Dense(4096, activation="relu"))
# model.add(Dense(1000, activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(Dense(32, activation='relu', kernel_initializer = kernel_initializer, bias_initializer="zeros"))
model.add(Dropout(0.2))

model.add(keras.layers.BatchNormalization())
model.add(Dense(16, activation='relu', kernel_initializer = kernel_initializer, bias_initializer="zeros"))
model.add(Dropout(0.2))

model.add(keras.layers.BatchNormalization())
model.add(Dense(8, activation='relu', kernel_initializer = kernel_initializer, bias_initializer="zeros"))
model.add(Dropout(0.2))

model.add(keras.layers.BatchNormalization())
model.add(Dense(4, activation='softmax'))

# compile model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history=model.fit(train_image,
                  train_label,
                  epochs=20,
                  batch_size=32,
                  validation_data=(val_image,val_label))
# # End VGG 16
loss,acuracy=model.evaluate(test_image,test_label)
print("the accuracy of test image is : ",acuracy)

def plot_acc_and_loss_of_train_and_val(history):
    #plt.figure(figsize=(15,15))
    #plt.suptitle("acc,loss of train VS acc,loss of val")
    epochs=[i for i in range(20)]
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


# # from keras.models import load_model

# # model_tr.save('result_vgg90ab.h5')