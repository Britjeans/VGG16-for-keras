from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.applications.vgg16 import VGG16

def load_Old_VGG():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    print("Model loaded..!")
    print(base_model.summary())
    return base_model


def New_VGG_16():
    #layer1
    model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(224,224,3)))
    model.add(Conv2D(64, (3, 3), activation='relu',input_shape=(224,224,3), padding='same', name='block1_conv1'))
    # model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    # model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    # model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    # model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    # model.add(ZeroPadding2D((1,1)))
    model.add( Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add( MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu',name='fc1'))
    model.add(Dropout(0.5,name='dropout1'))
    model.add(Dense(4096, activation='relu',name='fc2'))
    model.add(Dropout(0.5,name='dropout2'))
    model.add(Dense(2, activation='softmax', name='predictions'))

    print(model.summary())
    #initialize the model
    old_model=load_Old_VGG()
    weights=[]
    j=0
    
    for i,layer in enumerate(old_model.layers):
        #skip layer1's weights
        if i==0 or i==1:
            continue
        w=layer.get_weights()
        weights.append(w)
        j=j+1
    k=0
    for i,layer in enumerate(model.layers):
        if i==0:
            continue

        layer.set_weights(weights[i-1])
        k=k+1
        if k==j:
            break


    return model

if __name__ == "__main__":
    model=New_VGG_16()
