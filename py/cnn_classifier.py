import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.convolutional import *
from keras import models
from keras import layers
from keras.models import load_model
from keras.regularizers import l2
from keras.layers import BatchNormalization, Dropout, AveragePooling2D, GlobalAvgPool2D, MaxPooling2D
from keras.applications import inception_v3
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l1, l2
from numpy.random import seed
seed(111)
from tensorflow import set_random_seed
set_random_seed(1112)


def split_images():
    # images directory path for train, test, and validation.
    train_destination = '/Users/basselhaidar/Desktop/Final Project/train_dir'
    test_destination = '/Users/basselhaidar/Desktop/Final Project/test_dir'
    valid_destination = '/Users/basselhaidar/Desktop/Final Project/valid_dir'

    # get all the data in the directory split/test, and reshape them
    test_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            test_destination, 
            target_size=(112, 200),
            batch_size = 800, 
            seed = 1212) 

    train_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            train_destination, 
            target_size=(112, 200), 
            batch_size = 4000, 
            seed = 1212) 

    valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            valid_destination, 
            target_size=(112, 200),
            batch_size = 800,
            seed = 1212) 

    #split images and labels for train, test, and validation
    images_train, labels_train = next(train_data)

    images_test, labels_test = next(train_data)

    images_val, labels_val = next(valid_data)
    
    return images_train, labels_train, images_test, labels_test, images_val, labels_val

def set_cnn_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', dilation_rate=(2, 2), kernel_regularizer=l2(0.01), input_shape=(112, 200,  3)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    # cnn.add(Dropout(0.50)) # Added to see if it helps in reducing overfitting
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(32, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn



def fit_cnn_model(cnn, loss_param='categorical_crossentropy', epoch=100, batch_num=32):
    
    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
    
    optimize = SGD(lr=1e-2, momentum=0.9, decay=1e-2/epoch) # Added fo optimizing LR
    cnn.compile(loss=loss_param,
                    optimizer=optimize,
                    metrics=['accuracy'])
    
    

    cnn_fit = cnn.fit(images_train,
                        labels_train,
                        epochs=epoch,
                        batch_size=batch_num,
                        validation_data=(images_val, labels_val))
    
    cnn.save('/Users/basselhaidar/Desktop/Final Project/saved_models/CNN_LR_Decay.h5')
    
    return cnn, images_test, labels_test

def set_cnn_bn_l2_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', dilation_rate=(2, 2), kernel_regularizer=l2(0.01),input_shape=(112, 200, 3)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
#     cnn.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(1)))
#     cnn.add(layers.BatchNormalization())
#     cnn.add(layers.MaxPooling2D((2, 2))) 
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(32, activation='relu', dilation_rate=(2, 2),kernel_regularizer=l2(0.01)))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_bn_l2_model(cnn, batch_num=32, loss_param='categorical_crossentropy', epoch=20):
  
    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
   
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5) 
    
    
    sgd = SGD(lr=0.1, momentum=0.99, decay = 0.01, nesterov=True)
    cnn.compile(loss=loss_param,
                    optimizer=sgd,
                    metrics=['accuracy'])

    cnn_fit = cnn.fit(images_train,
                        labels_train,
                        epochs=epoch,
                        batch_size=batch_num,
                        validation_data=(images_val, labels_val),
                        callbacks=[rlrop])
    
    cnn.save('/Users/basselhaidar/Desktop/Final Project/saved_models/CNN_Run_1.h5')
    
    return cnn, images_test, labels_test


def imagenet_classifier(epoch=10, batch=100):
    imagenet=inception_v3.InceptionV3(weights='imagenet',include_top=False)
    imagenet_new =imagenet.output
    new_imagenet = models.Sequential()
    new_imagenet.add(imagenet)
    new_imagenet.add(GlobalAvgPool2D())
    new_imagenet.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.03)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.03))) 
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(512, activation='relu', kernel_regularizer=l2(0.03))) 
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(7,activation='softmax')) #final layer with softmax activation
 
    # Freeze layers - no training
    for layer in new_imagenet.layers[:1]:
        layer.trainable=False
    
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5) 
    sgd = SGD(lr=0.01, momentum=0.9, decay = 0.01, nesterov=True)
    
    new_imagenet.compile(optimizer=sgd,loss='categorical_crossentropy', metrics=['accuracy'])

    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
    new_imagenet.fit(images_train,
            labels_train,
            epochs=epoch,
            batch_size=batch,
            validation_data=(images_val, labels_val),
            callbacks=[rlrop])
    
    new_imagenet.save('/Users/basselhaidar/Desktop/Final Project/saved_models/imagenet_1.h5')
    
    return new_imagenet, images_test, labels_test