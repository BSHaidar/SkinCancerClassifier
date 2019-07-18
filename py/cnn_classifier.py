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
from keras.layers import BatchNormalization, Dropout, AveragePooling2D, GlobalAvgPool2D
from keras.applications import inception_v3
from sklearn.metrics import confusion_matrix, f1_score
from keras.models import Model
from numpy.random import seed
seed(111)
from tensorflow import set_random_seed
set_random_seed(1112)


def split_images():
    # images directory path for train, test, and validation.
    train_destination = '../train_dir'
    test_destination = '../test_dir'
    valid_destination = '../valid_dir'

    # get all the data in the directory split/test, and reshape them
    test_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            test_destination, 
            target_size=(450, 600), 
            seed = 1212) 

    train_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            train_destination, 
            target_size=(450, 600), 
            seed = 1212) 

    valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            valid_destination, 
            target_size=(450, 600), 
            seed = 1212) 

    #split images and labels for train, test, and validation
    images_train, labels_train = next(train_data)

    images_test, labels_test = next(train_data)

    images_val, labels_val = next(valid_data)
    
    return images_train, labels_train, images_test, labels_test, images_val, labels_val

def set_cnn_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(450, 600,  3), padding='SAME'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(32, (3, 3), activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2))) 
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(32, activation='relu'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn



def fit_cnn_model(cnn, loss_param='categorical_crossentropy', optimize='sgd', epoch=24, batch=150):
    
    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
    cnn.compile(loss=loss_param,
                    optimizer=optimize,
                    metrics=['acc'])

    cnn_fit = cnn.fit(images_train,
                        labels_train,
                        epochs=epoch,
                        batch_size=batch,
                        validation_data=(images_val, labels_val))
    
    cnn.save('CNN_Run_1.h5')
    
    return images_test, labels_test

def set_cnn_dropout_model():
    cnn = models.Sequential()
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(450, 600,  3), padding='SAME'))
    cnn.add(Dropout(0.2))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(32, (3, 3), activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.MaxPooling2D((2, 2))) 
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(32, activation='relu'))
    cnn.add(Dropout(0.2))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_model_dropout(cnn, loss_param='mean_squared_error', epoch=20, batch=2150):
    # # images directory path for train, test, and validation.
    # train_destination = '../train_dir'
    # test_destination = '../test_dir'
    # valid_destination = '../valid_dir'

    # # get all the data in the directory split/test, and reshape them
    # test_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
    #         test_destination, 
    #         target_size=(450, 600), 
    #         seed = 1212) 

    # train_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
    #         train_destination, 
    #         target_size=(450, 600), 
    #         seed = 1212) 

    # valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
    #         valid_destination, 
    #         target_size=(450, 600), 
    #         seed = 1212) 

    # #split images and labels for train, test, and validation
    # images_train, labels_train = next(train_data)

    # images_test, labels_test = next(train_data)

    # images_val, labels_val = next(valid_data)
    
    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
   
    sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    cnn.compile(loss=loss_param,
                    optimizer=sgd,
                    metrics=['acc'])

    cnn_fit = cnn.fit(images_train,
                        labels_train,
                        epochs=epoch,
                        batch_size=batch,
                        validation_data=(images_val, labels_val))
    
    cnn.save('CNN_Run_1.h5')
    
    return images_test, labels_test


def imagenet_classifier(epoch=20, batch=150):
    imagenet=inception_v3.InceptionV3(weights='imagenet',include_top=False)
    imagenet_new =imagenet.output
    new_imagenet = models.Sequential()
    new_imagenet.add(imagenet)
    new_imagenet.add(GlobalAvgPool2D())
    # new_imagenet.add(Dropout(0.2))
    new_imagenet.add(layers.BatchNormalization())
    new_imagenet.add(Dense(1024,activation='relu'))
    # new_imagenet.add(Dropout(0.2))
    new_imagenet.add(layers.BatchNormalization())
    new_imagenet.add(Dense(1024,activation='relu')) #dense layer 2
    # new_imagenet.add(Dropout(0.2))
    new_imagenet.add(layers.BatchNormalization())
    new_imagenet.add(Dense(512,activation='relu')) #dense layer 3
    # new_imagenet.add(Dropout(0.2))
    new_imagenet.add(layers.BatchNormalization())
    new_imagenet.add(Dense(7,activation='softmax')) #final layer with softmax activation
    
    # Freeze layers - no training
    for layer in new_imagenet.layers[:1]:
        layer.trainable=False
    
    new_imagenet.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

    images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
    new_imagenet.fit(images_train,
            labels_train,
            epochs=epoch,
            batch_size=batch,
            validation_data=(images_val, labels_val))
    
    new_imagenet.save('imagenet_1.h5')
    
    return images_test, labels_test