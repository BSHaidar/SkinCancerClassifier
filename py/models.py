# %load_ext autoreload
# %autoreload 1
import warnings
warnings.filterwarnings('ignore')
import sys
# sys.path.insert(0, '../SkinCancerClassifier/py')
import pandas as pd
import numpy as np
from numpy.random import seed
seed(111)
import os
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow
import seaborn as sns
from PIL import Image

import keras
from keras import models
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.layers import Activation, Reshape
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.convolutional import *
from keras.layers import BatchNormalization, Dropout, AveragePooling2D, GlobalAvgPool2D, MaxPooling2D, Dense, Flatten
from keras.applications import inception_v3, DenseNet121
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint
from keras.regularizers import l2
from keras.metrics import categorical_crossentropy
from keras.layers.core import Dense, Flatten


from tensorflow import set_random_seed
set_random_seed(1112)

from set_img import *

def set_cnn_model(input_shape=(90, 120, 3)):
    '''A very simple CNN model with one convolution layer
        and one dense layer just to get a baseline'''
    cnn = models.Sequential()
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', dilation_rate=(2, 2), kernel_regularizer=l2(0.02), input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(32, activation='relu'))
    cnn.add(Dropout(0.3))
    cnn.add(BatchNormalization())
    cnn.add(Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_model(cnn, epoch=20, batch_num=32, reduce_by=0.5, start_epoch=0):
    '''Fit the simple CNN model'''
   
    # images_train, labels_train, images_test, labels_test, images_val, labels_val = split_images()
    test_data, train_data, valid_data = split_images()
    optimize = SGD(lr=1e-1, momentum=0.9) 
    
    cnn.compile(loss='categorical_crossentropy',
                optimizer=optimize,
                metrics=['accuracy'])
    
    rlrop = ReduceLROnPlateau(
                                monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=0.0001, 
                                min_lr=0.000001
                            )
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/First_CNN_111.h5', verbose=1)
    
    cnn.fit_generator(train_data,
                       steps_per_epoch= 200 * reduce_by,
                       # np.ceil(train_data.samples/train_data.batch_size),
                       epochs=start_epoch, 
                       verbose=1,
                       callbacks=[checkpoint, rlrop], 
                       validation_data=valid_data, 
                       validation_steps=32,
                       workers=16, 
                       use_multiprocessing=True, 
                       shuffle=True
                       )
    return test_data

def set_cnn_bn_l2_model(input_shape=(90, 120, 3)):
    cnn = models.Sequential()
    cnn.add(Conv2D(128, (1, 1), activation='relu', kernel_initializer='he_uniform', 
            dilation_rate=(2, 2), kernel_regularizer=l2(0.02),input_shape=input_shape))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.02)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2))) 
    cnn.add(Flatten())
    cnn.add(layers.Dense(32, activation='relu', kernel_regularizer=l2(0.02)))
    cnn.add(Dropout(0.5))
    cnn.add(BatchNormalization())
    cnn.add(Dense(7, activation='softmax'))

    print(cnn.summary())
    
    return cnn

def fit_cnn_bn_l2_model(cnn, batch_num=32, loss_param='categorical_crossentropy', epoch=40, reduce_by=0.5):
      
    callbacks_list = []
    test_data, train_data, valid_data = split_images()
   
    rlrop = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5) 
    
    
    sgd = SGD(lr=0.1, momentum=0.95, nesterov=False)
    
    cnn.compile(loss=loss_param,
                optimizer=sgd,
                metrics=['accuracy'])
    
#     lrate = LearningRateScheduler(step_decay)
#     callbacks_list = [lrate]
    checkpointer = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/cnn_bnn_balanced_2.h5', verbose=1)
    cnn.fit_generator(train_data,
                       steps_per_epoch= 200 * reduce_by,
#                       np.ceil(train_data.samples/train_data.batch_size),
                       epochs=epoch, 
                       verbose=1,
                       callbacks=[checkpointer], 
                       validation_data=valid_data, 
                       validation_steps=np.ceil(valid_data.samples/valid_data.batch_size), 
                       workers=16, 
                       use_multiprocessing=True, 
                       shuffle=True)

# test_img=te_labels.argmax(axis=1)

def plot_confusion_matrix(te_label, y_pred):
    # Calculate Confusion Matrix
    cm = confusion_matrix(te_label, y_pred)
    df = set_df()
    # Figure adjustment and heatmap plot
    f = plt.figure(figsize=(10,10))
    ax= plt.subplot()
    labels = df.groupby('category_id').category_name.first().values
    sns.heatmap(cm, annot=True, ax = ax, vmax=100, cbar=False, cmap='Paired', mask=(cm==0), fmt=',.0f', linewidths=2, linecolor='grey', ); 

    # labels
    ax.set_xlabel('Predicted labels', fontsize=16);
    ax.set_ylabel('True labels', labelpad=30, fontsize=16); 
    ax.set_title('Confusion Matrix', fontsize=18); 
    ax.xaxis.set_ticklabels(labels, rotation=90); 
    ax.yaxis.set_ticklabels(labels, rotation=0);
    ax.set_facecolor('white')
    
    report = classification_report(te_label, y_pred, target_names=df.groupby('category_id').category_name.first().values)
    print(report)
    
# learning rate schedule
def step_decay(epoch=5):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def dense_model(epoch, num_classes=7, inp_shape=(90, 120, 3), reduce_by=0.5):
    
    base_model = DenseNet121(weights=None, include_top=False, input_shape=inp_shape)
    x = AveragePooling2D(pool_size=(2,2), name='avg_pool')(base_model.output) 
    BatchNormalization()
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    BatchNormalization()
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)
    
    
    # let's visualize layer names and layer indices to see how many layers we should freeze:
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    
    for layer in model.layers[:427]:
        layer.trainable = False
    
    for layer in model.layers[427:]:
        layer.trainable = True
    
    rlrop = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10) 
    sgd = SGD(lr=0.1, momentum=0.9, nesterov=False)
    
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    test_data, train_data, valid_data = split_images()
    
    model.fit_generator(generator=train_data,
                    validation_data=valid_data,
                    validation_steps = 32,
                    steps_per_epoch = 200 * reduce_by,
                    callbacks=[rlrop],
                    epochs=epoch)
    
    model.save('../SkinCancerClassifier/saved_models/densenet121_101.h5')
    
    return test_data, model

def imagenet_classifier(inp_shape=(90, 120, 3)):
    imagenet=inception_v3.InceptionV3(weights='imagenet',include_top=False, input_shape=inp_shape)
    imagenet_new =imagenet.output
    new_imagenet = models.Sequential()
    new_imagenet.add(imagenet)
    new_imagenet.add(GlobalAvgPool2D())
    new_imagenet.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(512, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(128, activation='relu', kernel_regularizer=l2(0.02)))
    new_imagenet.add(BatchNormalization())
    new_imagenet.add(Dense(7,activation='softmax')) #final layer with softmax activation
    
    return new_imagenet
 
def train_imagenet_classifier(new_imagenet, epoch=30, reduce_by=0.5):   
        
    # Print all layers in imagenet
    for i, layer in enumerate(new_imagenet.layers):
        print(i, layer.name)
    
    # Freeze layer - no training
    for layer in new_imagenet.layers[:1]:
        layer.trainable=False
    
    rlrop = ReduceLROnPlateau(
                                monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=1e-4, 
                                min_lr=1e-6
                            )
    
   # Use Stochastic Gradient Descent with set learnig rate and momentum
    sgd = SGD(lr=0.1, momentum=0.9)
    
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/imagenet_1157.h5', 
                                   verbose=1)
    
    new_imagenet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_imagenet.summary())
    
    test_data, train_data, valid_data = split_images()
    
    new_imagenet.fit_generator(generator=train_data,
                    validation_data=valid_data,
                    validation_steps = 32,
                    steps_per_epoch = 200 * reduce_by, 
                    callbacks=[rlrop, checkpoint],
                    epochs=epoch)
    

    return test_data, new_imagenet

def set_cnn_no_dense_model(input_shape = (90, 120, 3), num_classes = 7):
    ''' Created a model with only convolutional layers and no dense layers'''

    model = Sequential()

    model.add(Conv2D(32,kernel_size=(3, 3),activation='relu',name="Conv1", input_shape=input_shape)) 
    model.add(BatchNormalization(name="Norm1"))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',name="Conv2")) 
    model.add(BatchNormalization(name="Norm2"))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu',name="Conv3")) 
    model.add(BatchNormalization(name="Norm3"))
    model.add(MaxPooling2D(pool_size = (2, 2))) 
    model.add(Dropout(0.20))

    model.add(Conv2D(64, (3, 3), activation='relu',name="Conv4")) 
    model.add(BatchNormalization(name="Norm4"))
    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv5")) 
    model.add(BatchNormalization(name="Norm5"))
    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv6")) 
    model.add(BatchNormalization(name="Norm6"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(128, (3, 3), activation='relu',name="Conv7")) 
    model.add(BatchNormalization(name="Norm7"))
    model.add(Conv2D(256, (3, 3), activation='relu',name="Conv8")) 
    model.add(BatchNormalization(name="Norm8"))
    model.add(Conv2D(256, (3, 3), activation='relu',name="Conv9")) 
    model.add(BatchNormalization(name="Norm9"))
    model.add(MaxPooling2D(pool_size=(2, 2))) #6,9
    model.add(Dropout(0.20))

    model.add(Conv2D(7,(1,1),name="conv10",activation="relu")) 
    model.add(BatchNormalization(name="Norm10"))
    model.add(Conv2D(7,kernel_size=(6,9),name="Conv11"))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    return model

def fit_cnn_no_dense_model(model, start_at_epoch,  epoch, reduce_by=0.5):   
    # Compile the model
    model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics=["accuracy"])
    
    # Set learning rate plateau 
    rlrop = ReduceLROnPlateau(monitor='val_loss', 
                                patience=4, 
                                verbose=1, 
                                factor=1e-4, 
                                min_lr=1e-6)
    
    checkpoint = ModelCheckpoint(filepath='../SkinCancerClassifier/saved_models/cnn_nodense_model3.h5', 
                                   verbose=1)
    
    test_data, train_data, valid_data = split_images()
    
    model.fit_generator(generator=train_data,
                        validation_data=valid_data,
                        validation_steps = 32,
                        steps_per_epoch = 200 * reduce_by,
                        callbacks=[rlrop, checkpoint],
                        initial_epoch=start_at_epoch,
                        epochs=epoch)
    
    return test_data, model
