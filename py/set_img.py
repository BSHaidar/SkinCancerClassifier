import pandas as pd
import numpy as np
from numpy.random import seed
import os
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow
import seaborn as sns
from PIL import Image
from numpy.random import seed

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def img_name(dir_name):
    ''' 
    Get images filepath and store them in a list
    Required Parameter
        dir_name: Directory path
    Return
        None
    '''
    img_list = []
    for i in range(0, 7):
        sub_dir= dir_name + str(i) +'/'
        for sub in os.listdir(sub_dir):
            img_list.append(sub)
    return img_list

def create_img_dict(data_dir = '../SkinCancerClassifier/',
                    image_test_dir = '../SkinCancerClassifier/test_dir/', 
                    image_train_dir = '../SkinCancerClassifier/train_dir/', 
                    image_val_dir = '../SkinCancerClassifier/valid_dir/'):
    ''' 
    Creates raw dataframe and image dictionnary {key:image name, value:image path}
    
    Optional Parameters
        data_dir: Root directory for all files
        image_test_dir, image_train_dir, image_val_dir: 
        Directories for images in train, validation, and test
    Return
        Dataframe and image dictionnary
    '''
    
    # data_dir = '../SkinCancerClassifier/'
    
    # Create dataframe of raw data
    raw_metadata_df = pd.read_csv(data_dir + 'HAM10000_metadata.csv')
    
    # Directories for images in train, validation, and test
    # image_test_dir = '../SkinCancerClassifier/test_dir/'
    # image_train_dir = '../SkinCancerClassifier/train_dir/'
    # image_val_dir = '../SkinCancerClassifier/valid_dir/'
    
    # Create a combined list images in all directories with their full file/image path
    img_test_list = img_name(image_test_dir)
    img_test_list = [image_test_dir + img_name for img_name in img_test_list]
    img_train_list = img_name(image_train_dir)
    img_train_list = [image_train_dir + img_name for img_name in img_train_list]
    img_val_list = img_name(image_val_dir)
    img_val_list = [image_val_dir + img_name for img_name in img_val_list]
    all_img_list = img_test_list + img_train_list + img_val_list
    
    # Create a dictionnary that has the name of the image as its key and the image path as its value
    image_dict = {os.path.splitext(os.path.basename(img_name))[0]: img_name for img_name in all_img_list}
    
    return raw_metadata_df, image_dict

def set_df():
    '''
    This function creates 3 new columns for image file path (file_path), name of skin
    lesion (category_name), a number mapped to its name (category_id). For age column 
    with null value, it fills it with the mean age.
    
    Return
        Dataframe 
    '''
    
    # Create dictionary with the diagnostic categories of pigmented lesions
    lesion_cat_dict = {
        
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    # Get raw dataframe and the image dictionnary
    df, img_dict = create_img_dict()


    # Create new column file_path and use the image_id as the key of image_dict and map 
    # its corresponding value to get the path for the image
    df['file_path'] = df['image_id'].map(img_dict.get)

    # Create new column category_name and use dx as the key to lesion_cat_dict and map 
    # it to its corresponding value to get the lesion name
    df['category_name'] = df['dx'].map(lesion_cat_dict.get)

    # Create new column category_id and assign the integer codes 
    # of the category_name that were transformed into pandas categorical datatype
    df['category_id'] = pd.Categorical(df['category_name']).codes

    # Fill age null values by the mean age
    df.age.fillna(df.age.mean(), inplace=True)
    
    return df

def split_images(train_dir='../SkinCancerClassifier/train_dir/', 
                 test_dir='../SkinCancerClassifier/test_dir/', 
                 val_dir='../SkinCancerClassifier/valid_dir/', 
                 target_size=(90, 120)):
    


    # Rescale test, train, and validation images. Resize them to 90 x 120 pixels
    test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, 
                                                                        target_size=target_size,
                                                                        batch_size = 2000, 
                                                                        seed = 1212) 
   
    # Apply various transformations to the training data
    # to help the model generalize better on the test data
    train_data = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    rescale=1./255).flow_from_directory(train_dir, 
                                                                        target_size=target_size, 
                                                                        batch_size = 147, 
                                                                        seed = 1212) 

    valid_data = ImageDataGenerator(rescale=1./255).flow_from_directory( 
            val_dir, 
            target_size=target_size,
            batch_size = 300,
            seed = 1212) 


    return test_data, train_data, valid_data
