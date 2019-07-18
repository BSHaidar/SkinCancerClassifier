import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def create_img_dict():

    data_dir = '../skin-cancer-mnist-ham10000/'
    # Create dataframe and profile raw data
    raw_metadata_df = pd.read_csv(data_dir + 'HAM10000_metadata.csv')
    # Save the directory path of HAM10000_images_part_1 and HAM10000_images_part_2
    image_part1_dir = '../skin-cancer-mnist-ham10000/HAM10000_images_part_1/'
    image_part2_dir = '../skin-cancer-mnist-ham10000/HAM10000_images_part_2/'
    # Save the path of the images into their respective lists
    image_part_1_list = [image_part1_dir + image_path for image_path in os.listdir(image_part1_dir)]
    image_part_2_list = [image_part2_dir + image_path for image_path in os.listdir(image_part2_dir)]
    # Merge both lists
    all_image_list = list(image_part_1_list + image_part_2_list)
    print('Total number of images = {}'.format(len(all_image_list)))
    # Create a dictionnary key : 'name of image' and value : 'file path of image' 
    image_dict = {os.path.splitext(os.path.basename(image_path))[0]: image_path for image_path in all_image_list}
    len(image_dict) # both the list and dict display the correct total number of images
    # Check path of one image in the dictionnary
    print('Check path of ISIC_0025030.jpg = {}'.format(image_dict['ISIC_0025030']))
    
    return raw_metadata_df, image_dict

def set_df():
    # Create dictionary with the diagnostic categories of pigmented lesions
    lesion_cat_dict = {
        
        'akiec': 'Actinic keratoses',
        'bcc': 'Basal cell carcinoma',
        'bkl': 'Benign keratosis-like lesions ',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic nevi',
        'vasc': 'Vascular lesions'
    }

    # Make a copy of the dataframe as we will be adding new columns
    df, img_dict = create_img_dict()


    # Create new column file_path and use the image_id as the key of image_dict and map 
    # its corresponding value to get the path for the image
    df['file_path'] = df['image_id'].map(img_dict.get)

    # Create new column category_name and use dx as the key to lesion_cat_dict and map 
    # it to its corresponding value to get the lesion name
    df['category_name'] = df['dx'].map(lesion_cat_dict.get)

    # Create new column category_id and assign the integer codes 
    # of the category_name that was transformed into a pandas categorical 
    df['category_id'] = pd.Categorical(df['category_name']).codes

    # fill age null values by the mean age
    df.age.fillna(df.age.mean(), inplace=True)
    return df

def sns_countplot(df, col_name):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 6)
    sns.countplot(x=col_name, data=df, ax=ax);
    plt.show();
'''
# fill age null values by the mean age
metadata_df.age.fillna(metadata_df.age.mean(), inplace=True)

fig, ax1 = plt.subplots(1, 1, figsize= (10, 5))
metadata_df['category_name'].value_counts().plot(kind='bar', ax=ax1);



metadata_df.category_name.value_counts()

'''