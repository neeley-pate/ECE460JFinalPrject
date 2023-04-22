import os
import numpy as np
import random
import csv
from keras.utils import load_img
from keras.utils import img_to_array

DATASET_PATH = '../datasets/'
DIGITAL_IMAGES_PATH = 'digital/StableDiffusion/'
REAL_IMAGES_PATH = 'real-world/'
PAINTING_IMAGES_PATH = 'Non-digital Artwork/'

SEED = 69420

def sampleImages(digital_image_samples, real_world_samples, artwork_samples, dimension=(128, 128)):
    digital_images_files = getImageFiles('digital')
    real_world_images_files = getImageFiles('real')
    artwork_images_files = getImageFiles('painting')

    random.seed(SEED)

    digital_sampled_images = random.sample(list(digital_images_files), digital_image_samples)
    real_sampled_images = random.sample(list(real_world_images_files), real_world_samples)
    artwork_sampled_images = random.sample(list(artwork_images_files), artwork_samples)

    # concatenate all samples
    sampled_images = np.concatenate((digital_sampled_images, real_sampled_images, artwork_sampled_images), axis=0)

    images = np.empty((0, dimension[0], dimension[1], 3))
    for image in sampled_images:
        img = load_img(DATASET_PATH + image, target_size=dimension)
        data = img_to_array(img)
        data = np.expand_dims(data, axis=0)
        images = np.append(images, data, axis=0)

    # create labels
    labels = []
    for i in range(digital_image_samples):
        labels.append([1, 0, 0])
    for i in range(real_world_samples):
        labels.append([0, 0, 1])
    for i in range(artwork_samples):
        labels.append([0, 1, 0])
    # labels = np.concatenate((digital_labels, real_labels, artwork_labels), axis=0)
    x = 0
    return images, np.asarray(labels)
    
def getImageFiles(image_class):
    if image_class == 'digital':
        files = os.listdir(DATASET_PATH + DIGITAL_IMAGES_PATH)
        return [DIGITAL_IMAGES_PATH + file for file in files]
    
    elif image_class == 'real':
        dirs = os.listdir(DATASET_PATH + REAL_IMAGES_PATH)
        real_images_files = np.array([])
        for dir in dirs:
            images = os.listdir(DATASET_PATH + REAL_IMAGES_PATH + dir)
            images = [REAL_IMAGES_PATH + dir + '/' + image for image in images]
            real_images_files = np.append(real_images_files, images)
        return real_images_files
    
    elif image_class == 'painting':
        painting_images_dirs = os.listdir(DATASET_PATH + PAINTING_IMAGES_PATH)
        painting_images_files = np.array([])
        for dir in painting_images_dirs:
            images = os.listdir(DATASET_PATH + PAINTING_IMAGES_PATH + dir)
            images = [PAINTING_IMAGES_PATH + dir + '/' + image for image in images]
            painting_images_files = np.append(painting_images_files, images)
        return painting_images_files
    
    else:
        raise ValueError('image_class must be one of digital, real, or painting')

def loadImages(image_class):
    pass


def extract_features_from_images(image_class, offset, num_samples, dimension=(224, 224)):
    image_files = getImageFiles(image_class)
    
    # filename = f'features/{image_class}{offset}.csv'
    with open('features/' + image_class + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(offset, offset + num_samples):
            if(i == len(image_files)):
                print('Done')
                break
            print(i)
            img = load_img("../datasets/" + image_files[i], target_size=dimension)
            data = img_to_array(img)
            flat_data = data.flatten()
            writer.writerow(flat_data)