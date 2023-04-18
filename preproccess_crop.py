import os
import random
import cv2
import numpy as np

def count_images(input_folder):
    image_count = 0
    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.lower().endswith(".jpg"):
                image_count += 1
    return image_count

def random_crop(image, crop_height, crop_width):
    height, width = image.shape[:2]
    if width < crop_width or height < crop_height:
        return None

    left = random.randint(0, width - crop_width)
    upper = random.randint(0, height - crop_height)
    right = left + crop_width
    lower = upper + crop_height

    return image[upper:lower, left:right]

def crop_with_highest_variance(image, crop_height, crop_width, num_crops):
    max_variance = 0
    best_crop = None

    for _ in range(num_crops):
        cropped_image = random_crop(image, crop_height, crop_width)
        if cropped_image is not None:
            crop_array = np.array(cropped_image)
            variance = np.var(crop_array)

            if variance > max_variance:
                max_variance = variance
                best_crop = cropped_image

    return best_crop

def preprocess_image(image, crop_height, crop_width, num_crops, grayscale=True, equalize_hist=True, edge_detection=True, normalize=True):
    # Convert to grayscale
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    if equalize_hist:
        image = cv2.equalizeHist(image)

    # Apply Sobel filter for edge detection
    if edge_detection:
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        image = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize pixel values to [0, 1]
    if normalize:
        image = image / np.max(image)

    # Select the crop with the highest variance
    cropped_image = crop_with_highest_variance(image, crop_height, crop_width, num_crops)

    return cropped_image

def preprocess_images(input_folder, output_folder, crop_height, crop_width, num_crops, grayscale=True, equalize_hist=True, edge_detection=True, normalize=True):
    output_folder = f"{output_folder}{crop_height}{'x'}{crop_width}_{'grayscale' if grayscale else 'no_grayscale'}_{'equalize_hist' if equalize_hist else 'no_equalize_hist'}_{'edge_detection' if edge_detection else 'no_edge_detection'}_{'normalize' if normalize else 'no_normalize'}"
    total_images = count_images(input_folder)
    processed_images = 0

    for root, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)
                preprocessed_image = preprocess_image(image, crop_height, crop_width, num_crops, grayscale, equalize_hist, edge_detection, normalize)

                if preprocessed_image is not None:
                    rel_dir = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, rel_dir)

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    output_path = os.path.join(output_dir, filename)
                    cv2.imwrite(output_path, preprocessed_image * 255)

                processed_images += 1
                print(f"Processing image {processed_images} of {total_images}")


input_folder = "/Users/yurirykhlo/dev/ECE460JFinalPrject/datasets/"
output_folder = "/Users/yurirykhlo/dev/ECE460JFinalPrject/preproccessed_datasets/"

#images smaller than crop_height x crop_width would be omitted 
crop_height = 128
crop_width = 128
num_crops = 10 #how many crops would be sampled for highest variance 

# Set the desired preprocessing flags
grayscale = True
equalize_hist = True
edge_detection = True
normalize = True

preprocess_images(input_folder, output_folder, crop_height, crop_width, num_crops, grayscale, equalize_hist, edge_detection, normalize)
