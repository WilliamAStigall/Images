
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import ImageProcessingModule as ipm
#If the noisy folder does not exist we need to run the DatasetConstructor
folder_name = "noisy_folder"
if not os.path.exists(folder_name):
    print("dataset not present, please run DatasetConstructor.py")
    sys.exit(0)
#check for gaussian_blur folder and make it if it does not exist
blurred_folder = "gaussian_blur"
if not os.path.exists(blurred_folder):
    os.makedirs(blurred_folder)
# comparison folder for mathplotlib
comparison_folder = "comparison_folder"
if not os.path.exists(comparison_folder):
    os.makedirs(comparison_folder)
# define kernel size and sigma
kernel_size = 3
sigma = 1
# Iterate over all images in dataset
for filename in os.listdir(folder_name):
    # Load image
    input_path = os.path.join(folder_name, filename)
    #use PIL to open image
    image = Image.open(input_path)
    image_array = np.asarray(image)
    kernel = ipm.gaussian_kernel(kernel_size, sigma)
    # Apply Gaussian blur calling function from ipm
    blurred_image = ipm.gaussian_blur(image_array, kernel_size, sigma)
    # Save blurred image
    output_path = os.path.join(blurred_folder, filename)
    Image.fromarray(blurred_image).save(output_path)
#Repeat previous for median filter
median_filter_folder = "median_filter"

if not os.path.exists(median_filter_folder):
    os.makedirs(median_filter_folder)
for filename in os.listdir(folder_name):
    input_path = os.path.join(folder_name, filename)
    image = Image.open(input_path)
    image_array = np.asarray(image)
    median_filtered_image = ipm.median_filter(image_array, kernel_size)
    output_path = os.path.join(median_filter_folder, filename)
    median_filtered_image = Image.fromarray(median_filtered_image)
    median_filtered_image.save(output_path)
#sobel edge detection applied to gaussian blurred images
gaussian_sobel = "gaussian_sobel"
if not os.path.exists(gaussian_sobel):
    os.makedirs(gaussian_sobel)
for filename in os.listdir(blurred_folder):
    input_path = os.path.join(blurred_folder, filename)
    image = Image.open(input_path)
    image_array = np.asarray(image)
    gaussian_sobel_image = ipm.sobel_edge_detection(image_array)
    output_path = os.path.join(gaussian_sobel, filename)
    gaussian_sobel_image = Image.fromarray(gaussian_sobel_image)
    gaussian_sobel_image.save(output_path)
# kirsch edge detection applies to median blurred images
kirsch_median_folder = "kirsch_median"
if not os.path.exists(kirsch_median_folder):
    os.makedirs(kirsch_median_folder)

for filename in os.listdir(median_filter_folder):
    input_path = os.path.join(median_filter_folder, filename)
    image = Image.open(input_path)
    image_array = np.asarray(image)
    kirsch_median_image = ipm.kirsch_edge_detection(image_array)
    output_path = os.path.join(kirsch_median_folder, filename)
    kirsch_median_image = Image.fromarray(kirsch_median_image)
    kirsch_median_image.save(output_path)
#robert's operator applied to gaussian blurred images
robert_gaussian_folder = "robert_gaussian"
if not os.path.exists(robert_gaussian_folder):
    os.makedirs(robert_gaussian_folder)
for filename in os.listdir(blurred_folder):
    input_path = os.path.join(blurred_folder, filename)
    image = Image.open(input_path)
    image_array = np.asarray(image)
    robert_gaussian_image = ipm.roberts_operator(image_array)
    output_path = os.path.join(robert_gaussian_folder, filename)
    robert_gaussian_image = Image.fromarray(robert_gaussian_image)
    robert_gaussian_image.save(output_path)

for filename in os.listdir(folder_name):
    # Load original image
    input_path = os.path.join(folder_name, filename)
    image = Image.open(input_path)
    # Load blurred image
    blurred_input_path = os.path.join(blurred_folder, filename)
    blurred_image = Image.open(blurred_input_path)
    # Load median filtered image
    median_input_path = os.path.join(median_filter_folder, filename)
    median_image = Image.open(median_input_path)
    # Load Kirsch edge detected image
    kirsch_input_path = os.path.join(kirsch_median_folder, filename)
    kirsch_image = Image.open(kirsch_input_path)
    # Load Gaussian Sobel edge detected image
    sobel_input_path = os.path.join(gaussian_sobel, filename)
    sobel_image = Image.open(sobel_input_path)
    # Load robert gaussian edge detected images
    robert_input_path = os.path.join(robert_gaussian_folder,filename)
    robert_image = Image.open(robert_input_path)

    # Plot original and processed images side by side
    #Save them into comparison folder
    fig, ax = plt.subplots(1, 6, figsize=(20, 20))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original")
    ax[1].imshow(blurred_image, cmap='gray')
    ax[1].set_title("Gaussian Blur")
    ax[2].imshow(median_image, cmap='gray')
    ax[2].set_title("Median Filter")
    ax[3].imshow(kirsch_image, cmap='gray')
    ax[3].set_title("Kirsch Edge Detection")
    ax[4].imshow(sobel_image, cmap='gray')
    ax[4].set_title("Gaussian Sobel Edge Detection")
    ax[5].imshow(robert_image, cmap='gray')
    ax[5].set_title("Gaussian Robert Edge Detection")
    fig.savefig(os.path.join(comparison_folder, filename))
    plt.show()
