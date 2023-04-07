from datetime import time
from selenium import webdriver
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow
import tensorflow as tf
import os
import shutil
from bs4 import BeautifulSoup
import requests
#We use PIL for image preprocessing
from PIL import Image
import io

from selenium.common import NoSuchElementException
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
"""
This File is used to construct the dataset and is separated from the rest of the program for better encapsulation
"""
"""METHODOLOGY First we import a dataset of around 100-200 unblurred images #Images will be collected via webcrawler 
from pexels.com Second Preprocess the data for use in the program creating our finalized dataset Third using 
mathplotlib and numpy we are going to apply Gaussian Blur to all the images for use in the Artificial Neural Network 
we create Fourth Define the ANN Architecture using Tensorflow with input and output layers for median filtering and 
Gaussian Filter Train the network Measure Performance Fifth Apply edge detection algorithm to the filtered image to 
extract edges Display all 3 images """

"""
Functions
"""


# applies gaussian function to 2d list of coordinates
def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_blur(image, kernel_size=5, sigma=1):
    window_size = 1+(2*kernel_size)
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = np.zeros_like(image)
    padded_image = np.pad(image, ((kernel_size, kernel_size), (kernel_size, kernel_size)), mode='reflect')

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            blurred_image[i, j] = np.sum(window * kernel)
    return blurred_image.astype(np.uint8)


def convolution(image, kernel):
    height, width = image.shape
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, pad_width=((kernel_height // 2, kernel_height // 2),
                                            (kernel_width // 2, kernel_width // 2)), mode='constant')
    output_image = np.zeros_like(image)
    for i in range(height):
        for j in range(width):
            output_image[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])
    return output_image


"""
Step 1: Acquire Dataset via webcrawler (all pictures will be landscape pictures
YOU ONLY NEED TO RUN THIS ONCE UNLESS YOU WANT TO CREATE A DIFFERENT DATATSET WITH DIFFERENT SEARCH
"""
# specify the URL of the website to crawl
url = "https://pixabay.com/"

# create a folder to store the downloaded images
folder_name = "images"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
if len(os.listdir(folder_name)) >= 200:
    print("Already downloaded 200 files, skipping web crawling")
else:

    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": os.path.abspath(folder_name)}
    options.add_experimental_option("prefs", prefs)

    # create a Chrome webdriver instance
    driver = webdriver.Chrome(options=options)

    # navigate to the website and get the page content
    driver.get(url)
    time.sleep(1)

    # search for images
    search_box = WebDriverWait(driver, 10).until(
        EC.visibility_of_element_located((By.XPATH, "//input[@name='search']"))
    )
    search_box.send_keys("landscape")
    search_box.submit()

    # wait for the search results to load
    time.sleep(5)

    # scroll down to load more images
    for i in range(3):
        driver.execute_script("window.scrollBy(0, 1000)")
        time.sleep(2)

    # keep a running count of downloaded images
    count = 0

    # download images until 200 have been downloaded
    while count < 200:
        # find all the <img> tags in the HTML content
        img_tags = driver.find_elements(By.TAG_NAME, "img")

        # download each image and save it to the directory
        for img in img_tags:
            # get the source URL of the image
            src = img.get_attribute("src")

            # check if the URL points to an image
            if src.endswith(".jpg") or src.endswith(".jpeg"):
                # generate a filename based on the count
                filename = f"landscape_{count}.jpg"

                # check if the file already exists in the directory
                if filename in os.listdir(folder_name):
                    print(f"File {filename} already exists")
                else:
                    # download the image
                    response = requests.get(src)
                    with open(os.path.join(folder_name, filename), "wb") as f:
                        f.write(response.content)

                    print(f"Downloaded file {filename}")

                    # increment the count of downloaded images
                    count += 1

                    # break out of the loop if 200 images have been downloaded
                    if count == 201:
                        break

        # find the "Next page" link and click on it
        try:
            next_page_link = driver.find_element(By.XPATH, "//a[contains(text(),'Next page')]")
            driver.execute_script("arguments[0].scrollIntoView(true);", next_page_link)
            next_page_link.click()
            time.sleep(5)
        except NoSuchElementException:
            print("No more pages to load")
            break

    # close the webdriver instance
    driver.quit()

"""
Step 2: Preprocess images gained from web crawler
This is the only time we use a Image Processing library (which is ok since this is still the dataset stage
doing this process with numpy, repeats or discards pixels resulting in a loss of quality, this may result in
our ANN being less accurate because of bad data so we use PIL
"""

input_path = ""
images_folder = folder_name

dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

for filename in os.listdir(images_folder):
    input_path = os.path.join(images_folder, filename)
    image = Image.open(input_path)
    image = image.resize((256, 256))
    image = image.convert('L')
    output_path = os.path.join(dataset_folder, filename)
    image.save(output_path)

print("Resized images saved to dataset and turned grey")
# Define output directory for blurred images
blurred_folder = "finalized_dataset"
if not os.path.exists(blurred_folder):
    os.makedirs(blurred_folder)

comparison_folder = "comparison_folder"
if not os.path.exists(comparison_folder):
    os.makedirs(comparison_folder)
#define kernel size and sigma
kernel_size = 5
sigma = 1
# Iterate over all images in dataset
for filename in os.listdir(dataset_folder):
    # Load image
    input_path = os.path.join(dataset_folder, filename)
    image = Image.open(input_path)
    image_array = np.asarray(image)
    kernel = gaussian_kernel(kernel_size , sigma)
    convolution(image_array, kernel)
    # Apply Gaussian blur
    blurred_image = gaussian_blur(image_array)

    # Save blurred image
    output_path = os.path.join(blurred_folder, filename)
    Image.fromarray(blurred_image).save(output_path)

    # Plot original and blurred image side by side
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(blurred_image, cmap="gray")
    axs[1].set_title("Blurred")
    plt.savefig(os.path.join(comparison_folder,f"{os.path.splitext(filename)[0]}_comparison.png"))
    plt.close(fig)
