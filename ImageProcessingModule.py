import numpy as np


def gaussian_noise(image, mean, std):
    """
    Adds Gaussian noise to an image
    :param image:
    :param mean:
    :param std:
    :return: noisy image
    """
    #take numpy array representing image and apply the mean and std of the normal distribution
    noise = np.random.normal(mean, std, size=image.shape)
    #add noise to the image
    noisy_image = image + noise
    #clip values to the range to prevent loss
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    #return noisy image
    return noisy_image


# all functions rely on numpy arrays to function
# applies gaussian function to 2d list of coordinates
# Images are assumed to be greyscale (256,256)
def gaussian_kernel(size, sigma):
    """
    Uses the Gaussian Function and applies it to the 2d numpy array
    :param size:
    :param sigma:
    :return:
    """
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    #uses same exponential function as Gaussian Function
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()



def gaussian_blur(image, kernel_size, sigma):
    """
    Applies Gaussian Blur for the purpose of smoothing or blurring an image
    Dependencies:
    gaussian_kernel
    :param image expressed as numpy array:
    :param kernel_size:
    :param sigma:
    :return image with Gaussian blur applied expressed as numpy array:
    """
    #Set window size internally to prevent error from the window size and kernel size not matching
    window_size = 1 + (2 * kernel_size)
    #generate gaussian kernel with kernel size and sigma
    kernel = gaussian_kernel(kernel_size, sigma)
    #create a new numpy array with the shape of the image
    blurred_image = np.zeros_like(image)
    #padd the image so the kernel can be applied to the edges of the image
    padded_image = np.pad(image, ((kernel_size, kernel_size), (kernel_size, kernel_size)), mode='reflect')
    #nested for loop for each pixel selecting a window of pixels and multiplies it with the Gaussian Kernel
    #Sums it to get the corresponding pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            blurred_image[i, j] = np.sum(window * kernel)
    return blurred_image.astype(np.uint8)


def convolution(image, kernel):
    """
    Performes 2D Convolution of an Image with a given kernel
    :param image:
    :param kernel:
    :return convoluted image:
    """
    #Get the height and width from the image
    height, width = image.shape
    #get the height and width of the kernel
    kernel_height, kernel_width = kernel.shape
    #pad the 0's with half the distance of the kernel size
    padded_image = np.pad(image, pad_width=((kernel_height // 2, kernel_height // 2),
                                            (kernel_width // 2, kernel_width // 2)), mode='constant')
    output_image = np.zeros_like(image)
    #for each pixel in the input image multiply the kernel with the window and get the sum of the product
    #then assign to the output image
    for i in range(height):
        for j in range(width):
            output_image[i, j] = np.sum(kernel * padded_image[i:i + kernel_height, j:j + kernel_width])
    return output_image




def median_filter(image, kernel_size):
    """
    Applies median filtering to image expressed as numpy array
    :param image:
    :param kernel_size:
    :param sigma:
    :return filtered image:
    """
    # define dimensions of height and width of numpy array using .shape from numpy
    # create padded image by padding the image with 0's
    height, width = image.shape
    pad_width = kernel_size // 2
    #pad the image with 0's the value of half the kernel width
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    #create a kernel using the numpy ones function defined by the kernel size
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    #for each pixel in the image the patch is extracted using slicing
    for i in range(height):
        for j in range(width):
            patch = padded_image[i:i + kernel_size, j:j + kernel_size]
            #calculate the median with the numpy median function
            filtered_image[i, j] = np.median(patch)
    return filtered_image


# applies sobel operator to numpy array
# applies sobel operator to numpy array
def sobel_edge_detection(image):
    """
    applies sobel_edge_detection to an image (DOES IT WORK THOUGH, NO)
    :param image: a numpy array representing an image
    :return: a numpy array with sobel edge detection applied
    """
    # Sobel operator kernels
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Calculate image gradients
    gradient_x = convolution(image, sobel_x)
    gradient_y = convolution(image, sobel_y)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    # Convert gradient magnitude to uint8
    output_image = np.uint8(gradient_magnitude)

    return output_image


# applies the kirsch operator to a numpy array

def kirsch_edge_detection(image):
    """
    Applies kirsch edge detection to an image
    Dependencies: convolution
    :param image:
    :return image with kirsch_edge_detection applied:
    """
    # Kirsch operator kernels
    kernels = np.array([
        [[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]],
        [[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]],
        [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]],
        [[5, 5, -3], [5, 0, -3], [-3, -3, -3]],
        [[5, -3, -3], [5, 0, -3], [5, -3, -3]],
        [[-3, -3, -3], [5, 0, -3], [5, 5, -3]],
        [[-3, -3, -3], [-3, 0, -3], [5, 5, 5]],
        [[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]
    ])

    # Apply convolution for each kernel and combine the results
    results = []
    for kernel in kernels:
        results.append(convolution(image, kernel))
    output_image = np.max(np.stack(results), axis=0)
    #Normalize around 0 and 255
    output_image = output_image / np.max(output_image) * 255
    output_image = np.uint8(output_image)

    return output_image

def roberts_operator(image):
    """ Applies Robert's Operator on a given image
    Dependencies: convolution
    only takes parameter image
     """
    # Create Robert's Operator kernels
    #45 degree angle
    kernel_1 = np.array([[1, 0], [0, -1]])
    kernel_2 = np.array([[0, 1], [-1, 0]])
    #135 degree angle


    # Apply kernels to image
    filtered_image_1 = convolution(image, kernel_1)
    filtered_image_2 = convolution(image, kernel_2)

    # Combine results
    output_image = np.sqrt(np.square(filtered_image_1) + np.square(filtered_image_2))

    # Normalize output
    output_image = (output_image / np.max(output_image)) * 255

    # Convert to uint8
    output_image = np.uint8(output_image)

    return output_image
