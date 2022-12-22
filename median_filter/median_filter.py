import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_image(path: str) -> np.ndarray:
    """
    This function loads an image using OpenCV-python built-in method and converts it into RGB from BGR.
    :param path:  str -- full or relative path to the image file
    :return: np.ndarray -- RGB image (channels last)
    """
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def grayscale(img: np.ndarray, weights: tuple = (0.2989, 0.5870, 0.1140)) -> np.ndarray:
    """
    This function replicates OpenCV RGB2GRAY algorithm.
    :param img: np.ndarray -- RGB image (channels last)
    :param weights: tuple -- weighting for the grayscale conversion (defaults to CCIR 601)
    :return: np.ndarray -- grayscale-converted image
    """
    assert img.shape[-1] == 3, "Images with the number of channels not equal to 3 are not supported."
    return np.dot(img, weights).astype(np.uint8)


def median_(a: np.ndarray) -> np.ndarray:
    """
    :param a: numpy array
    :return: median value
    """
    f_a = a.flatten()  # flattened array
    idx1 = int(np.floor((len(f_a)-1)/2))
    idx2 = int(np.ceil((len(f_a)-1)/2))
    s_a = np.sort(f_a)  # sorted array
    return np.mean((s_a[idx1], s_a[idx2])).astype(a.dtype)


def ceil_(x: int, y: int) -> int:
    """
    A function for obtaining the number of strides along an image dimension.
    :param x: int -- image dimension
    :param y: int -- stride value
    :return: int -- number of strides plus 1
    """
    return x//y + 1


def relu_(x: float) -> float:
    """
    Simple Rectified Linear Unit implementation. Supports only single values as inputs.
    :param x: float -- input value
    :return: 0 if x < 0, else x
    """
    return max(0, x)


def conv2d_grayscale(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Simple convolution function. It implements "same" padding scheme (and therefore doesn't support any stride
    values except for 1). It also does not support multichannel images (however, it can be called for each channel
    separately). ReLU activation function is applied after each convolution step.
    :param img: np.ndarray -- grayscale image (2D array)
    :param kernel: np.ndarray -- filter with odd dimensions (2D array)
    :return: np.ndarray -- resulting feature map
    """
    assert len(img.shape) == 2, "This function works only with grayscale or binary images (single channel)."
    assert len(kernel.shape) == 2, "Kernel must be two-dimensional."
    assert kernel.shape[0] * kernel.shape[1] // 2 != 0, "Filter dimensions must be odd."

    # Placeholder for the convolution output
    output = np.zeros_like(img)
    # Add zero padding to the input image
    pd_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    pd_img[1:-1, 1:-1] = img
    # Loop over every pixel of the image
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = relu_((kernel * pd_img[y: y + 3, x: x + 3]).sum())
    return output.astype(np.uint8)


def apply_median_filter(img: np.ndarray) -> np.ndarray:

    """
    :param img: 2D np.ndarray (grayscale image)
    :return: grayscale denoised image
    """

    assert len(img.shape) == 2, "This function works only with grayscale or binary images (single channel)."

    # Placeholder for the convolution output
    output = np.zeros_like(img)
    # Add zero padding to the input image
    pd_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    pd_img[1:-1, 1:-1] = img
    # Loop over every pixel of the image
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = relu_(median_(pd_img[y: y + 3, x: x + 3]))
    return output.astype(np.uint8)


def calculate_histogram(img: np.ndarray) -> tuple:
    """
    :param img: 2D numpy array (grayscale image)
    :return: (x: list, y: list) -- a tuple containing possible pixel values and their amounts in the image
    """

    hist_data = ([], [])
    for pixel_value in np.arange(0,256,1):
        hist_data[0].append(pixel_value)
        hist_data[1].append(img[img == pixel_value].size)

    return hist_data


def main(path: str) -> None:
    img = load_image(path)  # loading image using the custom function
    gs_img = grayscale(img)  # converting image to grayscale using the custom function

    mean_filter = np.asarray([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]).astype(np.float32)

    mean_filter /= mean_filter.size

    # apply mean filter to the grayscale image
    mean_img = conv2d_grayscale(gs_img, mean_filter)

    # apply median filter to the grayscale image
    median_img = apply_median_filter(gs_img)

    if not os.path.exists("./results/"):
        os.mkdir("./results/")

    _, img_ext = path.split("/")[-1].split('.')

    cv2.imwrite(f"./results/output1.{img_ext}", mean_img)
    cv2.imwrite(f"./results/output2.{img_ext}", median_img)

    hist_names = ["noise_image_his.png", "output1_his.png", "output2_his.png"]

    # calculate histograms
    for item, name in zip([gs_img, mean_img, median_img], hist_names):
        x, y = calculate_histogram(item)
        fig, ax = plt.subplots()
        ax.bar(x, y, align='center')
        fig.savefig(f"./results/{name}")
        del fig
        del ax


if __name__ == "__main__":
    image_path = "./noise_image.png"
    main(image_path)

