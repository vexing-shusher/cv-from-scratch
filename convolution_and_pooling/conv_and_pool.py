import numpy as np
import cv2

from typing import Iterable

import os

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


def binarize(img: np.ndarray, thresh: int = 128) -> np.ndarray:
    """

    :param img: np.ndarray -- grayscale image
    :param thresh: threshold value above which all pixel values are set to 255 (and to 0 below it)
    :return: np.ndarray -- the binarized image
    """
    result = np.zeros_like(img)
    result[img >= thresh] = 255
    return result.astype(np.uint8)

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
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
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


def max_pooling2d(img: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    assert len(img.shape) == 2, "Only 2D feature maps are accepted."
    h, w = img.shape

    ny = ceil_(h, stride)
    nx = ceil_(w, stride)
    size = ((ny-1)*stride+kernel_size+1, (nx-1)*stride+kernel_size+1)
    pd_img = np.full(size, 0)
    pd_img[:h, :w] = img

    # Lazy way of determining output dimensions: we will calculate them while extracting features
    num_cols = 0
    features = []
    # Iterate over every block in the padded image
    for i in np.arange(0, h, step=stride):
        num_cols += 1
        num_rows = 0
        for j in np.arange(0, w, step=stride):
            num_rows += 1
            features.append(pd_img[i:i+stride, j:j+stride].max())

    return np.asarray(features).reshape((num_cols, num_rows)).astype(np.uint8)


def main(paths: Iterable):
    for path in paths:
        img = load_image(path)  # loading image using the custom function
        gs_img = grayscale(img)  # converting image to grayscale using the custom function

        edge_detection_filter = np.asarray(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        ).astype(np.float32)

        # applying custom convolution operation using the edge detection filter defined above
        feature_map = conv2d_grayscale(gs_img, edge_detection_filter)

        # applying custom pooling operation with kernel size 2 and stride 2
        pooled_feature_map = max_pooling2d(feature_map, 2, 2)

        # applying custom binarization function to the pooled feature map with threshold of 128
        binarized_pooled_feature_map = binarize(pooled_feature_map, 128)

        # saving outputs
        if not os.path.exists("./results/"):
            os.mkdir("./results/")

        img_name, img_ext = path.split("/")[-1].split('.')

        cv2.imwrite(f"./results/{img_name}_gs.{img_ext}", gs_img)
        cv2.imwrite(f"./results/{img_name}_gs_conv.{img_ext}", feature_map)
        cv2.imwrite(f"./results/{img_name}_gs_conv_pool.{img_ext}", pooled_feature_map)
        cv2.imwrite(f"./results/{img_name}_gs_conv_pool_bin.{img_ext}", binarized_pooled_feature_map)

    print("Finished processing images! The processed images can be found in the results folder.")
    return None


if __name__ == "__main__":
    image_paths = ("liberty.png", "car.png")

    main(image_paths)
