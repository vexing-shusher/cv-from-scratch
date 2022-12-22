import numpy as np
from skimage.filters import gaussian
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from tqdm import tqdm

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

def IM(a: float, b: float, N:int) -> np.ndarray:
    """
    Helper function to perform matrix optimization of the snake curve.
    
    :a: float -- alpha parameter
    :b: float -- beta parameter
    :N: int -- the number of points sampled on the snake curve: (x(p_i), y(p_i)), i=0,...,N-1
    :return: np.ndarray -- numeric integration matrix
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A

def external_edge_force(img: np.ndarray, sigma: float = 30.) -> tuple:
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.

    :img: ndarray -- the grayscale image.
    :sigma: float -- Gaussian standard deviation.
    :return: tuple(Callable, Callable) -- external edge forces.
    """
    # Gaussian smoothing.
    smoothed = gaussian( (img-img.min()) / (img.max()-img.min()), sigma )
    
    # Gradient of the image in x and y directions.
    giy, gix = np.gradient( smoothed )
    # Gradient magnitude of the image.
    gmi = (gix**2 + giy**2)**(0.5)
    # Normalize the gradient magnitude.
    gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())

    # Gradient of gradient magnitude of the image in x and y directions.
    # Required for optimization.
    ggmiy, ggmix = np.gradient(gmi)

    def fx(x, y):
        """
        Return external edge force in the x direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmix[ (y.round().astype(int), x.round().astype(int)) ]

    def fy(x, y):
        """
        Return external edge force in the y direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """
        # Check bounds.
        x[ x < 0 ] = 0.
        y[ y < 0 ] = 0.

        x[ x > img.shape[1]-1 ] = img.shape[1]-1
        y[ y > img.shape[0]-1 ] = img.shape[0]-1

        return ggmiy[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy

def optimize(x: np.ndarray, 
             y: np.ndarray, 
             a: float, 
             b: float, 
             fx: Callable, 
             fy: Callable, 
             gamma: float = 0.1, 
             n_iters: int = 10, 
             return_all: bool = True):
    """
    :x: np.ndarray -- intial x coordinates of the snake.
    :y: np.ndarray -- initial y coordinates of the snake.
    :a: float -- alpha parameter.
    :b: float -- beta parameter.
    :fx: Callable -- partial derivative of first coordinate of external energy function.
    :fy: Callable -- see fx.
    :gamma: float -- step size of the iteration.
    :n_iters: int -- number of times to iterate the snake.
    :return_all: bool -- if True, a list of (x,y) coords are returned corresponding to each iteration.
    :return: list or tuple -- snake coordinates for each optimization step or only for the last step.
    """
    A = IM(a,b,x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append((x_.copy(),y_.copy()))
            
    if return_all:
        return snakes
    else:
        return (x,y)
    
def main():
    # create initial snake curve
    t = np.linspace(0, 2*np.pi, 400)
    x = 500 + 500*np.sin(t)
    y = 500 + 500*np.cos(t)

    # hyperparameters
    sigma = 25

    alpha = 0.001
    beta  = 0.1
    gamma = 1000
    iterations = 1000

    root_dir = "./"
    for path in np.sort(os.listdir(f"{root_dir}/test_images/")):

        img = load_image(f"{root_dir}/test_images/{path}")

        gs_img = grayscale(img)

        # fx and fy are callable functions
        fx, fy = external_edge_force(gs_img, sigma=sigma)

        # optimize the curve
        snakes = optimize(
            x = x,
            y = y,
            a = alpha,
            b = beta,
            fx = fx,
            fy = fy,
            gamma = gamma,
            n_iters = iterations,
            return_all = True # to create videos, we need all intermediate outputs
        )

        # drop every 9 frames to shorten the video
        indexing = np.arange(0, len(snakes), 10).astype(np.int32)

        video_name = path.split(".")[0]
        video = cv2.VideoWriter(f'{video_name}.mp4', 
                                cv2.VideoWriter_fourcc(*'avc1'), 
                                1, 
                                (400,400))

        for sn in tqdm(np.asarray(snakes)[indexing]):
            fig = plt.figure(figsize=(10,10))
            plt.imshow(img, cmap=cm.Greys_r)
            plt.scatter(sn[0], sn[1], color='blue')

            # put pixel buffer in numpy array
            canvas = FigureCanvas(fig)
            canvas.draw()
            mat = np.array(canvas.renderer._renderer)
            mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
            image = cv2.resize(mat, (400,400))

            # write frame to video
            video.write(image)

            plt.close()

        # close video writer
        cv2.destroyAllWindows()
        video.release()

        # save final outputs
        fig = plt.figure()
        plt.imshow(img, cmap=cm.Greys_r)
        plt.scatter(snakes[0][0], snakes[0][1], color='blue')
        plt.scatter(snakes[-1][0], snakes[-1][1], color='red')
        plt.axis('off')
        plt.savefig(f"{video_name}.jpg", bbox_inches='tight')
        
if __name__ == "__main__":
    main()
