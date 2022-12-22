import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#utility functions

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

def sHalf(T, sigma):
    temp = -np.log(T) * 2 * (sigma ** 2)
    return np.round(np.sqrt(temp))

def MaskGeneration(T, sigma):
    shalf = sHalf(T, sigma)
    y, x = np.meshgrid(range(-int(shalf), int(shalf) + 1), range(-int(shalf), int(shalf) + 1))
    return x, y

def Gaussian(x,y, sigma):
    temp = ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return (np.exp(-temp))

def calculate_gradient_X(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((x * np.exp(-temp)) / sigma ** 2)

def calculate_gradient_Y(x,y, sigma):
    temp = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    return -((y * np.exp(-temp)) / sigma ** 2)

def Create_Gx(fx, fy, sigma):
    gx = calculate_gradient_X(fx, fy, sigma)
    gx = (gx * 255)
    return np.around(gx)

def Create_Gy(fx, fy, sigma):    
    gy = calculate_gradient_Y(fx, fy, sigma)
    gy = (gy * 255)
    return np.around(gy)

def pad(img, kernel):
    r, c = img.shape
    kr, kc = kernel.shape
    padded = np.zeros((r + kr,c + kc), dtype=img.dtype)
    insert = int((kr)/2)
    padded[insert: insert + r, insert: insert + c] = img
    return padded

def ApplyMask(image, kernel):
    i, j = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))    
    output = np.zeros_like(image)           
    image_padded = pad(image, kernel)
    for x in range(image.shape[0]):    
        for y in range(image.shape[1]):
            output[x, y] = (kernel * image_padded[x:x+i, y:y+j]).sum()        
    return output

#main functions

def get_gaussian_filter(T, sigma):
    x, y = MaskGeneration(T, sigma)
    return Gaussian(x, y, sigma)

def gaussian_smoothing(img, kernel=None):
    if kernel is None:
        mask = np.array([[1,1,1],[1,1,1],[1,1,1]])
    else:
        mask = kernel
    i, j = mask.shape
    output = np.zeros((img.shape[0], img.shape[1]))           
    image_padded = pad(img, mask)
    for x in range(img.shape[0]):    
        for y in range(img.shape[1]):
            output[x, y] = (mask * image_padded[x:x+i, y:y+j]).sum() / mask.sum()  
    return output

def get_gradients(T, sigma):
    x, y = MaskGeneration(T, sigma)
    return -Create_Gx(x,y, sigma), -Create_Gy(x,y, sigma)

def gradient_magnitude(fx, fy):
    mag = np.zeros_like(fx)
    mag = np.sqrt((fx ** 2) + (fy ** 2))
    mag = mag * 100 / mag.max()
    return np.around(mag)

def gradient_direction(fx, fy):
    g_dir = np.zeros_like(fx)
    g_dir = np.rad2deg(np.arctan2(fy, fx)) + 180
    return g_dir

def quantize_direction(g_dir):
    q_dir = np.zeros_like(g_dir)
    #group conditions
    c11 = np.logical_and(g_dir >= 22.5, g_dir <= 67.5)
    c12 = np.logical_and(g_dir >= 202.5, g_dir <= 247.5)
    group1 = np.logical_or(c11, c12)
    
    c21 = np.logical_and(g_dir >= 67.5, g_dir <= 122.5)
    c22 = np.logical_and(g_dir >= 247.5, g_dir <= 292.5)
    group2 = np.logical_or(c21, c22)
    
    c31 = np.logical_and(g_dir >= 112.5, g_dir <= 157.5)
    c32 = np.logical_and(g_dir >= 292.5, g_dir <= 337.5)
    group3 = np.logical_or(c31, c32)
    
    q_dir[group1] = 1
    q_dir[group2] = 2
    q_dir[group3] = 3
    return q_dir

def NMS(qdir, mag, gdir):
    M = np.zeros_like(qdir)
    a, b = np.shape(qdir)
    for i in range(a-1):
        for j in range(b-1):
            if qdir[i,j] == 0:
                if  mag[i,j-1]< mag[i,j] or mag[i,j] > mag[i,j+1]:
                    M[i,j] = gdir[i,j]
                else:
                    M[i,j] = 0
            if qdir[i,j]==1:
                if  mag[i-1,j+1]<= mag[i,j] or mag[i,j] >= mag[i+1,j-1]:
                    M[i,j] = gdir[i,j]
                else:
                    M[i,j] = 0       
            if qdir[i,j] == 2:
                if  mag[i-1,j]<= mag[i,j] or mag[i,j] >= mag[i+1,j]:
                    M[i,j] = gdir[i,j]
                else:
                    M[i,j] = 0
            if qdir[i,j] == 3:
                if  mag[i-1,j-1]<= mag[i,j] or mag[i,j] >= mag[i+1,j+1]:
                    M[i,j] = gdir[i,j]
                else:
                    M[i,j] = 0
    return M


def double_thresholding(g_sup, low_threshold, high_threshold):
    g_thr = np.zeros_like(g_sup)
    g_thr[np.logical_and(g_sup >= low_threshold, g_sup<high_threshold)] = 128
    g_thr[g_sup>high_threshold] = 255
    
    return g_thr

def hysteresis(g_thr):
    g_str = np.zeros_like(g_thr)
    for i in range(0, g_thr.shape[0]):
        for j in range(0, g_thr.shape[1]):
            check = g_thr[i-1:i+1, j-1:j+1]
            check[check <= 128] = 0
            if g_thr[i,j] == 128 and check.sum() > 0:
                g_str = 255
            elif g_thr[i,j] == 255:
                g_str[i,j] = 255
    return g_str

def canny_edge_detection(img, sigma, T, t1, t2):
    
    gauss_filter = get_gaussian_filter(T, sigma)
    
    gx, gy = get_gradients(T, sigma)
    
    gauss_img = gaussian_smoothing(img, gauss_filter)
    
    fx = ApplyMask(gauss_img, gx)
    fy = ApplyMask(gauss_img, gy)
    
    gmag = gradient_magnitude(fx, fy)
    gdir = gradient_direction(fx, fy)
    qdir = quantize_direction(gdir)
    
    nms_img = NMS(qdir, gdir, gmag)
    
    thr_img = double_thresholding(nms_img, t1, t2)
    
    hys_img = hysteresis(thr_img)
    
    return hys_img, gauss_img

def hough_transform(g_str, h_theta, h_rho, t3):
    
    assert len(g_str.shape) == 2, "Input image must be 2-dimensional."
    
    #maximal rho == image diagonal
    rho_max = np.linalg.norm(g_str.shape)
    
    rho_d = np.arange(0, rho_max + 1, h_rho)
    theta_d = np.deg2rad(np.arange(0, 360, h_theta))
    
    H = np.zeros((len(rho_d), len(theta_d)), dtype=np.uint8)
    y_idxs, x_idxs = np.nonzero(g_str) # find all edge pixel indices

    for x, y in zip(x_idxs, y_idxs):
        for n, theta in enumerate(theta_d):
            rho = int(x * np.cos(theta) + y * np.sin(theta))

            try:
                H[int(rho), n] += 1
            except IndexError:
                print("Warning: not all rho values are acceptable. Decrease rho step size.")
    
    #print(np.unique(H))
    # find local maxima
    loc_max = np.zeros_like(H)
    loc_max[H > t3] = 1
    
    # determine local maxima indices
    rho_idxs, theta_idxs = np.nonzero(loc_max)
    
    return rho_d[rho_idxs], theta_d[theta_idxs]
    
    
              
def draw_lines(img, rhos, thetas):
    
        for rho, theta in zip(rhos, thetas):
        # reverse engineer lines from rhos and thetas
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            # these are then scaled so that the lines go off the edges of the image
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
        return img
    
    
def main():
    
    root_dir = "./"

    configs = [
        [{"sigma":0.5,"T":0.3,"t1":10,"t2":30},{"h_theta":2,"h_rho":1,"t3":60}],
        [{"sigma":0.5,"T":0.3,"t1":10,"t2":30},{"h_theta":2,"h_rho":1,"t3":50}],
        [{"sigma":0.5,"T":0.3,"t1":10,"t2":20},{"h_theta":2,"h_rho":1,"t3":60}],
        [{"sigma":0.5,"T":0.3,"t1":10,"t2":30},{"h_theta":2,"h_rho":1,"t3":50}],   
    ]
    
    
    for path in os.listdir(f"{root_dir}/test_images/"):
        
        n = int(path.split('.')[0]) - 1

        print(f"Processing image {path}...")
        img = load_image(f"{root_dir}/test_images/{path}")
        gs_img = grayscale(img)

        hys_img, gauss_img = canny_edge_detection(gs_img, **configs[n][0])
        rhos, thetas = hough_transform(hys_img, **configs[n][1])
        lin_img = draw_lines(img, rhos, thetas)

        if not os.path.exists(f"{root_dir}/results/"):
            os.mkdir(f"{root_dir}/results/")
            
        cv2.imwrite(f"{root_dir}/results/gauss_{n}.jpg", gauss_img)
        cv2.imwrite(f"{root_dir}/results/edges_{n}.jpg", hys_img)
        cv2.imwrite(f"{root_dir}/results/hough_{n}.jpg", cv2.cvtColor(lin_img, cv2.COLOR_BGR2RGB)) 

if __name__ == "__main__":
    main()
