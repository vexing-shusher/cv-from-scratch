# CV from scratch
A couple of assignments made by me for the Computer Vision class at National Taipei University of Technology.

All the functions necessary for running the scripts in the repository are placed inside the script files. This was a requirement for the homeworks, I am sorry :) For the same reason the code is duplicated frequently in different scripts.

The repository contains the following scripts:

1. /convolution_and_pooling/conv_and_pool.py -- implementation of a 2d convolution operation and a pooling operation for grayscale images along with two sample images for testing; 
2. /median_filter/median_filter.py -- applying median filter to a noisy image; I have lost the original image somewhere, so feel free to use your own;
3. /canny_and_hough/detect_lines.py -- this script runs the Canny edge detector on the test images, after which it applies the Hough transform and detects lines (road lines, to be precise);
4. /active_contour/active_contour.py -- an implementation of the active contour ("snakes") model; I was unable to make this one run with my own Gaussian filter, so it uses the one from scikit-image. 

# Acknowledgements
The following repositories were used as references for my scripts:
1. Canny Edge Detection from scratch: https://github.com/UsamaI000/CannyEdgeDetection-from-scratch-python
2. Hough Transform from scratch: https://gist.github.com/ri-sh/45cb32dd5c1485e273ab81468e531f09
3. Snakes from scratch: https://github.com/notmatthancock/snakes

A big shoutout to these guys! If anyone is reading this, go give them a star :) 

