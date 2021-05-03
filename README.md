# 3D_contour_detection_and_fit_by_an_ellipsoid
This program has been writen in order to detect the contour of an ellipsoid object in space then to determine the main characteristics of this ellipsoid.
(config used to run develop the program : Python 3.6.7 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Windows 10 )

The file "picture_example_to_be_treated" is a tiff file that you can use to try the program. It's a bead inserted in a zebrafish embryo. 
The given parameters are used to analyse this picture with the program.

The file "functions.py" gather all the functions written  to make this program work. 



Finaly, I add a warning : the function  "inverse_gaussian_gradient" is working with Python 3.6.7.
However, the function might have changed with the latest version of Python and the program do not work well anymore.


I didn't took time to check this out. Be my guest to find the changing. 

If I can help you for anything, fell free to contact me : alexandre.souchaud91@gmail.com



The active contour method : https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/morphsnakes.py
