import cv2
import numpy as np
import matplotlib.pyplot as plt
from kernels import Kernels
from formulae import Formulae
from utilities import ImageUtilities

def showOutputs(outputs, figsize: tuple, w: int, h: int):
    """
    Displays the output images captioned in matplotlib

    Params:
        outputs : list : list of [image, 'imageName']
        figsize : tuple : size of output figure
        w : int : number of columns
        h : int : number of rows
    """
    fig, axes = plt.subplots(w, h, figsize=figsize)
    axes = axes.flatten()

    for ax in axes:
        ax.set_axis_off()
    for ax, img in zip(axes.flatten(), outputs):
        if img[0] is not None: 
            imgRgb = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
            ax.imshow(imgRgb)
            ax.set_title(img[1])
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

        
def showcase(imagePath: str, grayscale: bool = False):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)

    if grayscale:
        image = ImageUtilities.applyFilter(image, Formulae.ncstGrayscale, image.shape)
    outputs = []
    for name, formula in [[attr, getattr(Formulae, attr)] for attr in dir(Formulae) if callable(getattr(Formulae, attr))]:
        if '__' not in name:
            print(f'Running: {name}')
            outputImg = ImageUtilities.applyFilter(image, formula, image.shape)
            outputs.append([outputImg, name])
    for name, kernel in [[attr, getattr(Kernels, attr)] for attr in dir(Kernels)]:
        if '__' not in name:
            print(f'Running: {name}')
            neighborhoodSize = 3 if '3X3' in name else 5 if '5x5' in name else 11
            outputImg = ImageUtilities.applyFilter(image, lambda x: ImageUtilities.applyKernelOnNieghborhood(x, kernel), image.shape, neighborhoodSize=neighborhoodSize)
            outputs.append([outputImg, name])
    print(f'Running Histogram Equalization')
    outputs.append([ImageUtilities.equalizeImage(image), "equalized"])
    showOutputs(outputs, (15, 10), 5, 6)


def blend(img1: str, img2: str):
    """
    Blends two images in a Marilyn Einstein fashion

    Params:
        img1 : str : path to first image
        img2 : str : path to second image
    """
    im1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    im2Resized = cv2.resize(im2, (im1.shape[1], im1.shape[0]))
    im1F = ImageUtilities.applyFilter(im1, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.NEIGHBORHOOD_11X11), im1.shape, neighborhoodSize=11)
    im2F = ImageUtilities.applyFilter(im2Resized, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.NEIGHBORHOOD_11X11), im2Resized.shape, neighborhoodSize=11)
    im1H = np.subtract(im1, im1F) + 127
    combined = cv2.addWeighted(im1H, 0.5, im2F, 0.5, 0)

    blurred = ImageUtilities.applyFilter(combined, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.GAUSS_BLUR_3X3), combined.shape, neighborhoodSize=3)

    outputs = [[im1H, 'Image 1 High Pass'], [im2F, 'Image 2 Low Pass'], [blurred, 'Hybrid']]
    showOutputs(outputs, (15, 5), 1, 3)

def showcaseBlur(filepath: str):
    """
    Showcases the affect of blurring using 3d visualization and gaussian blur kernels.

    Params:
        filepath : str : path to image 
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    blurred5x5 = ImageUtilities.applyFilter(image, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.GAUSS_BLUR_5X5), image.shape, neighborhoodSize=5)
    blurred11x11 = ImageUtilities.applyFilter(image, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.GAUSS_BLUR_11X11), image.shape, neighborhoodSize=11)

    outputs = [[image, 'Regular Image'], [blurred5x5, 'Gauss 5x5'], [blurred11x11, 'Gauss 11x11']]
    ImageUtilities.visualizeIn3D(outputs)

def showcaseEdge(filepath: str):
    """
    Showcases Sobel and Canny edge detection of an image before and after blur

    Params:
        filepath : str : path to image
    """
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    sobelX = ImageUtilities.applyFilter(image, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.SOBEL_X_3X3), image.shape, neighborhoodSize=3)
    sobelY = ImageUtilities.applyFilter(image, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.SOBEL_Y_3X3), image.shape, neighborhoodSize=3)
    sobel = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    sobelHystersis50 = ImageUtilities.applyFilter(sobel, lambda x : Formulae.hysterisisThreshold(x, 25, 255), sobel.shape)
    sobelHystersis150 = ImageUtilities.applyFilter(sobel, lambda x : Formulae.hysterisisThreshold(x, 75, 255), sobel.shape)
    canny = cv2.Canny(image, 100, 100)
    blurred = ImageUtilities.applyFilter(image, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.GAUSS_BLUR_5X5), image.shape, neighborhoodSize=5)
    blurredSobelX = ImageUtilities.applyFilter(blurred, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.SOBEL_X_3X3), blurred.shape, neighborhoodSize=3)
    blurredSobelY = ImageUtilities.applyFilter(blurred, lambda x : ImageUtilities.applyKernelOnNieghborhood(x, Kernels.SOBEL_Y_3X3), blurred.shape, neighborhoodSize=3)
    blurredSobel = cv2.addWeighted(blurredSobelX, 0.5, blurredSobelY, 0.5, 0)
    blurredSobelHystersis50 = ImageUtilities.applyFilter(blurredSobel, lambda x : Formulae.hysterisisThreshold(x, 25, 255), blurredSobel.shape)
    blurredSobelHystersis150 = ImageUtilities.applyFilter(blurredSobel, lambda x : Formulae.hysterisisThreshold(x, 75, 255), blurredSobel.shape)
    blurredCanny = cv2.Canny(blurred, 100, 100)
    outputs = [ 
        [image, 'Original'], 
        [sobelHystersis50, 'Sobel Hystersis 25/255'], 
        [sobelHystersis150, 'Sobel Hystersis 75/255'], 
        [canny, 'Canny 100/100'],
        [blurred, 'Blurred'],
        [blurredSobelHystersis50, 'Blurred Sobel Hystersis 25/255'],
        [blurredSobelHystersis150, 'Blurred Sobel Hystersis 75/255'],
        [blurredCanny, 'Blurred Canny 100/100']
    ]
    showOutputs(outputs, (15, 10), 2, 4)

blend('./images/dog.jpeg', './images/xray.jpeg')



