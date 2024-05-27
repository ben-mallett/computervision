"""
Lab 3 for Pattern Recognition and Computer Vision

Author: Ben Mallett
Date: 5/15/24
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from collections.abc import Iterable

class Kernels:
    IDENTITY_3X3 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    AVERAGE_3X3 = [
        [1/8, 1/8, 1/8],
        [1/8, 0, 1/8],
        [1/8, 1/8, 1/8]
    ]
    AVERAGE_5X5 = [
        [1/24, 1/24, 1/24, 1/24, 1/24],
        [1/24, 1/24, 1/24, 1/24, 1/24],
        [1/24, 1/24, 0, 1/24, 1/24],
        [1/24, 1/24, 1/24, 1/24, 1/24],
        [1/24, 1/24, 1/24, 1/24, 1/24],
    ]
    RIDGE_3X3 = [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ]
    EDGE_3X3 = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ]
    SHARPEN_3X3 = [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]
    BOX_BLUR_3X3 = [
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ]
    DERIVATIVE_3X3 = [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ]
    GAUSS_BLUR_3X3 = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    GAUSS_BLUR_5X5 = [
        [1/256, 4/256, 6/256, 4/256, 1/256],
        [4/256, 16/256, 24/256, 16/256, 4/256],
        [6/256, 24/256, 36/256, 24/256, 6/256],
        [4/256, 16/256, 24/256, 16/256, 4/256],
        [1/256, 4/256, 6/256, 4/256, 1/256],
    ]
    UNSHARP_MASKING_5X5 = [
        [-1/256, -4/256, -6/256, -4/256, -1/256],
        [-4/256, -16/256, -24/256, -16/256, -4/256],
        [-6/256, -24/256, 476/256, -24/256, -6/256],
        [-4/256, -16/256, -24/256, -16/256, -4/256],
        [-1/256, -4/256, -6/256, -4/256, -1/256],
    ]
    EMBOSS_3X3 = [
        [1, 1, 0],
        [1, 0, -1],
        [0, -1, -1]
    ]
    EMBOSS_5X5 = [
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, -1, -1],
        [1, 0, -1, -1, -1],
        [0, -1, -1, -1, -1],
    ]
    EMBOSS_2_3X3 = [
        [2, 1, 0],
        [1, 0, -1],
        [0, -1, -2],
    ]
    SOBEL_X_3X3 = [
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ]
    SOBEL_Y_3X3 = [
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]
    MOTION_BLUR_3x3 = [
        [1/3, 0, 0],
        [0, 1/3, 0],
        [0, 0, 1/3],
    ]
    MOTION_BLUR_5X5 = [
        [1/5, 0, 0, 0, 0],
        [0, 1/5, 0, 0, 0],
        [0, 0, 1/5, 0, 0],
        [0, 0, 0, 1/5, 0],
        [0, 0, 0, 0, 1/5],
    ]

class Formulae:
    def averageGrayscale(pixel):
        """
        Standard formula for calculating grayscale from color.
        Averages each channel.

        Params:
            pixel : tuple(b, g, r)
        """
        return np.sum(pixel) // 3

    def ncstGrayscale(pixel):
        """
        Scales each channel by a set coefficient to balance out the visual effect each channel has on grayscale images.

        Params:
            pixel : tuple(b, g, r)
        """
        if isinstance(pixel, Iterable):
            return 0.299*(pixel[2]) + 0.587*(pixel[1]) + 0.114*(pixel[0])
        else:
            return pixel

    def noise(pixel, percentage: int = 10):
        """
        Returns 0 or 1 on a percentage chance, otherwise the original pixel

        Params:
            pixel : pixel : image pixel value
            percentage : int : percentage of replacing pixel with noise
        """
        value = randint(1, 100)
        if value <= percentage:
            val = randint(0, 1) * 255
            return val
        else:
            return pixel
        
    def thermal(pixel):
        """
        Returns a mapping of the intensity of the given pixel to a thermal value

        For a brightness x, 
        b = 255 - x
        r = x
        g = 2 * (((x - 127)^2 / -127) + 127) (A parabola centered at 127 with peak 255 and intercepts 0, 255)

        Params:
            pixel : Pixel : pixel of an image
        """
        brightness = pixel
        if isinstance(pixel, Iterable):
            brightness = Formulae.ncstGrayscale(pixel)
        return [255 - brightness, max(min(int(2 * (((brightness - 127)**2 / -127) + 127)), 255), 0), brightness]
    
    def invert(pixel):
        """
        Inverts the colors of the given pixel

        Params: 
            pixel : Pixel : pixel of an image
        """
        return 255 - pixel
    
    def binaryThreshold(pixel, threshold: int = 127):
        """
        Maps a pixel to a binary pixel given a threhsold value

        Params: 
            pixel : Pixel : pixel of an image
        """
        brightness = pixel
        if isinstance(pixel, Iterable):
            brightness = Formulae.ncstGrayscale(pixel)
        return 0 if brightness < threshold else 255
    
    def sepiaTone(pixel, factor=1.0):
        """
        Adds a sepia tone to the image

        Params:
            pixel : Pixel : pixel of image
            factor : float : intensity of sepia tone
        """
        if isinstance(pixel, Iterable):
            r = int(0.393 * pixel[2] + 0.769 * pixel[1] + 0.189 * pixel[0]) * factor
            g = int(0.349 * pixel[2] + 0.686 * pixel[1] + 0.168 * pixel[0]) * factor
            b = int(0.272 * pixel[2] + 0.534 * pixel[1] + 0.131 * pixel[0]) * factor
            return [min(b, 255), min(g, 255), min(r, 255)]
        else:
            return pixel
        
    def posterize(pixel, levels: int = 50):
        """
        Brings the pixels into discrete bands of color to appear similar to a movie poster

        Params:
            pixel : Pixel : pixel of image
        """
        return (pixel // levels) * levels

        
class ImageUtilities:
    def getNeighborhoodFromPosition(x: int, y: int, image, neighborhoodSize: int):
        """
        Returns the neighborhood of pixels centered on the given position from the given image. Assumes a zero padding

        Params:
            x : int : x position in image
            y : int : y position in image
            image : image : image to get neighborhood from
            neighborhoodSize : int : size of neighborhood to get

        Returns: 
            neighborhood : Pixel[][] : list of pixels in the neighborhood as an array
        """
        neighborhood = None
        if len(image.shape) == 3:
            neighborhood = np.zeros((neighborhoodSize, neighborhoodSize, image.shape[2]))
        else:
            neighborhood = np.zeros((neighborhoodSize, neighborhoodSize), dtype=np.uint8)
        offset = neighborhoodSize // 2
        rowStart = max(0, x - offset)
        rowEnd = min(image.shape[0], x + offset + 1)
        colStart = max(0, y - offset)
        colEnd = min(image.shape[1], y + offset + 1)
        
        rowOffsetStart = offset - x + rowStart
        rowOffsetEnd = rowOffsetStart + rowEnd - rowStart
        colOffsetStart = offset - y + colStart
        colOffsetEnd = colOffsetStart + colEnd - colStart
        
        neighborhood[rowOffsetStart:rowOffsetEnd, colOffsetStart:colOffsetEnd] = image[rowStart:rowEnd, colStart:colEnd]
        
        return neighborhood
        

    def applyFilter(image, filter, outputShape: tuple, neighborhoodSize = 1):
        """
        Returns a grayscale version of the given image using the given formula

        Params:
            image : image : image to convert to grayscale
            formula : function : formula to use on a neighborhood to get a new pixel value
            outputShape : tuple : the desired output shape of the image
            neighborhoodSize : int : an odd number corresponding to the size of the neighborhood to pass to a filter function
        """
        if neighborhoodSize % 2 != 0:
            newImage = np.zeros(outputShape, dtype=np.uint8)
            for i, row in enumerate(image):
                for j, pixel in enumerate(row):
                    if neighborhoodSize == 1:
                        newImage[i][j] = filter(pixel)
                    else:
                        neighborhood = ImageUtilities.getNeighborhoodFromPosition(i, j, image, neighborhoodSize)
                        newImage[i][j] = filter(neighborhood)
            newImage = np.clip(newImage, 0, 255).astype(np.uint8)
            return newImage
        

    def getHistogram(image):
        """
        Takes in an image and returns a mapping of its value to histogram count

        Params:
            image : image

        Returns:
            histogramMapping : dict value -> histogram
        """
        histogram = {}
        for row in image:
            for pixel in row:
                if pixel in histogram:
                    histogram[pixel] += 1
                else:
                    histogram[pixel] = 1
        return histogram

    def getCDF(histogramMapping):
        """
        Given a histogram mapping returns a CDF mapping

        Params:
            histogramMapping : dict value -> histogram

        Returns:
            cdfMapping : dict value -> cdf
        """
        cdfMapping = {}
        prior = 0
        for value in sorted(histogramMapping.keys()):
            cdfMapping[value] = prior + histogramMapping[value]
            prior += histogramMapping[value]
        return cdfMapping

    def equalizationMapping(value, cdf):
        """
        Given a value and a CDF returns the scaled value

        Params:
            value : pixel : value of pixel to scale
            cdf : dict value -> cdf : dictionary of CDF values
        
        Returns:
            scaled : pixel : the scaled pixel value
        """
        minVal = min(cdf, key=cdf.get)
        maxVal = max(cdf, key=cdf.get)
        mapping = (cdf[value] - cdf[minVal]) / (cdf[maxVal] - cdf[minVal]) * 255 
        return int(mapping)

    def equalizeImage(image):
        """
        Given an image, equalizes it based on its histogram.

        For gray scale images works directly with the intensity value.
        For images with multiple dimensions will split and perform the equalization per dimension, then merge for resulting image.

        Params:
            image: image

        Returns:
            equalized: image : equalized image
        """
        if (len(image.shape) > 2):
            channelImages = []
            for i in range(0, image.shape[2]):
                channelImage = ImageUtilities.applyFilter(image, lambda x : x[i], (image.shape[0], image.shape[1]))
                channelImages.append(channelImage)
            equalizedChannels = []
            for channel in channelImages:
                equalizedChannels.append(ImageUtilities.equalizeImage(channel))
            equalized = np.full(image.shape, dtype=np.uint8, fill_value=0)
            for k, channel in enumerate(equalizedChannels):
                for i, row in enumerate(channel):
                    for j, pixel in enumerate(row):
                        equalized[i][j][k] = pixel
            return equalized
        else:
            histogram = ImageUtilities.getHistogram(image)
            cdf = ImageUtilities.getCDF(histogram)
            equalizationFilter = lambda x : ImageUtilities.equalizationMapping(x, cdf)
            return ImageUtilities.applyFilter(image, equalizationFilter, image.shape)

    def applyKernelOnNieghborhood(neighborhood, kernel):
        """
        Applies the given kernel on the given neighborhood.

        Params:
            neighborhood : image : portion of image to apply kernel on 
            kernel : image : mask to apply on neighborhood
        """
        if isinstance(neighborhood[0][0], Iterable):
            result = np.zeros(neighborhood.shape[2], dtype=float) 
            for i, row in enumerate(kernel):
                for j, mask in enumerate(row):
                    result += mask * neighborhood[i][j]
            return np.clip(result, 0, 255).astype(np.uint8).tolist()  
        else:
            result = 0
            for i, row in enumerate(kernel):
                for j, mask in enumerate(row):
                    result += mask * neighborhood[i][j]
            return np.clip(result, 0, 255).astype(np.uint8)
        
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
            neighborhoodSize = 3 if '3X3' in name else 5
            outputImg = ImageUtilities.applyFilter(image, lambda x: ImageUtilities.applyKernelOnNieghborhood(x, kernel), image.shape, neighborhoodSize=neighborhoodSize)
            outputs.append([outputImg, name])


    fig, axes = plt.subplots(5, 6, figsize=(15, 10))
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

showcase('./images/dog.jpeg')
