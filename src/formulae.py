import numpy as np
from random import randint
from collections.abc import Iterable

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
        
    def posterize(pixel, levels: int = 100):
        """
        Brings the pixels into discrete bands of color to appear similar to a movie poster

        Params:
            pixel : Pixel : pixel of image
        """
        return (pixel // levels) * levels