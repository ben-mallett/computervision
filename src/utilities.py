import numpy as np
from collections.abc import Iterable
        
class ImageUtilities:
    def getNeighborhoodFromPosition(x: int, y: int, image, neighborhoodSize: int, constant: int = 255):
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
            neighborhood = np.ones((neighborhoodSize, neighborhoodSize, image.shape[2])) * constant
        else:
            neighborhood = np.ones((neighborhoodSize, neighborhoodSize), dtype=np.uint8) * constant
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
        Applies the given filter on the given image

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