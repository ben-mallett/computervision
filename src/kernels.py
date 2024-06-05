import numpy as np

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
    MALLETT_EDGE_1_5X5 = [
        [-1, 0, 0, 0, -1],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [-1, 0, 0, 0, -1],
    ]
    NEIGHBORHOOD_11X11 = [
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
        [1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121, 1/121],
    ] 
    GAUSS_BLUR_11X11 = [
        [  1/1034576,   10/1034576,    45/1034576,   120/1034576,   210/1034576,   252/1034576,   210/1034576,   120/1034576,    45/1034576,   10/1034576,   1/1034576], 
        [ 10/1034576,  100/1034576,   450/1034576,  1200/1034576,  2100/1034576,  2520/1034576,  2100/1034576,  1200/1034576,   450/1034576,  100/1034576,  10/1034576], 
        [ 45/1034576,  450/1034576,  2025/1034576,  5400/1034576,  9450/1034576, 11340/1034576,  9450/1034576,  5400/1034576,  2025/1034576,  450/1034576,  45/1034576], 
        [120/1034576, 1200/1034576,  5400/1034576, 14400/1034576, 25200/1034576, 30240/1034576, 25200/1034576, 14400/1034576,  5400/1034576, 1200/1034576, 120/1034576], 
        [210/1034576, 2100/1034576,  9450/1034576, 25200/1034576, 44100/1034576, 52920/1034576, 44100/1034576, 25200/1034576,  9450/1034576, 2100/1034576, 210/1034576], 
        [252/1034576, 2520/1034576, 11340/1034576, 30240/1034576, 52920/1034576, 63504/1034576, 52920/1034576, 30240/1034576, 11340/1034576, 2520/1034576, 252/1034576], 
        [210/1034576, 2100/1034576,  9450/1034576, 25200/1034576, 44100/1034576, 52920/1034576, 44100/1034576, 25200/1034576,  9450/1034576, 2100/1034576, 210/1034576], 
        [120/1034576, 1200/1034576,  5400/1034576, 14400/1034576, 25200/1034576, 30240/1034576, 25200/1034576, 14400/1034576,  5400/1034576, 1200/1034576, 120/1034576], 
        [ 45/1034576,  450/1034576,  2025/1034576,  5400/1034576,  9450/1034576, 11340/1034576,  9450/1034576,  5400/1034576,  2025/1034576,  450/1034576,  45/1034576], 
        [ 10/1034576,  100/1034576,   450/1034576,  1200/1034576,  2100/1034576,  2520/1034576,  2100/1034576,  1200/1034576,   450/1034576,  100/1034576,  10/1034576], 
        [  1/1034576,   10/1034576,    45/1034576,   120/1034576,   210/1034576,   252/1034576,   210/1034576,   120/1034576,    45/1034576,   10/1034576,   1/1034576], 
    ]
    CORNER_DETECTION_3X3 = [
        [ 1, -2,  1],
        [-2,  4, -2],
        [ 1, -2,  1],
    ]
    SOBEL_XY_3X3 =  np.add(SOBEL_X_3X3, SOBEL_Y_3X3)