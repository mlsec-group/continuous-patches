import numpy as np

class params:
    # old one
    # checkerboard = (4,3)
    # pattern_size = (0.05025, 0.051) # x, y in meters

    # # jung-su's
    # checkerboard = (9,7)
    # pattern_size = (0.025, 0.025) # in meters

    # checkerboard = (10,7)
    # pattern_size = (0.02433, 0.0235) # in meters

    checkerboard = (8,5)
    pattern_size = (0.031, 0.031) # in meters

    fx = 183.7350
    fy = 184.1241
    px = 160
    py = 160
    initial_intrinsic_matrix = np.array([[fx,0,px], # initial guess, taken from Shushuai's calibration results 
                                         [0,fy,py], 
                                          [0,0,1]])
    img_shape=(96,160)
