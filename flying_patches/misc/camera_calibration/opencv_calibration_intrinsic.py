#!/usr/bin/env python

import cv2
import numpy as np
import yaml
from utils import *
from offline_calibration_intrinsic import compute_calibration
from parameters import params
import argparse
from pathlib import Path
# get image points and object points and save in yaml file
# python3 opencv_calibration.py PATH-TO-FOLDER

# each checkerboard pattern in checkerboard frame
def pattern_checkerboard(row, column, size_x, size_y):
    chess_points = np.zeros((row * column, 3), np.float32) 
    for i in range(0, column):
        for j in range(0, row):
            chess_points[i*row+j,0] = i*size_x
            chess_points[i*row+j,1] = j*size_y
    return chess_points


def main():
    parser = argparse.ArgumentParser(description='Get dataset from the folder')
    parser.add_argument('path') #"-path", help="Path to saved dataset with images, .csv and .yaml file")
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    folder = Path(args.path)

    img_obj_points_yaml = folder / "img_obj_points.yaml"
    if img_obj_points_yaml.exists():
        print("img_obj_points exists - skipping interactive part")
        with open(img_obj_points_yaml, 'r') as stream:
            points = yaml.safe_load(stream)
        files = list(points.keys())
    else:
        files = None

    img_size = (96,160)
    parameters = params()
    CHECKERBOARD = parameters.checkerboard
    size_x, size_y = parameters.pattern_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    pattern_chess = pattern_checkerboard(CHECKERBOARD[0], CHECKERBOARD[1], size_x, size_y)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) # 12x3

    img_obj_data = {}
    for fname in folder.glob("*.png"):
        fname = fname.name
        per_image={}
        img = cv2.imread(str(folder / fname))
        # Crop borders of the 324x324
        if img.shape[0] != img_size[0]:
            print("WARNING: wrong image shape! cropping...")
            img = img[2:img_size[0]+2,2:img_size[1]+2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_MARKER)
        print(corners)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (5,5),(-1,-1), criteria)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            if files is None:
                if args.auto:
                    take_image = True
                else:
                    cv2.imshow(fname,img)
                    keyboard = cv2.waitKey(0) % 256
                    take_image = keyboard == ord('y') # press 'y', if the detection is visually fine
                    cv2.destroyWindow(fname)
            else:
                take_image = fname in files # in previous run, a user selected 'y'
            if take_image: 
                # T_CF_M = np.eye(4)
                # T_CH_M = np.eye(4)
                # objp = pattern_cf(T_CF_M, T_CH_M, pattern_chess) #object points in robot frame
                objp = pattern_chess
                # print(objp)
                # exit()
                per_image['imgpoints']=np.squeeze(corners2).tolist()
                per_image['objpoints']=np.squeeze(objp).tolist()
                img_obj_data[fname]=per_image
                print("Accept {}".format(fname))
            else:
                print("Skip {}".format(fname))
        else:
            print('Not detected for {}'.format(fname))


    with open(img_obj_points_yaml, "w") as f:
        yaml.dump(img_obj_data, f)

    compute_calibration(img_obj_points_yaml, folder / "calibration.yaml")

if __name__ == "__main__":
    main()
