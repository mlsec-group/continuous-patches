from turtle import back
from fastapi import background
from httpx import patch
import numpy as np
from threading import Thread
import cv2

from collections import deque

import yaml

import rowan

from util import opencv2quat, project_patch, scale_tx_ty, construct_T

# from sympy import Line3D, Plane, evalf

from matplotlib import pyplot as plt

import time

def get_bbox(T, patch_size=80, img_size=(96, 160)):
    # print("Transformation matrix for cf img space:", T)
    patch = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
    background = np.zeros((*img_size, 3), dtype=np.uint8)
    projected_patch = project_patch(patch, T, background)
    # projected_patch = cv2.warpPerspective(patch, T, img_size)

    # cv2.imshow("Projected patch", projected_patch)
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    patch_coords = np.nonzero(projected_patch)

    xmin = patch_coords[1][0]
    ymin = patch_coords[0][0]
    xmax = patch_coords[1][-1]
    ymax = patch_coords[0][-1]

    return (xmin, ymin, xmax, ymax)

def bb2camera(bbox, intrinsic, dist_coeffs):
    fx = np.array(intrinsic)[0][0]
    fy = np.array(intrinsic)[1][1]
    ox = np.array(intrinsic)[0][2]
    oy = np.array(intrinsic)[1][2]
    
    ul_image = np.array([bbox[0], bbox[1]], dtype=np.float32)
    ur_image = np.array([bbox[2], bbox[1]], dtype=np.float32)
    ll_image = np.array([bbox[0], bbox[3]], dtype=np.float32)
    lr_image = np.array([bbox[2], bbox[3]], dtype=np.float32)
    center_image = np.array([(ul_image[0] + lr_image[0])/2, (ul_image[1] + lr_image[1])/2], dtype=np.float32)
    
    # ul_image = cv2.undistortPoints(ul_image, intrinsic, dist_coeffs, None, intrinsic).flatten()
    # ur_image = cv2.undistortPoints(ur_image, intrinsic, dist_coeffs, None, intrinsic).flatten()
    # ll_image = cv2.undistortPoints(ll_image, intrinsic, dist_coeffs, None, intrinsic).flatten()
    # lr_image = cv2.undistortPoints(lr_image, intrinsic, dist_coeffs, None, intrinsic).flatten()
    # center_image = cv2.undistortPoints(center_image, intrinsic, dist_coeffs, None, intrinsic).flatten()


    ul_camera = np.array([(ul_image[0]-ox)/fx, (ul_image[1]-oy)/fy, 1.0], dtype=np.float32)
    ur_camera = np.array([(ur_image[0]-ox)/fx, (ur_image[1]-oy)/fy, 1.0], dtype=np.float32)
    ll_camera = np.array([(ll_image[0]-ox)/fx, (ll_image[1]-oy)/fy, 1.0], dtype=np.float32)
    lr_camera = np.array([(lr_image[0]-ox)/fx, (lr_image[1]-oy)/fy, 1.0], dtype=np.float32)
    center_camera = np.array([(center_image[0]-ox)/fx, (center_image[1]-oy)/fy, 1.0], dtype=np.float32)


    ul_camera_norm = ul_camera / np.linalg.norm(ul_camera)
    ur_camera_norm = ur_camera / np.linalg.norm(ur_camera)
    ll_camera_norm = ll_camera / np.linalg.norm(ll_camera)
    lr_camera_norm = lr_camera / np.linalg.norm(lr_camera)
    center_camera_norm = center_camera / np.linalg.norm(center_camera)
    
    center_camera = np.array([0., 0., 0.])
    
    return center_camera, ul_camera_norm, ur_camera_norm, ll_camera_norm, lr_camera_norm, center_camera_norm

def camera2drone(p, extrinsic):
    return (np.linalg.inv(extrinsic) @ np.array([*p, 1.]))[:3]

def drone2world(p, drone_position, drone_quaternion):
    T_drone_world = np.zeros((4,4))
    T_drone_world[:3, :3] = rowan.to_matrix(drone_quaternion)
    T_drone_world[:3, 3] = drone_position
    T_drone_world[-1, -1] = 1.
    
    return (T_drone_world @ np.array([*p, 1.]))[:3]

def calc_intersection(line, plane):
    intersection = plane.intersection(line)[0]
    return (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))

def check_point_in_display(p, projector_size=(1050, 1680)):
    return p[0] >= 0 and p[0] <= projector_size[1] and p[1] >= 0 and p[1] <= projector_size[0]

def corners2transformation(ul, ur, ll, lr, projector_matrix):
    ul_pixels = (ul @ projector_matrix)[:2]
    ur_pixels = (ur @ projector_matrix)[:2]
    ll_pixels = (ll @ projector_matrix)[:2]
    lr_pixels = (lr @ projector_matrix)[:2]
    # center_pixels = (center @ projector_matrix)[:2]

    # print("Upper left in pixels:", ul_pixels)
    # print("Upper right in pixels:", ur_pixels)
    # print("Lower left in pixels:", ll_pixels)
    # print("Lower right in pixels:", lr_pixels)
    # print("Center in pixels:", center_pixels)

    points = np.array([ul_pixels, ur_pixels, ll_pixels, lr_pixels], dtype=np.float32)

    # print("Points:", points, points.shape, points.dtype)

    original_patch_corners = np.array([[0, 0], [80, 0], [0, 80], [80, 80]], dtype=np.float32)

    # print(original_patch_corners, original_patch_corners.shape, original_patch_corners.dtype)

    transformation_matrix = cv2.getPerspectiveTransform(original_patch_corners, points)
    # print("Fround matrix: ",  transformation_matrix, transformation_matrix.shape)
    return transformation_matrix


    # background = np.zeros((1050, 1680, 3), dtype=np.uint8)
    # cv2.circle(background, (int(ul_pixels[0]), int(ul_pixels[1])), 50, (255, 0, 0), -1)
    # cv2.circle(background, (int(ur_pixels[0]), int(ur_pixels[1])), 50, (0, 255, 0), -1)
    # cv2.circle(background, (int(ll_pixels[0]), int(ll_pixels[1])), 50, (0, 0, 255), -1)
    # cv2.circle(background, (int(lr_pixels[0]), int(lr_pixels[1])), 50, (255, 255, 0), -1)
    # cv2.circle(background, (int(center_pixels[0]), int(center_pixels[1])), 50, (0, 255, 255), -1)
    # cv2.imshow("Corners", background)
    # key = cv2.waitKey(0) & 0xFF
    # if key == ord('q'):
    #     cv2.destroyAllWindows()

    
    # points = {'ul': ul_pixels, 'ur': ur_pixels, 'll': ll_pixels, 'lr': lr_pixels, 'center': center_pixels}
    # widths = [
    #     np.abs(ur_pixels[1] - ul_pixels[1]),
    #     np.abs(lr_pixels[1] - ll_pixels[1]),
    #     np.abs((ul_pixels[1] - center_pixels[1])) * 2,
    #     np.abs((ll_pixels[1] - center_pixels[1])) * 2,
    #     np.abs((center_pixels[1] - ur_pixels[1])) * 2,
    #     np.abs((center_pixels[1] - lr_pixels[1])) * 2
    # ]
    
    # heights = [
    #     np.abs(ur_pixels[0] - lr_pixels[0]),
    #     np.abs(ul_pixels[0] - ll_pixels[0]),
    #     np.abs((lr_pixels[0] - center_pixels[0])) * 2,
    #     np.abs((ll_pixels[0] - center_pixels[0])) * 2,
    #     np.abs((center_pixels[0] - ur_pixels[0])) * 2,
    #     np.abs((center_pixels[0] - ul_pixels[0])) * 2
    # ]
    
    # mean_size = np.mean([np.mean(widths), np.mean(heights)])
    # scale_factor = mean_size / 80
    # ty, tx = ul_pixels
    
    # return scale_factor, tx, ty
    
def line_plane_intersection(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise RuntimeError("No intersection or line is within plane")
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * ray_direction + plane_point
    return Psi

    

class PatchDisplayThread(Thread):
    def __init__(self, name, position, drone_pose):
        super().__init__()
        self.name = name
        self.position = position
        self._stay_alive = True
        self.queue = deque(maxlen=1)
        self.drone_pose = drone_pose

        with open('data/camera_calibration.yaml') as f:
            camera_config = yaml.load(f, Loader=yaml.FullLoader)

        self.cf_intrinsic = np.array(camera_config['camera_matrix'], dtype=np.float32)
        self.cf_distortion = np.array(camera_config['dist_coeff'], dtype=np.float32)
        rvec = np.array(camera_config['rvec'])
        tvec = camera_config['tvec']

        camera_extrinsic = np.zeros((4,4))
        camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
        camera_extrinsic[:3, 3] = tvec
        camera_extrinsic[-1, -1] = 1.
        self.cf_extrinsic = camera_extrinsic

        self.cf_im_size = (96, 160)

        with open('data/projector_calibration.yaml') as f:
            projector_matrix = yaml.load(f, Loader=yaml.FullLoader)

        self.projector_matrix = np.array(projector_matrix['projector_calibration'])

        self.projector_size = (1050, 1680)
        self.projector_world = np.array([[2, 1, 2.2],
                                        [2, -1.5, 2.2],
                                        [2, 1, 0.8],
                                        [2, -1.5, 0.8]])

        # self.projector_plane = Plane(self.projector_world[0].tolist(), self.projector_world[1].tolist(), self.projector_world[2].tolist())

    def run(self):
        # Create a named window and move it to the second monitor
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, *self.position)  # Assuming the second monitor is to the right of the primary monitor
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        background = np.zeros((*self.projector_size, 3), dtype=np.uint8)
        cv2.imshow(self.name, background)
        patch = None
        while self._stay_alive:
            if self.queue:
                patch, sf, tx, ty = self.queue.popleft()
            if patch is not None:
                T = self.bb_opt2transformation(sf, tx, ty, np.array(self.drone_pose[0]))
                img = project_patch(patch, T, background)
                # img = cv2.warpPerspective(patch, T, self.projector_size)
                cv2.imshow(self.name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()
            time.sleep(1)


    # def update_simple(self, img):
    #     # Add the image to the queue
    #     self.queue.append(img)

    def update(self, patch, sf, tx, ty):
        # Add the image to the queue
        # T = self.bb_opt2transformation(sf, tx, ty, np.array(self.drone_pose[0]))
        # img = project_patch(patch, T, np.zeros((*self.projector_size, 3)))
        self.queue.append((patch, sf, tx, ty))
        # print("time passed:", time.time() - start_time)
    
    def bb_opt2transformation(self, sf_opt, tx_opt, ty_opt, drone_pose, patch_size=80):
        tx_scaled, ty_scaled = scale_tx_ty(sf_opt, tx_opt, ty_opt, patch_size, self.cf_im_size)
        T = construct_T(sf_opt, tx_scaled, ty_scaled)
        bounding_box = get_bbox(T, patch_size, self.cf_im_size)

        # # print("Patch bounding box:", bounding_box)
        
        camera_center, ul_camera, ur_camera, ll_camera, lr_camera, center_camera = bb2camera(bounding_box, self.cf_intrinsic, self.cf_distortion)
        # # print("Upper left in camera:", ul_camera)
        camera_center_drone = camera2drone(camera_center, self.cf_extrinsic)
        # # print("Camera center in drone:", camera_center_drone)
        ul_drone = camera2drone(ul_camera, self.cf_extrinsic)
        # # print("Upper left in drone:", ul_drone)
        ur_drone = camera2drone(ur_camera, self.cf_extrinsic)
        ll_drone = camera2drone(ll_camera, self.cf_extrinsic)
        lr_drone = camera2drone(lr_camera, self.cf_extrinsic)
        # center_drone = camera2drone(center_camera, self.cf_extrinsic)
        
        camera_center_world = drone2world(camera_center_drone, drone_pose[:3], drone_pose[3:])
        # # print("Drone position: ", drone_pose[:3])
        # # print("Drone quaternion: ", drone_pose[3:])
        # # T = np.zeros((4,4))
        # # T[:3, :3] = rowan.to_matrix(drone_pose[3:])
        # # T[:3, 3] = drone_pose[:3]
        # # T[-1, -1] = 1.
        # # print("Transformation matrix:", T)
        # # print("Camera center in world:", camera_center_world)
        ul_world = drone2world(ul_drone, drone_pose[:3], drone_pose[3:])
        # # print("Upper left in world:", ul_world)
        ur_world = drone2world(ur_drone, drone_pose[:3], drone_pose[3:])
        # # print("Upper right in world:", ur_world)
        ll_world = drone2world(ll_drone, drone_pose[:3], drone_pose[3:])
        # # print("Lower left in world:", ll_world)
        lr_world = drone2world(lr_drone, drone_pose[:3], drone_pose[3:])
        # # print("Lower right in world:", lr_world)
        # center_world = drone2world(center_drone, drone_pose[:3], drone_pose[3:])
        # # print("Center in world:", center_world)
        plane_normal = np.cross(self.projector_world[1] - self.projector_world[0], self.projector_world[2] - self.projector_world[0])
        plane_point = self.projector_world[0]

        intersection_ul = line_plane_intersection(plane_normal, plane_point, ul_world - camera_center_world, camera_center_world)
        intersection_ur = line_plane_intersection(plane_normal, plane_point, ur_world - camera_center_world, camera_center_world)
        intersection_ll = line_plane_intersection(plane_normal, plane_point, ll_world - camera_center_world, camera_center_world)
        intersection_lr = line_plane_intersection(plane_normal, plane_point, lr_world - camera_center_world, camera_center_world)

        

        T = corners2transformation(intersection_ul, intersection_ur, intersection_ll, intersection_lr, self.projector_matrix)
        
        # ray_ul = Line3D(camera_center_world, ul_world)
        # ray_ur = Line3D(camera_center_world, ur_world)
        # ray_ll = Line3D(camera_center_world, ll_world)
        # ray_lr = Line3D(camera_center_world, lr_world)
        # # ray_center = Line3D(camera_center_world, center_world)
        
        # intersection_ul = calc_intersection(ray_ul, self.projector_plane)
        # intersection_ur = calc_intersection(ray_ur, self.projector_plane)
        # intersection_ll = calc_intersection(ray_ll, self.projector_plane)
        # intersection_lr = calc_intersection(ray_lr, self.projector_plane)
        # intersection_center = calc_intersection(ray_center, self.projector_plane)

        # intersection_ul = np.array([2., -1, 1.5])
        # intersection_ur = np.array([2., 0., 1.5])
        # intersection_ll = np.array([2., -1, 0.8])
        # intersection_lr = np.array([2., 0., 1.5])

        # print("Intersection upper left in world:", intersection_ul)
        # print("Intersection upper right in world:", intersection_ur)
        # print("Intersection lower left in world:", intersection_ll)
        # print("Intersection lower right in world:", intersection_lr)
        # print("Intersection center in world:", intersection_center)
        
        # sf, tx, ty = corners2transformation(intersection_ul, intersection_ur, intersection_ll, intersection_lr, intersection_center, self.projector_matrix)
        # print(sf, tx, ty)

        # T = construct_T(sf, tx, ty)

        # T = corners2transformation(intersection_ul, intersection_ur, intersection_ll, intersection_lr, self.projector_matrix)
        # T = np.random.rand(3,3)
        return T


    def close(self):
        # Destroy the window
        self._stay_alive = False
        cv2.destroyAllWindows()