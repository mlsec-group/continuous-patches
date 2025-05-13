import numpy as np

from threading import Thread
import cv2
from collections import deque

import time

import yaml, rowan

from sympy import Plane, Line3D, evalf

# rotation vectors are axis-angle format in "compact form", where
# theta = norm(rvec) and axis = rvec / theta
# they can be converted to a matrix using cv2. Rodrigues, see
# https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
def opencv2quat(rvec):
    angle = np.linalg.norm(rvec)
    if angle == 0:
        q = np.array([1,0,0,0])
    else:
        axis = rvec.flatten() / angle
        q = rowan.from_axis_angle(axis, angle)
    return q



def project_patch(patch, T, image):
    # using cv2 to project the patch instead of FAP place_patch() function,
    # since we don't need to calculate gradients
    width, height = image.shape[:2]
    # print(height, width)
    mask = np.ones_like(patch)

    warped_patch = cv2.warpPerspective(patch, T, (height, width), flags=cv2.INTER_NEAREST)
    mask = cv2.warpPerspective(mask, T, (height, width), flags=cv2.INTER_NEAREST)

    mod_img = image * ~mask.astype(bool)
    mod_img += warped_patch

    return mod_img # return a np array instead of jnp array and convert to double


class PatchDisplayThread(Thread):
    def __init__(self, name, position):
        super().__init__()
        self.name = name
        self.position = position
        self._stay_alive = True
        self.queue = deque(maxlen=5)

    def run(self):
        # Create a named window and move it to the second monitor
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.name, *self.position)  # Assuming the second monitor is to the right of the primary monitor
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        while self._stay_alive:
            if self.queue:
                img = self.queue.popleft()
                cv2.imshow(self.name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.close()

    def update(self, img):
        # Add the image to the queue
        self.queue.append(img)

    def close(self):
        # Destroy the window
        self._stay_alive = False
        cv2.destroyAllWindows()


def scale_tx_ty(sf, tx, ty, patch_size=80, projector_size=(1050, 1680)):
    scaled_patch_size = patch_size * sf
    max_tx = projector_size[1] - scaled_patch_size
    max_ty = projector_size[0] - scaled_patch_size
    return tx * max_tx, ty * max_ty


background = np.zeros((96, 160, 3), dtype=np.uint8)

patch = np.ones((80, 80, 3), dtype=np.uint8) * 255

sf = 1.0
tx = 0.5
ty = 0.0

tx, ty = scale_tx_ty(sf, tx, ty, projector_size=(96, 160))

T = np.array([[sf, 0.0, tx],
              [0.0, sf, ty],
              [0.0, 0.0, 1.0]])

projected_patch = project_patch(patch, T, background)



patch_coords = np.nonzero(projected_patch)
# print(len(patch_coords), patch_coords[0], patch_coords[1])
xmin = patch_coords[1][0]
ymin = patch_coords[0][0]
xmax = patch_coords[1][-1]
ymax = patch_coords[0][-1]

print(xmin, ymin, xmax, ymax)

# from matplotlib import pyplot as plt
# plt.imshow(projected_patch)

# plt.show()

with open('data/camera_calibration.yaml') as f:
    camera_config = yaml.load(f, Loader=yaml.FullLoader)

camera_intrinsic = np.array(camera_config['camera_matrix'])
distortion_coeffs = np.array(camera_config['dist_coeff'])

# print(camera_intrinsic)
# optimal_camera_matrix = cv2.getOptimalNewCameraMatrix(camera_intrinsic, distortion_coeffs, (96, 160), 1, (96, 160))[0]
# print(optimal_camera_matrix)

rvec = np.array(camera_config['rvec'])
tvec = camera_config['tvec']

camera_extrinsic = np.zeros((4,4))
camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
camera_extrinsic[:3, 3] = tvec
camera_extrinsic[-1, -1] = 1.

print("Camera calibration:")
print("Intrinsic:", camera_intrinsic)
print("Distortion: ", distortion_coeffs)
print("Extrinsic:", camera_extrinsic)


def rays_from_bb(bb,mtrx, dist_coeffs):
    # bb - xmin,ymin,xmax,ymax
    # mtrx, dist_vec = get_camera_parameters()
    fx = np.array(mtrx)[0][0]
    fy = np.array(mtrx)[1][1]
    ox = np.array(mtrx)[0][2]
    oy = np.array(mtrx)[1][2]
    # get pixels for bb side center
    # P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    # P2 = np.array([bb[2],(bb[1] + bb[3])/2])

    # center = np.array([(P1[0] + P2[0])/2, (P1[1] + P2[1])/2])
    # # rectify pixels
    # P1_rec = cv2.undistortPoints(P1, mtrx, dist_coeffs, None, mtrx).flatten()
    # P2_rec = cv2.undistortPoints(P2, mtrx, dist_coeffs, None, mtrx).flatten()
    # center_rec = cv2.undistortPoints(center, mtrx, dist_coeffs, None, mtrx).flatten()


    # # get rays for pixels
    # a1 = np.array([(P1_rec[0]-ox)/fx, (P1_rec[1]-oy)/fy, 1.0])
    # a2 = np.array([(P2_rec[0]-ox)/fx, (P2_rec[1]-oy)/fy, 1.0])
    # # normalize rays
    # a1_norm = np.linalg.norm(a1)
    # a2_norm = np.linalg.norm(a2)

    # a_center = np.array([(center_rec[0]-ox)/fx, (center_rec[1]-oy)/fy, 1.0])
    # a_center_norm = np.linalg.norm(a_center)

    # print(P1, P2)
    # print(P1_rec, P2_rec)
    # print(a1, a2)
    # print(a1_norm, a2_norm)

    # print(a_center, a_center_norm)

    ul = np.array([bb[0], bb[1]], dtype=np.float32)
    ur = np.array([bb[2], bb[1]], dtype=np.float32)
    ll = np.array([bb[0], bb[3]], dtype=np.float32)
    lr = np.array([bb[2], bb[3]], dtype=np.float32)
    center = np.array([(ul[0] + lr[0])/2, (ul[1] + lr[1])/2], dtype=np.float32)

    ul_rect = cv2.undistortPoints(ul, mtrx, dist_coeffs, None, mtrx).flatten()
    ur_rect = cv2.undistortPoints(ur, mtrx, dist_coeffs, None, mtrx).flatten()
    ll_rect = cv2.undistortPoints(ll, mtrx, dist_coeffs, None, mtrx).flatten()
    lr_rect = cv2.undistortPoints(lr, mtrx, dist_coeffs, None, mtrx).flatten()
    center_rect = cv2.undistortPoints(center, mtrx, dist_coeffs, None, mtrx).flatten()

   

    print("Rectified Points:")
    print(ul_rect, ur_rect, ll_rect, lr_rect, center_rect)

    ray_ul = np.array([(ul[0]-ox)/fx, (ul[1]-oy)/fy, 1.0])
    ray_ur = np.array([(ur[0]-ox)/fx, (ur[1]-oy)/fy, 1.0])
    ray_ll = np.array([(ll[0]-ox)/fx, (ll[1]-oy)/fy, 1.0])
    ray_lr = np.array([(lr[0]-ox)/fx, (lr[1]-oy)/fy, 1.0])
    ray_center = np.array([(center[0]-ox)/fx, (center[1]-oy)/fy, 1.0])

    print("Rays:")
    print(ray_ul, ray_ur, ray_ll, ray_lr, ray_center)


    ray_ul_norm = np.linalg.norm(ray_ul)
    ray_ur_norm = np.linalg.norm(ray_ur)
    ray_ll_norm = np.linalg.norm(ray_ll)
    ray_lr_norm = np.linalg.norm(ray_lr)
    ray_center_norm = np.linalg.norm(ray_center)

    print("Ray Norms:")
    print(ray_ul/ray_ul_norm, ray_ur/ray_ur_norm, ray_ll/ray_ll_norm, ray_lr/ray_lr_norm, ray_center/ray_center_norm)


    # return ((0., 0., 0.), ray_ul/ray_ul_norm), ((0., 0., 0.), ray_ur/ray_ur_norm), ((0., 0., 0.), ray_ll/ray_ll_norm), ((0., 0., 0.), ray_lr/ray_lr_norm), ((0., 0., 0.), ray_center/ray_center_norm)
    return (0., 0., 0.), (ray_ul/ray_ul_norm, ray_ur/ray_ur_norm, ray_ll/ray_ll_norm, ray_lr/ray_lr_norm, ray_center/ray_center_norm)


p, (ray_ul, ray_ur, ray_ll, ray_lr, ray_center) = rays_from_bb((xmin, ymin, xmax, ymax), camera_intrinsic, distortion_coeffs)
# print("Camera frame:" , p, ray_ul, ray_ur, ray_ll, ray_lr, ray_center)

p_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*p, 1.0])

ul_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*ray_ul, 1.0])
ur_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*ray_ur, 1.0])
ll_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*ray_ll, 1.0])
lr_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*ray_lr, 1.0])
center_in_drone = np.linalg.inv(camera_extrinsic) @ np.array([*ray_center, 1.0])

print("Corners in drone:", ul_in_drone, ur_in_drone)

# print("Drone frame: ", p_in_drone, ul_in_drone, ur_in_drone, ll_in_drone, lr_in_drone, center_in_drone)


# ray_1_drone = np.dot(np.linalg.inv(camera_extrinsic), np.array([*p1, 1.0]))
# ray_2_drone = np.dot(np.linalg.inv(camera_extrinsic), np.array([*p2, 1.0]))
# ray_3_drone = np.dot(np.linalg.inv(camera_extrinsic), np.array([*p3, 1.0]))
# ray_4_drone = np.dot(np.linalg.inv(camera_extrinsic), np.array([*p4, 1.0]))
# ray_5_drone = np.dot(np.linalg.inv(camera_extrinsic), np.array([*p5, 1.0]))

# print(ray_1_drone, ray_2_drone)

yaw = np.radians(30)
pitch = 0
roll = 0

quats = rowan.from_euler(roll, pitch, yaw, 'zyx', axis_type='extrinsic')
print(quats)

drone_pose = np.array([0.0, 0.0, 0.5, *quats])
T_drone_world = np.zeros((4,4))
T_drone_world[:3, :3] = rowan.to_matrix(drone_pose[3:])
T_drone_world[:3, 3] = drone_pose[:3]
T_drone_world[-1, -1] = 1.

# print(T_drone_world)

p_in_world = T_drone_world @ p_in_drone
ul_in_world = T_drone_world @ ul_in_drone
ur_in_world = T_drone_world @ ur_in_drone
ll_in_world = T_drone_world @ ll_in_drone
lr_in_world = T_drone_world @ lr_in_drone
center_in_world = T_drone_world @ center_in_drone

# print(p_in_world, ul_in_world, ur_in_world, ll_in_world, lr_in_world, center_in_world)

# ray_1_world = np.dot(T_drone_world, ray_1_drone)
# ray_2_world = np.dot(T_drone_world, ray_2_drone)
# ray_3_world = np.dot(T_drone_world, ray_3_drone)
# ray_4_world = np.dot(T_drone_world, ray_4_drone)
# ray_5_world = np.dot(T_drone_world, ray_5_drone)
# print(ray_1_world, ray_2_world)


ray_ul = Line3D(p_in_world[:3], ul_in_world[:3])
ray_ur = Line3D(p_in_world[:3], ur_in_world[:3])
ray_ll = Line3D(p_in_world[:3], ll_in_world[:3])
ray_lr = Line3D(p_in_world[:3], lr_in_world[:3])
ray_center = Line3D(p_in_world[:3], center_in_world[:3])

# needs to be actually measured again
projector_ul_world = [2, 1, 2.2]
projector_ur_world = [2, -1.5, 2.2]
projector_ll_world = [2, 1, 0.8]
projector_lr_world = [2, -1.5, 0.8]
projector_plane = Plane(projector_ul_world, projector_ur_world, projector_ll_world)

intersection = projector_plane.intersection(ray_ul)[0]
# print(intersection)t
ul_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
print("Upper left: ", ul_in_world)

intersection = projector_plane.intersection(ray_ur)[0]
ur_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
print("Upper right: ", ur_in_world)

intersection = projector_plane.intersection(ray_ll)[0]
ll_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
print("Lower left: ", ll_in_world)

intersection = projector_plane.intersection(ray_lr)[0]
lr_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
print("Lower right: ", lr_in_world)

intersection = projector_plane.intersection(ray_center)[0]
center_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
print("Center: ", center_in_world)


# from sympy.plotting import plot3d 



projector_ul_display = np.array([0., 0., 1.])
projector_ur_display = np.array([0., 1680., 1.])
projector_ll_display = np.array([1050., 0., 1.])
projector_lr_display = np.array([1050., 1680., 1.])

projector_corners_display = np.array([projector_ul_display, projector_ur_display, projector_ll_display, projector_lr_display])
# projector_corners_world = np.array([np.array([*projector_ul_world, 1.]).T, np.array([*projector_ur_world, 1.]).T, np.array([*projector_ll_world, 1.]).T, np.array([*projector_lr_world, 1.]).T])
projector_corners_world = np.array([projector_ul_world, projector_ur_world, projector_ll_world, projector_lr_world])

print(projector_corners_display)
print(projector_corners_world)

A, res, rank, s = np.linalg.lstsq(projector_corners_world, projector_corners_display, rcond=None)

A = np.array(A)
A[np.abs(A) < 1e-10] = 0

# save as yaml
with open('data/projector_calibration.yaml', 'w') as f:
    yaml.dump({'projector_calibration': A.tolist()}, f)

print(A, A.shape)
# print(res)

transformed = projector_ur_world @ A #np.array([*projector_ul_display, 1.])
# transformed = (transformed / transformed[-1])[:2]
print(transformed)

print("error:", np.linalg.norm(transformed - projector_ur_display))

# intersection = projector_plane.intersection(ray_ur)[0]
# right_in_world = (float(intersection.x.evalf()), float(intersection.y.evalf()), float(intersection.z.evalf()))
# print(right_in_world)

# print(y, type(y))

# intersection = projector_in_world.intersection(ray_ur)
# print(intersection)

# translate to world frame


# def compute_drone_scale(drone_pos, screen_pos=(2, -0.3), d0=2):
#     """
#     Compute the scale factor based on drone distance.
    
#     Parameters:
#     - drone_pos: tuple (x_d, y_d) -> drone position in world frame
#     - screen_pos: tuple (x_s, y_s) -> screen position in world frame
#     - d0: reference distance where sf = 1
    
#     Returns:
#     - sf_drone: base scale factor from drone distance
#     """
#     x_d, y_d = drone_pos
#     x_s, y_s = screen_pos

#     # Compute distance from drone to screen
#     d = np.sqrt((x_s - x_d)**2 + (y_s - y_d)**2)
    
#     # Scale factor inversely proportional to distance
#     sf_drone = d0 / d
#     return sf_drone

# def scale_optimized_values(sf_opt, tx_opt, ty_opt, drone_pos, 
#                            patch_size=80, projector_size=(1050, 1680), 
#                            k_tx=1.0, k_ty=1.0):
#     """
#     Convert optimizer outputs [0,1] to actual scale and translation values.
    
#     Parameters:
#     - sf_opt: Optimized scale factor (0 to 1)
#     - tx_opt: Optimized translation x (0 to 1)
#     - ty_opt: Optimized translation y (0 to 1)
#     - drone_pos: Drone position (x_d, y_d)
#     - patch_size: Base size of the image
#     - projector_size: Projector display size (height, width)
#     - k_tx, k_ty: Scaling coefficients for drone movement influence on tx, ty
    
#     Returns:
#     - Scaled sf, tx, ty (clamped to prevent cropping)
#     """
#     x_d, y_d, z_d = drone_pos
#     x_s, y_s, z_s = 2, -0.3, 1.5  # Fixed screen position

#     # Compute base scale factor based on drone distance
#     sf_drone = compute_drone_scale(drone_pos)

#     # Compute final scale factor
#     sf_scaled = sf_opt * sf_drone  # Scale optimizer output with drone factor

#     # Compute scaled image size
#     scaled_patch_size = patch_size * sf_scaled

#     # Compute max allowed translations, ensuring non-negative values
#     max_tx = max(0, projector_size[1] - scaled_patch_size)  # Width constraint
#     max_ty = max(0, projector_size[0] - scaled_patch_size)  # Height constraint

#     # Compute a linear shift in translation based on drone position
#     # Shift right as y_d decreases, and left as y_d increases
#     tx_shift = k_tx * (y_s - y_d)  # Influence of drone's y position
#     ty_shift = k_ty * (z_d - z_s)  # Influence of drone's z position

#     # Scale translations from optimizer range [0,1] to valid screen coordinates
#     tx_scaled = tx_opt * max_tx + tx_shift
#     ty_scaled = ty_opt * max_ty + ty_shift

#     # Clamp translations to prevent cropping
#     tx_scaled = np.clip(tx_scaled, 0, max_tx)
#     ty_scaled = np.clip(ty_scaled, 0, max_ty)

#     return sf_scaled, tx_scaled, ty_scaled

# Example Usage:

# patch_size = 80
# projector_display_size = (1050, 1680)

# drone_position = (1., 1.0, 1.0)  # Drone moves 1.5 m forward and 1.5 m to the left
# sf_opt, tx_opt, ty_opt = 1.0, 1.0, 1.0  # Example optimizer outputs in [0,1]

# # sf_scaled, tx_scaled, ty_scaled = scale_optimized_values(sf_opt, tx_opt, ty_opt, drone_position)

# sf_alpha = 12.


# sf_scaled = sf_opt * sf_alpha * (np.linalg.norm(drone_position[0] - 2.0) / 2.)
# print(sf_scaled)


# scaled_patch_size = patch_size * sf_scaled
# max_tx = projector_display_size[1] - scaled_patch_size

# scaled_tx_pos = (-drone_position[1] + 1.3) / 2 * max_tx
# scaled_tx = tx_opt * scaled_tx_pos
# scaled_tx = max(0, scaled_tx)

# scaled_ty = ty_opt * (projector_display_size[0] - scaled_patch_size)
# scaled_ty = max(0, scaled_ty)



# print(f"Scaled Scale Factor: {sf_scaled}")
# print(f"Scaled Translation Vector: ({scaled_tx}, {scaled_ty})")



# T = np.zeros((3, 3))
# T[0, 0] = sf_scaled
# T[1, 1] = sf_scaled
# T[0, 2] = scaled_tx
# T[1, 2] = scaled_ty
# T[2, 2] = 1


# projector_display_size = (1050, 1680)
# background = np.zeros((*projector_display_size, 3), dtype=np.uint8)

# # Start the PatchDisplayThread
# display_thread = PatchDisplayThread("Patch", (2561, 0))
# display_thread.start()
# display_thread.update(background)

# projected_patch = project_patch(np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255, T, background)

# display_thread.update(projected_patch)