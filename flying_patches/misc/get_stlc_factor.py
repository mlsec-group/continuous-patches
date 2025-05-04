import numpy as np

import rowan

import torch
import cv2

import yaml


def gen_transformation_matrix(sf, tx, ty):
    matrix = np.zeros((3,3))
    matrix[0,0] = sf
    matrix[1,1] = sf
    matrix[0, 2] = tx
    matrix[1, 2] = ty
    matrix[2, 2] = 1.
    return matrix

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

def get_bb_patch(transformation):
    transformation_t = torch.tensor(transformation).unsqueeze(0)
    inv_transform = torch.inverse(transformation_t)[:, :2]
    affine_grid = torch.nn.functional.affine_grid(inv_transform, size=(1, 1, 96, 160), align_corners=True)

    transformed_patch = torch.nn.functional.grid_sample(torch.ones(1, 1, 96, 160).double(), affine_grid, align_corners=True, padding_mode='zeros')

    patch_coords = torch.nonzero(transformed_patch)[..., 2:]

    xmin = patch_coords[0][1].item()
    ymin = patch_coords[0][0].item()
    xmax = patch_coords[-1][1].item()
    ymax = patch_coords[-1][0].item()

    # fig, ax = plt.subplots()
    # ax.imshow(transformed_patch[0][0], cmap='gray')
    # plt.show()

    return [xmin, ymin, xmax, ymax]

# printed patch width = 28.2 cm
RADIUS = 0.137 # in m

# compute relative position of center of patch in camera frame
def xyz_from_bb(bb,mtrx, dist_coeffs):
    # bb - xmin,ymin,xmax,ymax
    # mtrx, dist_vec = get_camera_parameters()
    fx = np.array(mtrx)[0][0]
    fy = np.array(mtrx)[1][1]
    ox = np.array(mtrx)[0][2]
    oy = np.array(mtrx)[1][2]
    # get pixels for bb side center
    P1 = np.array([bb[0],(bb[1] + bb[3])/2])
    P2 = np.array([bb[2],(bb[1] + bb[3])/2])
    # rectify pixels
    P1_rec = cv2.undistortPoints(P1, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix
    P2_rec = cv2.undistortPoints(P2, mtrx, dist_coeffs, None, mtrx).flatten() # distortion is included in my camera intrinsic matrix

    # get rays for pixels
    a1 = np.array([(P1_rec[0]-ox)/fx, (P1_rec[1]-oy)/fy, 1.0])
    a2 = np.array([(P2_rec[0]-ox)/fx, (P2_rec[1]-oy)/fy, 1.0])
    # normalize rays
    a1_norm = np.linalg.norm(a1)
    a2_norm = np.linalg.norm(a2)
    # get the distance    
    distance = (np.sqrt(2)*RADIUS)/(np.sqrt(1-np.dot(a1,a2)/(a1_norm*a2_norm)))
    # get central ray
    ac = (a1+a2)/2
    # get the position
    xyz = distance*ac/np.linalg.norm(ac)
    return xyz

def get_patch_in_victim(camera_intrinsic, distortion, camera_extrinsic, affine_factors):

    transformation_matrix = gen_transformation_matrix(*affine_factors)
            
    bounding_box_placed_patch = get_bb_patch(transformation_matrix)

    patch_in_camera = xyz_from_bb(bounding_box_placed_patch, camera_intrinsic, distortion)
    # print(patch_in_camera)
    patch_in_victim = (np.linalg.inv(camera_extrinsic) @ [*patch_in_camera, 1])[:3]

    return patch_in_victim

if __name__ == '__main__':

    with open('misc/camera_calibration/calibration.yaml') as f:
        camera_config = yaml.load(f, Loader=yaml.FullLoader)

    camera_intrinsic = np.array(camera_config['camera_matrix'])
    distortion_coeffs = np.array(camera_config['dist_coeff'])

    rvec = np.array(camera_config['rvec'])
    tvec = camera_config['tvec']

    camera_extrinsic = np.zeros((4,4))
    camera_extrinsic[:3, :3] = rowan.to_matrix(opencv2quat(rvec))
    camera_extrinsic[:3, 3] = tvec
    camera_extrinsic[-1, -1] = 1.


    opt_trans_1 = np.array([0.4, 0.0, 0.0])
    opt_trans_2 = np.array([0.4, 0.3, 0.0])
    
    patch_in_victim_1 = get_patch_in_victim(camera_intrinsic, distortion_coeffs, camera_extrinsic, opt_trans_1)
    patch_in_victim_2 = get_patch_in_victim(camera_intrinsic, distortion_coeffs, camera_extrinsic, opt_trans_2)
    
    print("target", patch_in_victim_2-patch_in_victim_1)
    print("position", opt_trans_2 - opt_trans_1)