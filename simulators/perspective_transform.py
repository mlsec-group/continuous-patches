import numpy as np
import torch
import cv2



# opencv function:
# cv::Mat cv::getPerspectiveTransform(const Point2f src[], const Point2f dst[], int solveMethod)
# {
#     CV_INSTRUMENT_REGION();

#     Mat M(3, 3, CV_64F), X(8, 1, CV_64F, M.ptr());
#     double a[8][8], b[8];
#     Mat A(8, 8, CV_64F, a), B(8, 1, CV_64F, b);

#     for( int i = 0; i < 4; ++i )
#     {
#         a[i][0] = a[i+4][3] = src[i].x;
#         a[i][1] = a[i+4][4] = src[i].y;
#         a[i][2] = a[i+4][5] = 1;
#         a[i][3] = a[i][4] = a[i][5] =
#         a[i+4][0] = a[i+4][1] = a[i+4][2] = 0;
#         a[i][6] = -src[i].x*dst[i].x;
#         a[i][7] = -src[i].y*dst[i].x;
#         a[i+4][6] = -src[i].x*dst[i].y;
#         a[i+4][7] = -src[i].y*dst[i].y;
#         b[i] = dst[i].x;
#         b[i+4] = dst[i].y;
#     }

#     solve(A, B, X, solveMethod);
#     M.ptr<double>()[8] = 1.;

#     return M;
# }

def opencv_perspective_coeffs(startpoints, endpoints):
    a = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8, 1), dtype=np.float64)

    for i in range(4):
        a[i][0] = a[i+4][3] = startpoints[i][0]
        a[i][1] = a[i+4][4] = startpoints[i][1]
        a[i][2] = a[i+4][5] = 1
        a[i][3:6] = a[i+4][:3] = 0
        a[i][6] = -startpoints[i][0]*endpoints[i][0]
        a[i][7] = -startpoints[i][1]*endpoints[i][0]
        a[i+4][6] = -startpoints[i][0]*endpoints[i][1]
        a[i+4][7] = -startpoints[i][1]*endpoints[i][1]
        b[i] = endpoints[i][0]
        b[i+4] = endpoints[i][1]

    # print(a)
    # print(b)
    
        
    return np.linalg.lstsq(a, b)[0].flatten()

def opencv_warp_perspective(img, M, output_shape):
    # nearest neighbor only
    output_img = np.zeros((output_shape[1], output_shape[0]))

    for x in range(output_shape[0]):
        for y in range(output_shape[1]):
            coords = np.linalg.inv(M) @ np.array([x, y, 1])
            coords = coords / coords[2]  # normalize homogeneous coords

            src_x, src_y = int(round(coords[0])), int(round(coords[1]))

            # copy valid pixels from source img to output img
            if 0 <= src_x < img.shape[1] and 0 <= src_y < img.shape[0]:
                output_img[y, x] = img[src_y, src_x]

    return output_img

    # coords_x = np.linspace(0, img.shape[1]-1, output_shape[1])
    # coords_y = np.linspace(0, img.shape[0]-1, output_shape[0])
    # x_grid, y_grid  = np.meshgrid(coords_x, coords_y)
    # coords = np.array([x_grid.flatten(), y_grid.flatten(), np.ones_like(x_grid).flatten()])

    # # print(coords)

    # nominator = M @ coords
    # denominator = M[2] @ coords

    # homogeneous = nominator / denominator
    # normalized = homogeneous / homogeneous[2]

    # # Round to nearest integer
    # src_x = np.round(normalized[0, :]).astype(int)
    # src_y = np.round(normalized[1, :]).astype(int)

    # # Mask for points within bounds of original image
    # mask = (src_x >= 0) & (src_x < img.shape[1]) & (src_y >= 0) & (src_y < img.shape[0])

    # # Copy pixel values from original image to output image
    # output_img[y_grid.flatten()[mask], x_grid.flatten()[mask]] = img[src_y[mask], src_x[mask]]

    # return output_img
    
    # print(coords.shape)
    # img_coords = np.array([coords_x, coords_y, np.])
    # # [ [0, 0, 1]
    # #   [0, 1, 1] ...
    # print(grid[:3], grid.shape)
    # coeffs = M.flatten()


    # x_out = (coeffs[0] * coords_x + coeffs[1] * coords_y + coeffs[2]) / (coeffs[6] * coords_x + coeffs[7] * coords_y + 1)
    # y_out = (coeffs[3] * coords_x + coeffs[4] * coords_y + coeffs[5]) / (coeffs[6] * coords_x + coeffs[7] * coords_y + 1)

    # print(np.array())

    # warped_coords = np.linalg.inv(M) @ 



# the pytorch implementation seems to be actually closer to https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/
# but somehow, the perspective transformation matrix doesn't look right

def _get_perspective_coeffs(startpoints, endpoints):
    """Helper function to get the coefficients (a, b, c, d, e, f, g, h) for the perspective transforms.

    In Perspective Transform each pixel (x, y) in the original image gets transformed as,
     (x, y) -> ( (ax + by + c) / (gx + hy + 1), (dx + ey + f) / (gx + hy + 1) )

    Args:
        startpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the original image.
        endpoints (list of list of ints): List containing four lists of two integers corresponding to four corners
            ``[top-left, top-right, bottom-right, bottom-left]`` of the transformed image.

    Returns:
        octuple (a, b, c, d, e, f, g, h) for transforming each pixel.
    """
    if len(startpoints) != 4 or len(endpoints) != 4:
        raise ValueError(
            f"Please provide exactly four corners, got {len(startpoints)} startpoints and {len(endpoints)} endpoints."
        )
    a_matrix = torch.zeros(2 * len(startpoints), 8, dtype=torch.float64)

    for i, (p1, p2) in enumerate(zip(endpoints, startpoints)):
        a_matrix[2 * i, :] = torch.tensor([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        a_matrix[2 * i + 1, :] = torch.tensor([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    # print(a_matrix)

    b_matrix = torch.tensor(startpoints, dtype=torch.float64).view(8)
    # do least squares in double precision to prevent numerical issues
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gelss").solution.to(torch.float32)

    # output: List[float] = res.tolist()
    return res.tolist()

def _perspective_grid(coeffs, ow: int, oh: int):
    # https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/
    # src/libImaging/Geometry.c#L394

    #
    # x_out = (coeffs[0] * x + coeffs[1] * y + coeffs[2]) / (coeffs[6] * x + coeffs[7] * y + 1)
    # y_out = (coeffs[3] * x + coeffs[4] * y + coeffs[5]) / (coeffs[6] * x + coeffs[7] * y + 1)
    #
    theta1 = torch.tensor(
        [[[coeffs[0], coeffs[1], coeffs[2]], [coeffs[3], coeffs[4], coeffs[5]]]]
    )
    theta2 = torch.tensor([[[coeffs[6], coeffs[7], 1.0], [coeffs[6], coeffs[7], 1.0]]])

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3)
    x_grid = torch.linspace(d, ow * 1.0 + d - 1.0, steps=ow)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace(d, oh * 1.0 + d - 1.0, steps=oh).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    base_grid[..., 2].fill_(1)

    rescaled_theta1 = theta1.transpose(1, 2) / torch.tensor([0.5 * ow, 0.5 * oh])
    output_grid1 = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta1)
    output_grid2 = base_grid.view(1, oh * ow, 3).bmm(theta2.transpose(1, 2))

    output_grid = output_grid1 / output_grid2 - 1.0
    return output_grid.view(1, oh, ow, 2)


# def _apply_grid_transform(img, grid, mode, fill):

#     # img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

#     if img.shape[0] > 1:
#         # Apply same grid to a batch of images
#         grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])

#     # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
#     if fill is not None:
#         mask = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
#         img = torch.cat((img, mask), dim=1)

#     img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

#     # Fill with required color
#     if fill is not None:
#         mask = img[:, -1:, :, :]  # N * 1 * H * W
#         img = img[:, :-1, :, :]  # N * C * H * W
#         mask = mask.expand_as(img)
#         fill_list, len_fill = (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
#         fill_img = torch.tensor(fill_list, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
#         if mode == "nearest":
#             mask = mask < 0.5
#             img[mask] = fill_img[mask]
#         else:  # 'bilinear'
#             img = img * mask + (1.0 - mask) * fill_img

#     # img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
#     return img




start = np.array([[0., 0.], [0., 5.], [5., 5.], [5., 0.]], dtype=np.float32)  # 5x5 patch in top left corner  
end = np.array([[0., 10.], [0., 15.], [5., 15.], [5., 10.]], dtype=np.float32) # translation by ty + 10

# transformation matrix should look like
# [[ 1.  0.  0.]
#  [ 0.  1. 10.]
#  [ 0.  0.  1.]]

# print(cv2.getPerspectiveTransform(start, end))
# returns
# [[ 1.  0.  0.]
#  [ 0.  1. 10.]
#  [ 0.  0.  1.]]

# coeffs = _get_perspective_coeffs(start, end)

# matrix = np.zeros(9)
# matrix[:-1] = coeffs
# matrix[-1] = 1.
# print(np.round(matrix.reshape(3,3), 2))
# [[ 1.  0.  -0.]
#  [ 0.  1. -10.]
#  [ 0.  0.  1.]]

start = np.array([[0., 0.], [0., 5.], [5., 5.], [5., 0.]], dtype=np.float64)  # 5x5 patch in top left corner  
end = np.array([[10, 7], [7, 10], [10, 12], [14, 8]], dtype=np.float64)

# print(np.round(cv2.getPerspectiveTransform(start, end), 2))
# returns
# [[ 0.1  -0.6  10.  ]
#  [-0.2   0.6   7.  ]
#  [-0.05  0.    1.  ]]

cv_coeffs = opencv_perspective_coeffs(start, end)
print(cv_coeffs)
M = np.zeros(9)
M[:-1] = cv_coeffs
M[-1] = 1.
M = np.float64(np.reshape(M, (3,3)))

self_warp = opencv_warp_perspective(np.ones((5,5)), M, (20, 10))



from matplotlib import pyplot as plt
image = np.zeros((20, 10))
# # plt.imshow(image, cmap='gray')
# # plt.show()

# image[:5, :5] = 1.
# plt.imshow(image, cmap='gray')
# plt.show()

cv_warp = cv2.warpPerspective(np.ones((5, 5)), M, (20, 10), flags=cv2.INTER_NEAREST)

plt.imshow(cv_warp, cmap='gray')
plt.show()

plt.imshow(self_warp, cmap='gray')
plt.show()

# pt_warp = _perspective_grid(cv_coeffs, 20, 20)

# pt_a, pt_b, coeffs = _get_perspective_coeffs(start, end)
# print(pt_a)

# matrix = np.zeros(9)
# matrix[:-1] = coeffs
# matrix[-1] = 1.
# matrix = np.round(matrix.reshape(3,3), 2)
# print(matrix)
# returns
# [[-10.  -10.  170. ]
#  [  2.5 -10.   45. ]
#  [ -0.5  -0.5   1. ]]

# print(np.linalg.inv(matrix))




# notes
# cv2 solver methods:
# 'method == DECOMP_LU || method == DECOMP_SVD || method == DECOMP_EIG || method == DECOMP_CHOLESKY || method == DECOMP_QR'