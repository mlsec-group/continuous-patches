import numpy as np
import torch
import cv2

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
    res = torch.linalg.lstsq(a_matrix, b_matrix, driver="gels").solution.to(torch.float32)

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

coeffs = _get_perspective_coeffs(start, end)

matrix = np.zeros(9)
matrix[:-1] = coeffs
matrix[-1] = 1.
# print(np.round(matrix.reshape(3,3), 2))
# [[ 1.  0.  -0.]
#  [ 0.  1. -10.]
#  [ 0.  0.  1.]]

start = np.array([[0., 0.], [0., 5.], [5., 5.], [5., 0.]], dtype=np.float32)  # 5x5 patch in top left corner  
end = np.array([[10, 7], [7, 10], [10, 12], [14, 8]], dtype=np.float32)

print(np.round(cv2.getPerspectiveTransform(start, end), 2))
# returns
# [[ 0.1  -0.6  10.  ]
#  [-0.2   0.6   7.  ]
#  [-0.05  0.    1.  ]]

coeffs = _get_perspective_coeffs(start, end)

matrix = np.zeros(9)
matrix[:-1] = coeffs
matrix[-1] = 1.
matrix = np.round(matrix.reshape(3,3), 2)
print(matrix)
# returns
# [[-10.  -10.  170. ]
#  [  2.5 -10.   45. ]
#  [ -0.5  -0.5   1. ]]

# print(np.linalg.inv(matrix))




# notes
# cv2 solver methods:
# 'method == DECOMP_LU || method == DECOMP_SVD || method == DECOMP_EIG || method == DECOMP_CHOLESKY || method == DECOMP_QR'