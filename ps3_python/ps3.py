import os
import cv2
import numpy as np
import random

M_norm_a = np.array([[-0.4583, 0.2947, 0.0139, -0.0040],
                     [0.0509, 0.0546, 0.5410, 0.0524],
                     [-0.1090, -0.1784, 0.0443, -0.5968]], dtype=np.float32)
C_norm_a = np.array([-1.5125, -2.3515, 0.2826], dtype=np.float32)


def SVD_M(pts_2d, pts_3d, test_2d, test_3d):
    # calculate M and res
    N = int(pts_2d.shape[0])
    P = np.zeros((N * 2, 12))
    res = 0
    for i in range(N):
        P[2 * i] = [pts_3d[i, 0], pts_3d[i, 1], pts_3d[i, 2], 1,
                    0, 0, 0, 0,
                    -pts_2d[i, 0] * pts_3d[i, 0], -pts_2d[i, 0] * pts_3d[i, 1], -pts_2d[i, 0] * pts_3d[i, 2],
                    -pts_2d[i, 0]]
        P[2 * i + 1] = [0, 0, 0, 0,
                        pts_3d[i, 0], pts_3d[i, 1], pts_3d[i, 2], 1,
                        -pts_2d[i, 1] * pts_3d[i, 0], -pts_2d[i, 1] * pts_3d[i, 1], -pts_2d[i, 1] * pts_3d[i, 2],
                        -pts_2d[i, 1]]
    U, S, V = np.linalg.svd(P, full_matrices=True)
    M = V[11].reshape(3, 4)
    # converts to homogenous coordinates
    test_3d = np.concatenate((test_3d.T, np.ones((1, test_3d.shape[0]))))
    pts_2d_proj = np.dot(M, test_3d).T
    # print(pts_2d_proj)
    pts_2d_proj[:, 0] = pts_2d_proj[:, 0] / pts_2d_proj[:, 2]
    pts_2d_proj[:, 1] = pts_2d_proj[:, 1] / pts_2d_proj[:, 2]

    for i in range(test_2d.shape[0]):
        res += np.linalg.norm(test_2d[i] - pts_2d_proj[i, :2])
    res = res / test_2d.shape[0]
    return M, res


def SVD_F(pts_a, pts_b):
    # calculate F 3x3
    # pts_a , pts_b N x 3 array
    N = int(pts_a.shape[0])
    P = np.zeros((N, 9))
    for i in range(N):
        # P[i] = [ pts_a[i,0]* pts_b[i,0], pts_a[i,0] * pts_b[i,1], pts_a[i,0],
        #         pts_a[i,1]* pts_b[i,0], pts_a[i,1] * pts_b[i,1], pts_a[i,1],
        #         pts_b[i,0], pts_b[i,1], 1 ]
        P[i] = [pts_a[i, 0] * pts_b[i, 0], pts_a[i, 1] * pts_b[i, 0], pts_b[i, 0],
                pts_a[i, 0] * pts_b[i, 1], pts_a[i, 1] * pts_b[i, 1], pts_b[i, 1],
                pts_a[i, 0], pts_a[i, 1], 1]

    U, S, V = np.linalg.svd(P)
    F = V[8].reshape(3, 3)

    return F


def basic_calibration():
    # 1. Create the least squares function that will solve for the 3x4 matrix MnormA
    # 2. verification: given pts2d-norm-pic_a.txt and pts3d-norm.txt.
    # multiplying pts3d-norm points by your M matrix and comparing the resulting the
    # normalized 2D points to the normalized 2D points given in the file

    points_2d = np.genfromtxt("input/pts2d-norm-pic_a.txt", dtype=float, delimiter="")
    points_3d = np.genfromtxt("input/pts3d-norm.txt", dtype=float, delimiter="")

    if points_3d.shape[0] != points_2d.shape[0]:
        raise RuntimeError("number of points from 3d to 2d don't match")
    # print(points_3d)

    # converts to homogenous coordinates
    # data_2d = np.concatenate((points_2d,np.ones((1,points_2d.shape[0])).T), axis = -1)
    # data_3d = np.concatenate((points_3d, np.ones((1, points_3d.shape[0])).T), axis=-1)

    # create matrix for linear method, 2 rows for each correspondence pair
    N = int(points_2d.shape[0])
    P = np.zeros((N * 2, 12))
    for i in range(N):
        P[2 * i] = [points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 1,
                    0, 0, 0, 0,
                    -points_2d[i, 0] * points_3d[i, 0], -points_2d[i, 0] * points_3d[i, 1],
                    -points_2d[i, 0] * points_3d[i, 2], -points_2d[i, 0]]
        P[2 * i + 1] = [0, 0, 0, 0,
                        points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 1,
                        -points_2d[i, 1] * points_3d[i, 0], -points_2d[i, 1] * points_3d[i, 1],
                        -points_2d[i, 1] * points_3d[i, 2],
                        -points_2d[i, 1]]
    U, S, V = np.linalg.svd(P, full_matrices=True)
    M = V[11].reshape(3, 4)
    print("Calculated project matrix:\n", M)
    print("expected project matrix:\n", M_norm_a)

    # verification on last normalized 3d points given in the file
    # < 1.2323, 1.4421, 0.4506, 1.0 > and will project it to the < u, v > of < 0.1419, −0.4518 >
    pt_2d_proj = np.dot(M, np.append(points_3d[-1], 1))
    pt_2d_proj = pt_2d_proj[:2] / pt_2d_proj[2]
    print("projected 2d point: ", pt_2d_proj)
    res = np.linalg.norm(points_2d[-1] - pt_2d_proj)
    print("residual: ", res)


def camera_calibration():
    points_2d = np.genfromtxt("input/pts2d-pic_b.txt", dtype=float, delimiter="")
    points_3d = np.genfromtxt("input/pts3d.txt", dtype=float, delimiter="")

    # For the three point set sizes k of 8, 12, and 16, repeat 10 times:
    # - Randomly choose k points from the 2D list and their corresponding points in the 3D list.
    # - Compute the projection matrix M on the chosen points.
    # - Pick 4 points not in your set of k and compute the average residual.
    # - Save the M that gives the lowest residual.
    # - esitmation camera center

    repeat = 10
    k = [12, 16, 20]  ## the last 4 points for computing residual
    res = np.finfo(np.float).max
    M = np.zeros((3, 4))
    for num in k:
        for count in range(repeat + 1):
            # print(num)
            row_mask = np.random.choice(points_3d.shape[0], num, replace=False)
            pts_3d = points_3d[row_mask, :]
            pts_2d = points_2d[row_mask, :]
            _M, _res = SVD_M(pts_2d[:num - 4, :], pts_3d[:num - 4, :], pts_2d[-5:-1, :], pts_3d[-5:-1, :])
            print("%d points, repeat %d res: %f" % (num - 4, count, _res))
            if _res < res:
                M = _M
                res = _res

    ## the lowest residual point almost always come from 8 or 12 point estimation?
    print("the lowest residual:%f \n" % res)
    print("estimated project matrix: \n", M)
    # print("expected project matrix:\n", M_norm_a)

    center = np.zeros(3)
    center = - np.dot(np.linalg.inv(M[:, :3]), M[:, 3])
    print("estimated camera center:\n", center)
    # print("expected camera center:\n", C_norm_a)


def draw_epiline(pts_a, pts_b, pic_a, pic_b, F):
    # converts points to homogenous coordinates
    pts_a = np.concatenate((pts_a.T, np.ones((1, pts_a.shape[0]))))
    pts_b = np.concatenate((pts_b.T, np.ones((1, pts_b.shape[0]))))
    epi_lb = np.dot(F, pts_a).T
    epi_la = np.dot(F.T, pts_b).T
    n = pic_a.shape[0]
    m = pic_a.shape[1]
    pts_ul = [0, 0, 1]
    pts_bl = [0, n, 1]
    pts_ur = [m, 0, 1]
    pts_br = [m, n, 1]
    lL = np.cross(pts_ul, pts_bl)
    lR = np.cross(pts_ur, pts_br)

    pa_L = np.cross(epi_la, lL)
    pa_L[:, 0] = pa_L[:, 0] / pa_L[:, 2]
    pa_L[:, 1] = pa_L[:, 1] / pa_L[:, 2]
    pa_R = np.cross(epi_la, lR)
    pa_R[:, 0] = pa_R[:, 0] / pa_R[:, 2]
    pa_R[:, 1] = pa_R[:, 1] / pa_R[:, 2]

    pb_L = np.cross(epi_lb, lL)
    pb_L[:, 0] = pb_L[:, 0] / pb_L[:, 2]
    pb_L[:, 1] = pb_L[:, 1] / pb_L[:, 2]
    pb_R = np.cross(epi_lb, lR)
    pb_R[:, 0] = pb_R[:, 0] / pb_R[:, 2]
    pb_R[:, 1] = pb_R[:, 1] / pb_R[:, 2]

    for i in range(epi_la.shape[0]):
        cv2.line(pic_a, (int(pa_L[i, 0]), int(pa_L[i, 1])), (int(pa_R[i, 0]), int(pa_R[i, 1])), (0, 255, 0),
                 thickness=1)
        cv2.line(pic_b, (int(pb_L[i, 0]), int(pb_L[i, 1])), (int(pb_R[i, 0]), int(pb_R[i, 1])), (0, 255, 0),
                 thickness=1)

def fundamental_matrix_estimation():
    pts_a = np.genfromtxt("input/pts2d-pic_a.txt", dtype=float, delimiter="")
    pts_b = np.genfromtxt("input/pts2d-pic_b.txt", dtype=float, delimiter="")

    if pts_a.shape[0] != pts_b.shape[0]:
        raise RuntimeError("number of points from 3d to 2d don't match")

    F = SVD_F(pts_a, pts_b)
    ## why am I still getting rank 3 by SVD method?
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    S = np.diag(S)
    F = np.dot(np.dot(U, S), V)
    print("F:\n",F)
    print("rank : ", np.linalg.matrix_rank(F))

    pic_a = cv2.imread("input/pic_a.jpg", cv2.IMREAD_COLOR)
    pic_b = cv2.imread("input/pic_b.jpg", cv2.IMREAD_COLOR)

    draw_epiline(pts_a, pts_b, pic_a, pic_b, F)
    cv2.imwrite('output/ps3-2-c-1.png', pic_a)
    cv2.imwrite('output/ps3-2-c-2.png', pic_b)

def normalize_matrix(pts):
    ## strictly following the problem set description
    Cu = np.empty(pts.shape[0])
    Cv = np.empty(pts.shape[0])
    Cu.fill(np.mean(pts[:, 0]))  # mean of x
    Cv.fill(np.mean(pts[:, 1]))  # mean of Y
    m_pts_a = np.column_stack((Cu, Cv))
    deviation  = np.std(pts-m_pts_a)
    Ta = np.zeros((3,3),float)
    np.fill_diagonal(Ta, 1/deviation)
    Ta[2,2] = 1
    Tb = np.zeros((3,3),float)
    np.fill_diagonal(Tb, 1)
    Tb[0,2] = -np.mean(pts[:, 0])
    Tb[1,2] = -np.mean(pts[:, 1])
    T = np.dot(Ta, Tb)
    #print(Ta)
    #print(Tb)
    #print(T)
    return T

def normalize_matrix2(pts):
    ## CV2's way borrowed from Programming Computer Vision with Python
    c = np.mean(pts,axis =0)  #[ mean_x ,mean_y]
    maxstd = np.max(np.std(pts,axis=0 )) # max of [std_x, std_y]
    Ta = np.zeros((3,3),float)
    Ta[0,0] = Ta[1,1] =  1/maxstd
    Ta[2,2] = 1
    Tb = np.zeros((3,3),float)
    np.fill_diagonal(Tb, 1)
    Tb[0,2] = -c[0]
    Tb[1,2] = -c[1]
    T = np.dot(Ta, Tb)

    return T


if __name__ == '__main__':
    def fundamental_matrix_estimation2():
        pts_a = np.genfromtxt("input/pts2d-pic_a.txt", dtype=float, delimiter="")
        pts_b = np.genfromtxt("input/pts2d-pic_b.txt", dtype=float, delimiter="")


        if pts_a.shape[0] != pts_b.shape[0]:
            raise RuntimeError("number of points from 3d to 2d don't match")

        # normalize  points first before going through F estimation
        Ta = normalize_matrix2(pts_a)
        Tb = normalize_matrix2(pts_b)
        print("Ta:\n",Ta)
        print("Tb:\n",Tb)

        pts_homogenous_a = np.concatenate((pts_a.T, np.ones((1, pts_a.shape[0]))))
        pts_homogenous_b = np.concatenate((pts_b.T, np.ones((1, pts_b.shape[0]))))

        normalized_pts_a = np.dot(Ta, pts_homogenous_a).T
        normalized_pts_b = np.dot(Tb, pts_homogenous_b).T

        #print(normalized_pts_a)
        _F = SVD_F(normalized_pts_a, normalized_pts_b)
        if (np.linalg.matrix_rank(_F) == 3):
            U, S, V = np.linalg.svd(_F)
            S[-1] = 0
            S = np.diag(S)
            _F = np.dot(np.dot(U, S), V)
        print("_F:\n", _F)

        # the improved F
        F = np.dot(Tb.T,np.dot(_F,Ta))
        print("Final F:\n",F)
        print("rank : ", np.linalg.matrix_rank(F))

        pic_a = cv2.imread("input/pic_a.jpg", cv2.IMREAD_COLOR)
        pic_b = cv2.imread("input/pic_b.jpg", cv2.IMREAD_COLOR)
        draw_epiline(pts_a, pts_b,pic_a,pic_b,F)
        cv2.imwrite('output/ps3-2-e-1.png', pic_a)
        cv2.imwrite('output/ps3-2-e-2.png', pic_b)
        # the output looks better ps2-2-c-1 the first line  doesn't  go through the checkboard
        # with the improved F ps3-2-e-1 the first line goes through checkboard

if __name__ == "__main__":
    import argparse

    input_dir = "./input"
    output_dir = "./output"

    parser = argparse.ArgumentParser(description="PS1")
    parser.add_argument("-test", "--test", dest="test_item",
                        help="test item(basic_calibration/camera_calibration/fundamental_matrix_estimation/fundamental_matrix_estimation2/all)",
                        action="store")

    args = parser.parse_args()
    test_item = args.test_item
    print("testing " + test_item)

    if test_item == "basic_calibration":
        basic_calibration()
    elif test_item == "camera_calibration":
        camera_calibration()
    elif test_item == "fundamental_matrix_estimation":
        fundamental_matrix_estimation()
    elif test_item == "fundamental_matrix_estimation2":
        fundamental_matrix_estimation2()
    else:
        print("running all tests")
        basic_calibration()
        camera_calibration()
        fundamental_matrix_estimation()
        fundamental_matrix_estimation2()
