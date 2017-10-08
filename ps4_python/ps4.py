import os
import cv2
import numpy as np
import random


def grad(pic, axis=0, ksize=3, normalize=0):
    """
    pic: gray image data.
    axis: X(axis = 0) or Y(axis = 1) direction. default to X(axis = 0) direction
    return gradient Gx or Gy
    """
    ##use gaussian kernel and run derivative operators on the kernel and the
    ## use the kernel to convolve with the image
    ## all of this done in cv2.Sobel function
    if axis == 0:
        G = cv2.Sobel(pic, cv2.CV_32F, 1, 0, ksize)
    elif axis == 1:
        G = cv2.Sobel(pic, cv2.CV_32F, 0, 1, ksize)

    if normalize:
        G = cv2.normalize(G, G, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return G


def grad_angle(Gx, Gy):
    """
    :param Gx:  gradients of x direction
    :param Gy:  gradients of y direction
    :return: theta
    """
    return np.arctan2(Gy, Gx)


def grad_magnitude(Gx, Gy):
    """
    :param Gx:
    :param Gy:
    :return: squart roots of Gx^2 + Gy^2
    """
    return np.sqrt(Gx ** 2, Gy ** 2)


def grad_test(N=5):
    pic = cv2.imread("input/transA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    G_x = grad(gray, axis=0, normalize=1, ksize=N)
    G_y = grad(gray, axis=1, normalize=1, ksize=N)
    G_xy = np.hstack((G_x, G_y))
    cv2.imwrite('output/ps4-1-a-1.png', G_xy)

    pic = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    G_x = grad(gray, axis=0, normalize=1, ksize=N)
    G_y = grad(gray, axis=1, ksize=N, normalize=1)
    G_xy = np.hstack((G_x, G_y))
    cv2.imwrite('output/ps4-1-a-2.png', G_xy)


def harris_values(img, N=5, alpha=0.05, normalize=0):
    """
    :param Img: 8 bit gray image data
    :return: harris score R for each (x,y) pixel location
    """
    alpha = 0.06  # pratical value from the book
    # for each pixel in image compute M over (2N+1)*(2N+1) neighbourhood
    G_x = grad(img, axis=0, ksize=5)
    G_y = grad(img, axis=1, ksize=5)
    G_xx = G_x ** 2
    G_xy = G_x * G_y
    G_yy = G_y ** 2

    # gaussian weight
    weights = np.zeros((2 * N + 1, 2 * N + 1))
    weights[N, N] = 1
    weights = cv2.GaussianBlur(weights, weights.shape, sigmaX=1)
    # print(w)
    R = np.zeros(img.shape, dtype=float)
    Ixx = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
    Ixy = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
    Iyy = np.zeros((2 * N + 1, 2 * N + 1), dtype=float)
    for cy in range(img.shape[0]):
        for cx in range(img.shape[1]):
            Ixx.fill(0)
            Ixy.fill(0)
            Iyy.fill(0)
            min_x = max(0, cx - N)
            max_x = min(img.shape[1], cx + N + 1)
            min_y = max(0, cy - N)
            max_y = min(img.shape[0], cy + N + 1)
            x1 = 0
            x2 = 2 * N + 1
            y1 = 0
            y2 = 2 * N + 1
            ## cases to deal with points on the image boundary
            if cx - N < 0:
                x1 = N - cx
            if cy - N < 0:
                y1 = N - cy
            if cx + N + 1 > img.shape[1]:
                x2 = 2 * N + 1 - (cx + N + 1 - img.shape[1])
            if cy + N + 1 > img.shape[0]:
                y2 = 2 * N + 1 - (cy + N + 1 - img.shape[0])

            Ixx[y1:y2, x1:x2] = G_xx[min_y:max_y, min_x:max_x]
            Ixy[y1:y2, x1:x2] = G_xy[min_y:max_y, min_x:max_x]
            Iyy[y1:y2, x1:x2] = G_yy[min_y:max_y, min_x:max_x]
            Mxx = np.sum(Ixx * weights)
            Mxy = Myx = np.sum(Ixy * weights)
            Myy = np.sum(Iyy * weights)
            M = np.array([Mxx, Mxy, Myx, Myy]).reshape(2, 2)
            R[cy, cx] = np.linalg.det(M) - alpha * (M.trace() ** 2)
    # print("max and mean: ", np.max(R), np.mean(R))
    if normalize:
        R = cv2.normalize(R, R, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return R


def harris_values_test(N=5):
    pic = cv2.imread("input/TransA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    R = harris_values(gray, N, 0.05, normalize=1)
    cv2.imwrite('output/ps4-1-b-1.png', R)

    pic = cv2.imread("input/TransB.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    R = harris_values(gray, N, 0.05, normalize=1)
    cv2.imwrite('output/ps4-1-b-2.png', R)

    pic = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    R = harris_values(gray, N, 0.05, normalize=1)
    cv2.imwrite('output/ps4-1-b-3.png', R)

    pic = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    R = harris_values(gray, N, 0.05, normalize=1)
    cv2.imwrite('output/ps4-1-b-4.png', R)


def clamp(num, min, max):
    if (num < min):
        return min
    if (num > max):
        return max
    return num


def find_corners(R, N=5, threshold=10000):
    """
    :param R: harris value matrix
    :param N: neighborhood window size = (2N+1)*(2N+1).
    :param threshold
    :return: The R score with thresholded and local suppression treated
    """
    # threshold of R
    R[R < threshold] = 0
    Rs = np.zeros(R.shape, dtype=float)
    ## we could start from (N,N) to (R.height - N, R.width - N)
    for cy in range(N, R.shape[0] - N, 2 * N + 1):
        for cx in range(N, R.shape[1] - N, 2 * N + 1):
            region = R[cy - N:cy + N + 1, cx - N:cx + N + 1]
            _, maxVal, _, maxLoc = cv2.minMaxLoc(region)
            if maxVal > 0:
                # make the rest of the neighborhood all 0
                y, x = cy + maxLoc[1] - N, cx + maxLoc[0] - N  # maxLoc = (x,y) in cv
                Rs[y, x] = maxVal
                # print(maxLoc,cy,cx, y,x)
                R[cy - N:cy + N + 1, cx - N:cx + N + 1] = 0
                R[y, x] = maxVal

    return Rs


def harris_corners(pic, block_size_n=5, neighbor_size_m=5, alpha=0.05, threshold_p=0.001):
    """

    :param pic: gray picture
    :param block_size: (2*N +1)*(2*Ń+1) block size for calculating the 2x2 matrix for each pixel
    :param neighbor_size:  (2*M+1) * (2*M+1) window size for local no maxium suppresion
    :param alpha: pratical value 0.04~0.06
    :param threshold_p: % of the maxium value found in the R
    :return: harris corner pixel map r,c  pic[r,c] is a corner point
    """
    R = harris_values(pic, block_size_n, alpha)
    corners_loc = find_corners(R, neighbor_size_m, threshold_p * abs(np.amax(R)))
    return np.nonzero(corners_loc)


def harris_corners_test():
    # grad_test(N = 3)
    # harris_values_test(N = 5)
    image_list = ["input/TransA.jpg", "input/TransB.jpg", "input/simA.jpg", "input/simB.jpg"]
    i = 1
    for img in image_list:
        pic = cv2.imread(img, cv2.IMREAD_COLOR)
        copy = pic.copy()
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        r, c = harris_corners(gray, 5, 5, 0.001)
        pic[r[:], c[:]] = [0, 255, 0]
        ## compare with cv implementation
        # R = cv2.cornerHarris(gray, blockSize=11, ksize=5, k=0.05)
        # corners_loc = find_corners(R, N=5, threshold=0.001 * abs(np.amax(R)))
        # copy[corners_loc > 0] = [0, 255, 0]

        cv2.imwrite('output/ps4-1-c-' + str(i) + '.png', pic)
        # cv2.imwrite('output/ps4-1-c-cv-' + str(i) + '.png', copy)
        i = i + 1


def assign_key_points(gray):
    """
    :param gray:  gray picture
    :return: key points
    """
    r, c = harris_corners(gray, 5, 5, 0.001)
    Ix = grad(gray, axis=0)
    Iy = grad(gray, axis=1)
    Angle = grad_angle(Ix, Iy)
    Mag = grad_magnitude(Ix, Iy)

    ##assign key points
    kp = []
    for i in range(len(r)):
        # angle has to be degree. From CV docss
        # computed orientation of the keypoint (-1 if not applicable); it's in [0,360)
        # degrees and measured relative to image coordinate system, ie in clockwise
        pts = cv2.KeyPoint(c[i], r[i], _size=16, _angle=np.rad2deg(Angle[r[i], c[i]]), _octave=0)  # no scale
        kp.append(pts)

    return kp


def assign_key_points_test():
    pic_a = cv2.imread("input/TransA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    key_points = assign_key_points(gray)
    cv2.drawKeypoints(pic_a, key_points, pic_a, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic_b = cv2.imread("input/TransB.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    key_points = assign_key_points(gray)
    cv2.drawKeypoints(pic_b, key_points, pic_b, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic = np.hstack((pic_a, pic_b))
    cv2.imwrite('output/ps4-2-a-1.png', pic)

    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    key_points = assign_key_points(gray)
    cv2.drawKeypoints(pic_a, key_points, pic_a, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    key_points = assign_key_points(gray)
    cv2.drawKeypoints(pic_b, key_points, pic_b, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic = np.hstack((pic_a, pic_b))
    pic = np.hstack((pic_a, pic_b))
    cv2.imwrite('output/ps4-2-a-2.png', pic)

    ## use real opencv
    sift = cv2.xfeatures2d.SIFT_create()
    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    cv2.drawKeypoints(pic_a, kp, pic_a, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray, None)
    cv2.drawKeypoints(pic_b, kp, pic_b, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    pic = np.hstack((pic_a, pic_b))
    pic = np.hstack((pic_a, pic_b))
    cv2.imwrite('output/ps4-2-a-cv-sim.png', pic)


def putative_match(img1, img2, kp1, kp2):
    """
    :param img1:  image 1
    :param img2:  image 2
    :param kp1:  key points 1
    :param kp2:  key points 2
    :return:  matches in ascending order of distance between 2 points in kp1 and kp2
    """
    sift = cv2.xfeatures2d.SIFT_create()
    _, des_a = sift.compute(img1, kp1)
    _, des_b = sift.compute(img2, kp2)

    # brutal force match
    # http://docs.opencv.org/trunk/dc/dc3/tutorial_py_matcher.html
    # data structure of match:http://docs.opencv.org/3.1.0/d4/de0/classcv_1_1DMatch.html
    # distance, imgIdx, queryIdx, trainIdx
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_a, des_b)
    matches = sorted(matches, key=lambda x: x.distance)
    ## match is not a array structure inside
    return np.array(matches)

def draw_matches(matches,kp_a, kp_b, pic_a, pic_b,  num_drawn = 50):
    """

    :param matches: match pair
    :param kp_a:  key points of picture a
    :param kp_b:  key points of picture b
    :param pic_a:  picture a
    :param pic_b:  picture b
    :return: combined picture of a and b with match points drawm
    """
    pic = np.hstack((pic_a, pic_b))
    # just draw 50 lines. too many lines make the picture not seeable
    for i in range(len(matches[:num_drawn])):
        qIdx = matches[i].queryIdx
        tIdx = matches[i].trainIdx
        x1, y1 = kp_a[qIdx].pt
        x2, y2 = kp_b[tIdx].pt
        # adjustment pts x coordinates in hstacked picture
        x2 += pic_a.shape[1]
        cv2.line(pic, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0))

    return pic

def putative_match_test():
    pic_a = cv2.imread("input/TransA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp_a = assign_key_points(gray_a)

    pic_b = cv2.imread("input/TransB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp_b = assign_key_points(gray_b)

    matches = putative_match(gray_a, gray_b, kp_a, kp_b)
    pic = draw_matches(matches, kp_a,kp_b,pic_a, pic_b)
    cv2.imwrite('output/ps4-2-b-1.png', pic)

    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp_a = assign_key_points(gray_a)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp_b = assign_key_points(gray_b)

    matches = putative_match(gray_a, gray_b, kp_a, kp_b)
    pic = draw_matches(matches, kp_a, kp_b, pic_a, pic_b)
    cv2.imwrite('output/ps4-2-b-2.png', pic)


def sift_test():
    putative_match_test()

def ransac_parameters(n, r=0.5, P=0.1, p=0.05):
    """
    :param n   the smallest number of point required to estimate the model
    :param r  initial success rate
    :param P  rate of bad fitting % in all trials
    :param p  rate of bad fitting % in one trial
    :return: d min number of points to agree on the model
    :return: K  number of trials
    """
    d = int(np.log(p) / np.log(1 - r) + 1)
    K = int(np.log(P) / np.log(1 - r ** n) + 1)
    return d,K

def ransac_translation_transform(matches, kp_a, kp_b, theshold=100 ):
    """
    :param matches:  sorted putative matches between descriptor a and b
    :param kp_a:  key points a
    :param kp_b:  key points b
    :param theshold:
    :return:  the biggest consensus set of matches. it is index array
    """
    d, K = ransac_parameters(1,0.5, 0.1, 0.05)
    consensus = np.zeros((len(matches),K), dtype=int)
    offset = np.zeros((K,2),dtype=int)
    max_consensus = 0
    kth = 0

    pts_a = np.zeros((len(matches), 2), dtype = int)
    pts_b = np.zeros((len(matches), 2), dtype = int)

    #extract all points from keypoints
    for i in range(len(matches)):
        pts_a[i] = kp_a[matches[i].queryIdx].pt
        pts_b[i] = kp_b[matches[i].trainIdx].pt

    print("%d trials, %d consensus agreeing points" %(K, d))

    for num in range(K):
        mask = np.random.choice(len(matches), 1, replace=False)
        #set = matches[mask[0]]
        # use the 1st pair in the set to caculate translation x,y
        offset[num] = np.array(pts_b[mask] - pts_a[mask])

        # use the rest of pairs to decide fitting
        # criteria = ssd((pts_a_x + offset_x - pts_b_x)^2 +(pts_a_y + offset_y - pts_b_y))
        # array mask of all agreement points in the match set for kth trials
        consensus[:,num] = np.sum((pts_a + offset[num] - pts_b)**2, axis=1)< theshold
        if sum(consensus[:,num])> d and  max_consensus < sum(consensus[:,num]):
            max_consensus = sum(consensus[:,num])
            kth = num
    print("max_consensus :%d  rate: %f "% (max_consensus, max_consensus/len(matches)))
    print("offset :", offset[kth])
    return np.nonzero(consensus[:,kth]), offset[kth]

def ransac_translation_test():
    print("ransac fitting translation")
    pic_a = cv2.imread("input/TransA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp_a = assign_key_points(gray_a)

    pic_b = cv2.imread("input/TransB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp_b = assign_key_points(gray_b)

    matches = np.array(putative_match(gray_a, gray_b, kp_a, kp_b))
    max_censensus, _= ransac_translation_transform(matches, kp_a, kp_b, theshold=50)
    matches = matches[max_censensus]
    pic = draw_matches(matches, kp_a, kp_b, pic_a, pic_b)
    cv2.imwrite('output/ps4-3-a-1.png', pic)


def ransac_similarity_transform(matches, kp_a, kp_b, theshold=100):
    # 2 points to estimate a similarity transform M 2x3

    d, K = ransac_parameters(2, 0.5, 0.05, 0.01)
    consensus = np.zeros((len(matches),K), dtype=int)
    A_t = np.zeros((2,3), dtype = float)
    max_consensus = 0
    kth = 0

    pts_a = np.zeros((len(matches), 2), dtype = int)
    pts_b = np.zeros((len(matches), 2), dtype = int)

    #extract all points from keypoints
    for i in range(len(matches)):
        pts_a[i] = kp_a[matches[i].queryIdx].pt
        pts_b[i] = kp_b[matches[i].trainIdx].pt

    print("%d trials, %d consensus agreeing points" %(K, d))

    for num in range(K):
        mask = np.random.choice(len(matches), 2, replace=False)
        M = np.zeros((4, 4))
        H = np.zeros(4)

        pts_a_2 = pts_a[mask]
        pts_b_2 = pts_b[mask]
        for i in range(2):
            M[2*i] = [pts_a_2[i][0], -pts_a_2[i][1], 1, 0 ]
            M[2*i+1] = [pts_a_2[i][1], pts_a_2[i][0], 0, 1]
            H[2*i] = pts_b_2[i][0]
            H[2 * i+1 ] = pts_b_2[i][1]

        x = np.linalg.solve(M,H)
        #print(x)
        # A = [ x0   -x1    x2 ]
        #     [ x1    x0    x3 ]
        A = np.array([ x[0], -x[1], x[2] , x[1], x[0], x[3] ]).reshape(2,3)

        # use the rest of pairs to decide fitting
        # array mask of all agreement points in the match set for kth trials
        # up to now, pts_a = [ n x 2] make it 3 x n for the matrix muliplication
        pts_aa = np.vstack((pts_a.T, np.ones(pts_a.shape[0])))
        consensus[:,num] = np.sum(((np.dot(A, pts_aa)).T - pts_b)**2, axis=1)< theshold
        if sum(consensus[:,num])> d and  max_consensus < sum(consensus[:,num]):
            max_consensus = sum(consensus[:,num])
            A_t = A
            kth = num
    print("max_consensus :%d  rate: %f "% (max_consensus, max_consensus/len(matches)))
    print("similarity transform  :\n", A_t)
    return np.nonzero(consensus[:,kth]), A_t

def ransac_similarity_transform_test():
    print("ransac  fitting similarity ")
    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp_a = assign_key_points(gray_a)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp_b = assign_key_points(gray_b)

    matches = np.array(putative_match(gray_a, gray_b, kp_a, kp_b))
    max_censensus, _= ransac_similarity_transform(matches, kp_a, kp_b, theshold=50)
    matches = matches[max_censensus]
    pic = draw_matches(matches, kp_a, kp_b, pic_a, pic_b)
    cv2.imwrite('output/ps4-3-b-1.png', pic)

def ransac_affine_transform(matches, kp_a, kp_b, theshold=100):
    # 3 points to estimate an affine transform M 2x3
    n = 3
    d, K = ransac_parameters(n, 0.5,0.05, 0.01) #0.01, 0.005)
    consensus = np.zeros((len(matches),K), dtype=int)
    A_t = np.zeros((2,3), dtype = float)
    max_consensus = 0
    kth = 0

    pts_a = np.zeros((len(matches), 2), dtype = int)
    pts_b = np.zeros((len(matches), 2), dtype = int)

    #extract all points from keypoints
    for i in range(len(matches)):
        pts_a[i] = kp_a[matches[i].queryIdx].pt
        pts_b[i] = kp_b[matches[i].trainIdx].pt

    print("%d trials, %d consensus agreeing points" %(K, d))

    for num in range(K):
        mask = np.random.choice(len(matches), n, replace=False)
        #M = np.zeros((4,5))
        M = np.zeros((2*n, 2*n))
        H = np.zeros(2*n)

        pts_a_2 = pts_a[mask]
        pts_b_2 = pts_b[mask]

        for i in range(n):
            M[2*i] = [pts_a_2[i][0], pts_a_2[i][1], 1, 0, 0, 0 ]
            M[2*i+1] = [0, 0, 0, pts_a_2[i][0], pts_a_2[i][1], 1]
            H[2*i] = pts_b_2[i][0]
            H[2 * i+1 ] = pts_b_2[i][1]

        x = np.linalg.solve(M,H)
        #print(x)
        # A = [ x0   x1    x2 ]
        #     [ x3   x4    x5 ]
        A = np.array([ x[0], x[1], x[2] , x[3], x[4], x[5] ]).reshape(2,3)
        # use the rest of pairs to decide fitting
        # array mask of all agreement points in the match set for kth trials
        # up to now, pts_a = [ n x 2] make it 3 x n for the matrix muliplication
        pts_aa = np.vstack((pts_a.T, np.ones(pts_a.shape[0])))
        consensus[:,num] = np.sum(((np.dot(A, pts_aa)).T - pts_b)**2, axis=1)< theshold
        if sum(consensus[:,num])> d and  max_consensus < sum(consensus[:,num]):
            max_consensus = sum(consensus[:,num])
            A_t = A
            kth = num
    print("max_consensus :%d  rate: %f "% (max_consensus, max_consensus/len(matches)))
    print("affine transform  :\n", A_t)
    return np.nonzero(consensus[:,kth]), A_t

def ransac_affine_transform_test():
    print("ransac  fitting affine ")
    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)
    kp_a = assign_key_points(gray_a)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    kp_b = assign_key_points(gray_b)

    matches = np.array(putative_match(gray_a, gray_b, kp_a, kp_b))
    max_censensus, _= ransac_affine_transform(matches, kp_a, kp_b, theshold=100)
    matches = matches[max_censensus]
    pic = draw_matches(matches, kp_a, kp_b, pic_a, pic_b)
    cv2.imwrite('output/ps4-3-c-1.png', pic)

def warp( pic_a, pic_b, transform = 0):
    """

    :param transform:  transform method similarity = 0, affine = 1
    :param pic_a: pic gray a
    :param pic_b: pic a
    :return: warpped picture,   picture with orignal picture blended with warpped image
    """
    kp_a = assign_key_points(pic_a)
    kp_b = assign_key_points(pic_b)

    matches = np.array(putative_match(pic_a, pic_b, kp_a, kp_b))
    if transform==1:
        _, transform = ransac_affine_transform(matches, kp_a, kp_b, theshold=50)
    else:
        _, transform = ransac_similarity_transform(matches, kp_a, kp_b, theshold=50)
    # warp b to a through inverse map
    warpped = cv2.warpAffine(pic_b, transform, pic_b.shape[1::-1],flags=cv2.WARP_INVERSE_MAP)
    blended = warpped * 0.5
    blended[:pic_a.shape[0], :pic_a.shape[1]] += pic_a * 0.5
    blend = cv2.normalize(blended, blended, alpha=0, beta=255,
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return warpped, blended


def warp_test():
    print("ransac  warping ")
    pic_a = cv2.imread("input/simA.jpg", cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(pic_a, cv2.COLOR_BGR2GRAY)

    pic_b = cv2.imread("input/simB.jpg", cv2.IMREAD_COLOR)
    gray_b = cv2.cvtColor(pic_b, cv2.COLOR_BGR2GRAY)
    wrapped, blended = warp(gray_a, gray_b)
    cv2.imwrite('output/ps4-3-d-1.png', wrapped)
    cv2.imwrite('output/ps4-3-d-2.png', blended)

    wrapped, blended = warp(gray_a, gray_b,transform=1)
    cv2.imwrite('output/ps4-3-e-1.png', wrapped)
    cv2.imwrite('output/ps4-3-e-2.png', blended)

    ##affine transform seem to be a better choice it does better in estimate
    ## the offset x y


def ransac_test():
    #ransac_translation_test()
    #ransac_similarity_transform_test()
    #ransac_affine_transform_test()
    warp_test()

if __name__ == "__main__":
    import argparse

    input_dir = "./input"
    output_dir = "./output"

    parser = argparse.ArgumentParser(description="PS1")
    parser.add_argument("-test", "--test", dest="test_item",
                        help="test item(harris_corners/sift/ransac/all)",
                        action="store")

    args = parser.parse_args()
    test_item = args.test_item
    print("testing " + test_item)

    if test_item == "harris_corners":
        harris_corners_test()
    elif test_item == "sift":
        sift_test()
    elif test_item == "ransac":
        ransac_test()
    else:
        print("running all tests")
        harris_corners_test()
        sift_test()
        ransac_test()
