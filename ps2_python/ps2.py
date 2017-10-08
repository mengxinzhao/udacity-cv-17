# ps2
import os
import cv2
from disparity_ssd import  *
from disparity_ncorr import *

def basic_disparity():

    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1.0 / 255.0)  # grayscale, [0, 1]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1.0 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)

    D_L = disparity_ssd(L, R, size = 11, disparity = 15)
    D_R = disparity_ssd(R, L, size = 11, disparity = 15)

    # TODO: Save output images (D_L as output/ps2-1-a-1.png and D_R as output/ps2-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-1-a-1.png", D_L)
    cv2.imwrite("output/ps2-1-a-2.png", D_R)


def real_image_disparity():
    ## 2
    # Read images
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    # smaller windows would pick up too much details
    D_L = np.abs(disparity_ssd(L, R, size=11, disparity =70))
    D_R = np.abs(disparity_ssd(R, L, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-2-a-1.png", D_L)
    cv2.imwrite("output/ps2-2-a-2.png", D_R)

def ssd_under_noise():
    # Read images
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)

    ## apply guassian noise
    mean = 0
    sigma = 0.2
    gauss = np.random.normal(mean, sigma, (L.shape[0], L.shape[1]))
    gauss = gauss.reshape(L.shape[0], L.shape[1])
    L_gauss = L + gauss
    #output_L_gauss = cv2.normalize(L_gauss, L_gauss, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imwrite("output/ps2-3-a-gauss.png", output_L_gauss)

    D_L = np.abs(disparity_ssd(L_gauss, R, size=11, disparity =70))
    D_R = np.abs(disparity_ssd(R, L_gauss, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-3-a-1.png", D_L)
    cv2.imwrite("output/ps2-3-a-2.png", D_R)

    ##apply sharpening filter (unsharpening mask) on origial L
    smoothed = cv2.GaussianBlur(L, (5, 5), 0.1)
    L_sharpened = cv2.addWeighted(L,1.5, smoothed,-0.5, 1.1)
    ## has to normalize again to make sure the value in [0,1]
    L_sharpened = cv2.normalize(L_sharpened, L_sharpened, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    #output_L_sharpened = cv2.normalize(L_sharpened, L_sharpened, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    #cv2.imwrite("output/ps2-3-a-sharpened.png", output_L_sharpened)

    D_L = np.abs(disparity_ssd(L_sharpened, R, size=11, disparity =70))
    D_R = np.abs(disparity_ssd(R, L_sharpened, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-3-b-1.png", D_L)
    cv2.imwrite("output/ps2-3-b-2.png", D_R)

def correlation_disparity():
    # Read images
    L = cv2.imread(os.path.join('input', 'pair1-L.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), cv2.IMREAD_GRAYSCALE) * (1.0 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    # smaller windows would pick up too much details
    D_L = np.abs(disparity_ncorr(L, R, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R, L, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-4-a-1.png", D_L)
    cv2.imwrite("output/ps2-4-a-2.png", D_R)

    ## apply guassian noise
    mean = 0
    sigma = 0.2
    gauss = np.random.normal(mean, sigma, (L.shape[0], L.shape[1]))
    gauss = gauss.reshape(L.shape[0], L.shape[1])
    L_gauss = L + gauss

    D_L = np.abs(disparity_ncorr(L_gauss, R, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R, L_gauss, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-4-b-1.png", D_L)
    cv2.imwrite("output/ps2-4-b-2.png", D_R)

    ##apply sharpening filter (unsharpening mask) on origial L
    smoothed = cv2.GaussianBlur(L, (5, 5), 0.1)
    L_sharpened = cv2.addWeighted(L,1.5, smoothed,-0.5, 1.1)
    L_sharpened = cv2.normalize(L_sharpened, L_sharpened, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    D_L = np.abs(disparity_ncorr(L_sharpened, R, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R, L_sharpened, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-4-b-3.png", D_L)
    cv2.imwrite("output/ps2-4-b-4.png", D_R)

def laundry_pair():
    # Read images
    L = cv2.imread(os.path.join('input', 'pair2-L.png'), cv2.IMREAD_GRAYSCALE)* (1.0 / 255.0)
    R = cv2.imread(os.path.join('input', 'pair2-R.png'), cv2.IMREAD_GRAYSCALE)* (1.0 / 255.0)

    D_L = np.abs(disparity_ncorr(L, R, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R, L, size=11, disparity =70))

    cv2.imwrite("output/ps2-5-a-1.png", D_L)
    cv2.imwrite("output/ps2-5-a-2.png", D_R)

    ## apply guassian noise on both images one has more noise
    mean = 0
    sigma = 0.2
    gauss = np.random.normal(mean, sigma, (L.shape[0], L.shape[1]))
    gauss = gauss.reshape(L.shape[0], L.shape[1])
    L_gauss = L + gauss

    sigma = 0.1
    gauss = np.random.normal(mean, sigma, (R.shape[0], R.shape[1]))
    gauss = gauss.reshape(R.shape[0], R.shape[1])
    R_gauss = R + gauss

    D_L = np.abs(disparity_ncorr(L_gauss, R_gauss, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R_gauss, L_gauss, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-5-a-3.png", D_L)
    cv2.imwrite("output/ps2-5-a-4.png", D_R)

    ##apply sharpening filter (unsharpening mask) on origial L
    smoothed = cv2.GaussianBlur(L, (5, 5), 0.1)
    L_sharpened = cv2.addWeighted(L,1.5, smoothed,-0.5, 1.1)
    L_sharpened = cv2.normalize(L_sharpened, L_sharpened, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    D_L = np.abs(disparity_ncorr(L_sharpened, R, size=11, disparity =70))
    D_R = np.abs(disparity_ncorr(R, L_sharpened, size=11, disparity =70))

    # Note: They may need to be scaled/shifted before saving to show results properly there are negative values in the
    # D array
    D_L = cv2.normalize(D_L, D_L, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    D_R = cv2.normalize(D_R, D_R, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    cv2.imwrite("output/ps2-5-a-5.png", D_L)
    cv2.imwrite("output/ps2-5-a-6.png", D_R)


if __name__ == "__main__":
    import argparse

    input_dir = "./input"
    output_dir = "./output"

    parser = argparse.ArgumentParser(description="PS1")
    parser.add_argument("-test", "--test", dest="test_item",
                        help="test item(basic_disparity/real_image_disparity/ssd_under_noise/all)",
                        action="store")

    args = parser.parse_args()
    test_item = args.test_item
    print("testing " + test_item)

    if test_item == "basic_disparity":
        basic_disparity()
    elif test_item =="real_image_disparity":
        real_image_disparity()
    elif test_item =="ssd_under_noise":
        ssd_under_noise()
    elif test_item == "correlation_disparity":
        correlation_disparity()
    elif test_item == "laundry_pair":
        laundry_pair()
    else:
        print("running all tests")
        basic_disparity()
        real_image_disparity()
        ssd_under_noise()
        correlation_disparity()
        laundry_pair()