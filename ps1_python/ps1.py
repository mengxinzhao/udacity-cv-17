import numpy as np
import matplotlib.pyplot as plt
## from this lesson on I'm switching opencv2 to do the homework
import cv2


def save_image_to_file(image, file_name):
    print("saving to " + file_name)
    cv2.imwrite(file_name, image)

def canny(img, sigma=0.33, kernel_size=3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    low = (1 - sigma) * mean
    high = max(255, (1 + sigma) * mean)
    edges = cv2.Canny(img, low, high, apertureSize=kernel_size)
    return edges

def hough_lines_acc(edges, rho_res=1, thetas=np.arange(0, 180, 1)):
    ## Write a function hough_lines_acc that computes the Hough Transform for lines and produces an accumulator array.
    ## Output: Store the hough accumulator array (H) as ps1-2-a-1.png (note: write a normalized uint8 version of the
    ## array so that the minimum value is mapped to 0 and maximum to 255).
    ## H[rho][theta] =  hough_lines_acc(edges)
    ## H[rho][theta] =  hough_lines_acc(edges, rho_res, theta_res)


    # diagonal length of the image 2-norm
    rho_max = int(round(np.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2)))
    rhos = np.arange(-rho_max, rho_max, rho_res)
    if min(thetas) < 0:
        thetas -= min(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    yaxies, xaxies = np.nonzero(edges)  ## get the edge point [y,x]
    for index in range(len(yaxies)):
        x = xaxies[index]
        y = yaxies[index]
        d = x * np.cos(np.deg2rad(thetas)) + y * np.sin(np.deg2rad(thetas))  ##cos/sin needs to be radians
        d = np.round((d + rho_max) / rho_res)

        ## get rid of the out-of-boundary value.
        m, n = accumulator.shape
        valid_idxs = np.nonzero((d < m) & (thetas < n))
        d = d[valid_idxs]
        temp_thetas = thetas[valid_idxs]

        ## form a matrx of (rho, theta)
        rho_theta = np.stack([d, temp_thetas], 1)

        ## remove duplicated rows and compute the counts for each (rho, theta) pair
        binary_form = np.ascontiguousarray(rho_theta).view(
            np.dtype((np.void, rho_theta.dtype.itemsize * rho_theta.shape[1])))
        unique_vals, idxs, counts = np.unique(binary_form, return_index=True, return_counts=True)
        uni_rho_theta = rho_theta[idxs].astype(np.uint)
        accumulator[uni_rho_theta[:, 0], (uni_rho_theta[:, 1])] += counts.astype(np.uint)

    accumulator = cv2.normalize(accumulator, accumulator, 0, 255,
                                cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return accumulator, thetas, rhos

def hough_circles_acc(edges, radius):
    ## voting for the center of the circle
    accumulator = np.zeros((edges.shape[0], edges.shape[1]), dtype=np.uint8)
    thetas = np.arange(0, 360, 1)
    yaxies, xaxies = np.nonzero(edges)  ## get the edge point [y,x]

    for index in range(len(yaxies)):
        x = xaxies[index]
        y = yaxies[index]
        x0 = np.round(x - radius * np.cos(np.deg2rad(thetas)))
        y0 = np.round(y - radius * np.sin(np.deg2rad(thetas)))

        ## get rid of the out-of-boundary value.
        m, n = accumulator.shape
        valid_idxs = np.nonzero((y0 < m) & (x0 < n))
        y0 = y0[valid_idxs]
        x0 = x0[valid_idxs]

        ## form a matrx of (y0, x0)
        x_y = np.stack([y0, x0], 1)

        ## remove duplicated rows and compute the counts for each (x0, y0) pair
        binary_form = np.ascontiguousarray(x_y).view(
            np.dtype((np.void, x_y.dtype.itemsize * x_y.shape[1])))
        unique_vals, idxs, counts = np.unique(binary_form, return_index=True, return_counts=True)
        uni_x_y = x_y[idxs].astype(np.uint)
        accumulator[uni_x_y[:, 0], uni_x_y[:, 1]] += counts.astype(np.uint)

    return accumulator

def clamp(num, min, max):
    if (num < min):
        return min
    if (num > max):
        return max
    return num

def hough_peaks(accumulator, num_peaks=10, threshold=100.0, n_hood_size=50):
    ##finds indices of the accumulator array (here line parameters) that correspond to local maxima
    ##return : Qx2 matrix with row indices (here rho) in column 1, and column indices (here theta) in column 2.
    peak_loc = np.zeros((num_peaks, 2), dtype=np.int)
    _accumulator = accumulator.copy()
    height = _accumulator.shape[0]
    width = _accumulator.shape[1]
    for i in range(num_peaks):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(_accumulator)
        if max_val > threshold:
            col, row = max_loc  ## this return (x,y) coordinates
            peak_loc[i] = row, col
            # print("col: %d "% col+" row:%d "%row)
            ## suppress neighbors
            radius = np.int(n_hood_size / 2)
            _accumulator[clamp(row - radius, 0, height):clamp(row + radius + 1, 0, height),
            clamp(col - radius, 0, width):clamp(col + radius + 1, 0, width)] = 0
        else:
            peak_loc = peak_loc[:i]
    ## return none zero row. it might not find num_peaks for a given threshold
    return peak_loc

def hough_circle_draw(img, outfile, center_locs, radius):
    ##return line parameters start(x,y) end(x,y)
    ## peak_loc = y, x in accumulator

    for i in range(len(center_locs)):
        row, col = center_locs[i]
        cv2.circle(img, (col, row), radius[i], color=(0, 255, 0), thickness=2)

    if outfile != "":
        save_image_to_file(img, outfile)

def hough_lines_draw(img, outfile, rhos, thetas, peak_loc):
    ##return line parameters start(x,y) end(x,y)
    ## peak_loc = rho,theta in accumulator
    ## find all the points x * cos + y sin = rho
    rho_max = int(round(np.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)))
    for peak in peak_loc:
        lines = np.zeros((img.shape[0] * img.shape[1], 2), dtype=int)
        rho = rhos[peak[0]]
        theta = thetas[peak[1]]
        x0 = rho * np.cos(np.deg2rad(theta))
        y0 = rho * np.sin(np.deg2rad(theta))

        # two easy points.
        x1 = -rho_max*np.sin(np.deg2rad(theta)) + x0
        y1 = rho_max*np.cos(np.deg2rad(theta)) + y0

        x2 = rho_max*np.sin(np.deg2rad(theta)) + x0
        y2 = -rho_max*np.cos(np.deg2rad(theta)) + y0

        cv2.line(img, (np.int(x1), np.int(y1)), (np.int(x2), np.int(y2)), (0, 255, 0), thickness=2)
        ## too slow but accurate. is there a better retro-fit algorithm to use?
        if 0:
            i = 0
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    if (round(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))) == rho):
                        lines[i] = int(x), int(y)
                        # print(round(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))))
                        i += 1
            # print(i)
            cv2.line(img, tuple(lines[0]), tuple(lines[i - 1]), (0, 255, 0), thickness=2)
    if outfile != "":
        save_image_to_file(img, outfile)

def find_circles(img_edges, radius, num_peaks=4):
    centers = []
    valid_radius = []
    index = 0
    ## not every radius will be associating with circles found. Need to
    ## remove the empty discoveries.

    for r in radius:
        #print(r)
        acc = hough_circles_acc(img_edges, r)
        peaks = hough_peaks(acc, num_peaks, threshold=130, n_hood_size=r * 2)
        if peaks.size > 0:
            rs= np.zeros(peaks.shape[0]).astype(np.int)
            rs[:] = r
            centers.append(peaks)
            valid_radius.append(rs)
            #print(r,peaks,acc[peaks[:,0],peaks[:,1]] )
            ## to do: need to remove duplicate centers for slightly different radius.
            ## for example. (255,354) found for radius 28 but [256 354] also found for 27
            ## those two are actually the same circle in the picture

    return centers, valid_radius

def find_circles_optimized(img_edges, radius, num_peaks=4):
    centers = []
    valid_radius = []
    seen = set()
    acc = np.zeros((len(radius),img_edges.shape[0],img_edges.shape[1]),dtype=np.uint8)
    delta = 75
    r_loc = np.empty(3).astype(np.int)

    for r in radius:
        index = np.int(r - radius[0])
        acc[index] = hough_circles_acc(img_edges, r)
        peaks = hough_peaks(acc[index], num_peaks, threshold=120, n_hood_size=r * 2)
        if peaks.size > 0:
            temp_r = np.zeros(peaks.shape[0]).astype(np.int)
            temp_r[:] = index
            r_loc= np.vstack((r_loc,np.stack((temp_r,peaks[:,0],peaks[:,1]),1) ))

    ## search the most voted centers in (r, row, col) space and get rid of duplicated centers
    ## is it even worth it?
    for i in range(1,len(r_loc)):
        r, row, col = r_loc[i]
        temp = np.array([r_loc[j] - r_loc[i] for j in range(1,len(r_loc)) if j != i] )
        candidate=(r,row, col)
        #print("candidate:", candidate)
        for k in range(len(temp)):
            if sum(temp[k]**2) < delta and acc[temp[k,0]+r, temp[k,1]+row, temp[k,2]+col]>acc[candidate] :
                candidate = (temp[k,0]+r, temp[k,1]+row, temp[k,2]+col)
        if candidate not in seen:
            seen.add(candidate)
            centers.append([candidate[1],candidate[2]])
            valid_radius.append(candidate[0]+radius[0])
            #print("finding circle: ",candidate)
    return centers, valid_radius

def parallel_line_filter(acc,rhos, thetas, peaks, distance = 25):
    theta_delta = 5
    acc_delta = 20
    rho_delta = distance  ## distance between two lines
    filtered = []
    for i in range(len(peaks)):
        theta_sad = np.array([abs(thetas[peaks[j, 1]] - thetas[peaks[i, 1]])
                              for j in range(len(peaks)) if j != i])
        rho_sad = np.array([abs(np.int(rhos[peaks[j, 0]]) - np.int(rhos[peaks[i, 0]]))
                            for j in range(len(peaks)) if j != i])
        acc_sad = np.array([abs(np.int(acc[peaks[j, 0], peaks[j, 1]]) - np.int(acc[peaks[i, 0], peaks[i, 1]]))
                            for j in range(len(peaks)) if j != i])

        temp = np.stack((rho_sad, theta_sad, acc_sad), 1)
        for k in range(len(temp)):
            if (temp[k] < [rho_delta, theta_delta, acc_delta]).all():
                filtered.append(peaks[i])
                break
    return filtered

def edge_detection():
    input_file = input_dir + '/' + 'ps1-input0.png'
    output_file = output_dir + '/' + 'ps1-1-a-1.png'

    img = cv2.imread(input_file, cv2.IMREAD_COLOR)

    kernel_sizes = [3, 7]  ## kernel size is limited by sobel operator only 1,3,5,7
    ## from the web: As a rule of thumb, we set the low threshold to 0.66*[mean value] and
    ## set the high threshold to 1.33*[mean value]
    sigmas = [0.1, 0.33, 0.5]  ## tight, median, wide
    edges_set = {}
    index = 0
    for kernel_size in kernel_sizes:
        for sigma in sigmas:
            ## large aperture size seems to be detecting a lot of lousy edges
            edges = canny(img, sigma, kernel_size)
            name = 'size_' + str(kernel_size) + '_sigma_' + str(sigma)
            edges_set[name] = edges
            index += 1

    ##plot
    if 0:
        fig = plt.figure(num=9)
        index = 1
        for key in edges_set:
            # plt.subplot(len(edges_set), 1, index)
            plt.subplot(2, 3, index)
            plt.imshow(edges_set[key], cmap='gray')
            plt.title(key)
            plt.xticks([]), plt.yticks([])
            index += 1
        # plt.subplot_tool()  ## this is dialog ui for adjustment figure attributes right/left/etc
        plt.show()

    ## save the sigma = 0.33 this is the best result
    if output_file:
        save_image_to_file(edges_set['size_3_sigma_0.33'], output_file)

    return edges_set['size_3_sigma_0.33']


def hough_basic():
    input_file = input_dir + '/' + 'ps1-input0.png'
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    edges = canny(img)
    acc, theta, rho = hough_lines_acc(edges)
    output_file = './output' + '/ps1-2-a-1.png'
    save_image_to_file(acc, output_file)
    # find up to 10 strongest liness
    peaks = hough_peaks(acc, num_peaks=10, threshold=0.25 * acc.max(), n_hood_size=50)
    # peaks = hough_peaks(acc, num_peaks=4, threshold=mean, n_hood_size=50)
    # ps1-2-b-1.png - with peaks highlighted (you can use drawing functions).
    output_file = './output' + '/ps1-2-b-1.png'
    acc_copy = acc.copy()
    ## draw white dots on accumulator
    # print(peaks)
    for i in range(len(peaks)):
        row, col = peaks[i]
        cv2.circle(acc_copy, (col, row), 2, 255, -1)
    save_image_to_file(acc_copy, output_file)
    output_file = './output' + '/ps1-2-c-1.png'
    hough_lines_draw(img, output_file, rho, theta, peaks)


def hough_and_noise():
    input_file = './input/ps1-input0-noise.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (19, 19), 4.5)

    ## how to set the correct minval, maxval for canny detector without looking at actual gradients
    ##  I tried many values to get this

    edges_img = cv2.Canny(orig, 25, 50)
    edges_smoothed = cv2.Canny(smoothed, 25, 50)
    acc, theta, rho = hough_lines_acc(edges_smoothed)
    # find up to 10 strongest lines
    peaks = hough_peaks(acc, num_peaks=10, threshold=0.25 * acc.max(), n_hood_size=50)

    acc_copy = acc.copy()
    ## draw white dots on accumulator
    for i in range(len(peaks)):
        row, col = peaks[i]
        cv2.circle(acc_copy, (col, row), 2, 255, -1)

    save_image_to_file(smoothed, './output/ps1-3-a-1.png')
    save_image_to_file(edges_img, './output/ps1-3-b-1.png')
    save_image_to_file(edges_smoothed, './output/ps1-3-b-2.png')
    save_image_to_file(acc_copy, './output/ps1-3-c-1.png')
    hough_lines_draw(orig, './output/ps1-3-c-2.png', rho, theta, peaks)

def find_lines():
    input_file = './input/ps1-input0.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (19, 19), 4.5)

    edges = cv2.Canny(smoothed, 25, 50)
    acc, theta, rho = hough_lines_acc(edges)

    peaks = hough_peaks(acc, num_peaks=4, threshold=0.25 * acc.max(), n_hood_size=50)

    acc_copy = acc.copy()
    ## draw white dots on accumulator
    for i in range(len(peaks)):
        row, col = peaks[i]
        cv2.circle(acc_copy, (col, row), 2, 255, -1)

    save_image_to_file(smoothed, './output/ps1-4-a-1.png')
    save_image_to_file(edges, './output/ps1-4-b-1.png')
    save_image_to_file(acc_copy, './output/ps1-4-c-1.png')
    hough_lines_draw(orig, './output/ps1-4-c-2.png', rho, theta, peaks)


def find_circles_basic():
    input_file = './input/ps1-input1.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (15, 15), 4.5)

    edges = cv2.Canny(smoothed, 25, 50)
    r = 20
    acc = hough_circles_acc(edges, radius=r)

    ##threshold= around 120, n_hood_size=r*2 seem to be generating good result for circle detection
    centers = hough_peaks(acc, num_peaks=10, threshold=110, n_hood_size=r * 2)

    acc_copy = acc.copy()
    ## draw white dots on accumulator
    for i in range(len(centers)):
        row, col = centers[i]
        cv2.circle(acc_copy, (col, row), 2, 255, -1)
        cv2.circle(orig, (col, row), r, color=(0, 255, 0), thickness=2)

    save_image_to_file(smoothed, './output/ps1-5-a-1.png')
    save_image_to_file(edges, './output/ps1-5-a-2.png')
    save_image_to_file(orig, './output/ps1-5-a-3.png')
    save_image_to_file(acc_copy, './output/ps1-5-a-4.png')

    centers_list, radius_list = find_circles(edges, np.arange(start=20, stop=50),num_peaks=10)
    index = 0
    for r in range(len(radius_list)):
        hough_circle_draw(orig, "", centers_list[r], radius_list[r])
    save_image_to_file(orig, './output/ps1-5-b-1.png')

def find_lines_in_clutter():
    input_file = './input/ps1-input2.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray, (15, 15), 3)

    edges = cv2.Canny(smoothed, 25, 50)
    #save_image_to_file(edges,'./output/ps1-6-canny.png')
    acc, thetas, rhos = hough_lines_acc(edges)
    peaks = hough_peaks(acc, num_peaks=15, threshold=100, n_hood_size=25)

    #acc_copy = acc.copy()
    ## draw white dots on accumulator
    #for i in range(len(peaks)):
    #    row, col = peaks[i]
    #    cv2.circle(acc_copy, (col, row), 2, 255, -1)

    #save_image_to_file(acc_copy, './output/ps1-6-acc.png')

    ## rho is the distance from the origian to the closest point on the straight line and
    ## theta is the angle between x axis and the line connecting the origin with that closest point
    ## the two parallel lines should have very close theta or theta+pi(my range is 0 to 180 so skip this check)
    ## votes should be similar too
    filtered = parallel_line_filter(acc,rhos, thetas,peaks, 45)
    hough_lines_draw(orig, './output/ps1-6-a-1.png', rhos, thetas, filtered)


def find_circles_in_clutter():
    input_file = './input/ps1-input2.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    ## so far the best canny pictures
    eroded = cv2.erode(gray, np.ones((5,5),np.uint8), 1)
    smoothed = cv2.bilateralFilter(eroded,5,100,100)

    edges = cv2.Canny(smoothed, 20, 40)
    save_image_to_file(edges, './output/ps1-7-canny.png')
    centers_list, radius_list = find_circles_optimized(edges, np.arange(start=20, stop=36), num_peaks=10)
    #print(centers_list)
    #print(radius_list)
    hough_circle_draw(orig, "", centers_list[1:], radius_list[1:])
    save_image_to_file(orig, './output/ps1-7-a-1.png')


def find_lines_circles_in_distortion():
    input_file = './input/ps1-input3.png'
    orig = cv2.imread(input_file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ## so far the best canny pictures
    eroded = cv2.erode(gray, np.ones((5,5),np.uint8), 1)
    #smoothed = cv2.bilateralFilter(gray,5,100,100)
    smoothed = cv2.GaussianBlur(gray, (9,9), 2.0)

    edges = cv2.Canny(smoothed, 30, 50)
    save_image_to_file(edges, './output/ps1-8-canny.png')

    acc, thetas, rhos = hough_lines_acc(edges)
    peaks = hough_peaks(acc, num_peaks=10, threshold=100, n_hood_size=10)
    filtered = parallel_line_filter(acc, rhos,thetas, peaks, distance=50)
    hough_lines_draw(orig, "", rhos, thetas, filtered)

    centers_list, radius_list = find_circles_optimized(edges, np.arange(start=20, stop=40), num_peaks=10)
    hough_circle_draw(orig, "", centers_list[1:], radius_list[1:])
    save_image_to_file(orig, './output/ps1-8-a-1.png')


if __name__ == "__main__":

    import argparse

    input_dir = "./input"
    output_dir = "./output"

    parser = argparse.ArgumentParser(description="PS1")
    parser.add_argument("-test", "--test", dest="test_item",
                        help="test item(edges/hough/hough_and_noises/find_lines/find_circles/find_lines_in_clutter\
                        /find_circles_in_clutter/find_lines_circles_in_distortion/all)",
                        action="store")

    args = parser.parse_args()
    test_item = args.test_item
    print("testing " + test_item)

    if test_item == "edges":
        edge_detection()
    elif test_item == "hough":
        hough_basic()
    elif test_item == "hough_and_noises":
        hough_and_noise()
    elif test_item == "find_lines":
        find_lines()
    elif test_item == "find_circles":
        find_circles_basic()
    elif test_item == "find_lines_in_clutter":
        find_lines_in_clutter()
    elif test_item == "find_circles_in_clutter":
        find_circles_in_clutter()
    elif  test_item== "find_lines_circles_in_distortion":
        find_lines_circles_in_distortion()
    else:
        print("running all tests")
        edge_detection()
        hough_basic()
        hough_and_noise()
        find_lines()
        find_circles_basic()
        find_lines_in_clutter()
        find_circles_in_clutter()
        find_lines_circles_in_distortion()
