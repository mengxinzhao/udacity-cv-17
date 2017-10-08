import numpy as np
import cv2

# very slow
# this needs to be optimized heavily
def min_ssd(template, target, window_size):
    """
    :param template:
    :param target:
    :param window_size
    :return: min_ssd, matching box top left x
    """
    template_height = template.shape[0]
    target_height = target.shape[0]
    #print("template height %d target_height: %d" % (template_height, target_height))
    #print("template width %d target width : %d" % (template.shape[1], target.shape[1]))
    if target_height != template_height:
        print("error! template and target should have the same rows")
        return -1, -1
    ssd_array = np.zeros((target.shape[1]-template.shape[1]+1))
    #print(ssd_array)
    for col in range(0,target.shape[1]-window_size+1):
        box = target[:,col:col+window_size]
        #print(box, col)
        ssd_array[col] = sum(sum((template-box)**2))
        #print(col, ssd_array[col])
    x = np.argmin(ssd_array)
    return ssd_array[x], x

def ssd(template, target, window_size):
    """
    :param template:
    :param target:
    :param window_size
    :return: ssd array of
    """
    ssd_array = np.zeros((1, target.shape[1]-template.shape[1]+1), dtype =np.float32)
    #print(ssd_array)
    for col in range(0,target.shape[1]-window_size+1):
        #print(box, col)
        ssd_array[:,col] = sum(sum((template-target[:,col:col+window_size])**2))
        #print(col, ssd_array[col])
    return ssd_array

def disparity_ssd(L, R, size = 5, disparity = 15):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    :rtype: D
    """

    # TODO: Your code here
    window_size = size  # from Fundementals of Computer Vision

    dmax = disparity * 2 + 1
    moving_range = np.arange(dmax)-disparity # -10, -9 ..0..9, 10 it has to be bigger than window_size
    height, width = L.shape
    tpl_rows = tpl_cols = window_size
    D = np.zeros(L.shape, dtype = np.float32)

    for r in range(tpl_rows//2, height - tpl_rows//2):
        for c in range(tpl_cols//2, width - tpl_cols//2):
            tpl_r_min = r - tpl_rows//2
            tpl_r_max = r + tpl_rows//2+1       ## python range max is not included
            tpl_c_min = c - tpl_cols//2
            tpl_c_max = c + tpl_cols//2+1
            tpl = L[tpl_r_min:tpl_r_max, tpl_c_min:tpl_c_max].astype(np.float32)
            mt_c_min = max(c + moving_range[0]- tpl_cols//2 ,0)
            mt_c_max = min(c + moving_range[dmax-1] + tpl_cols//2 +1 , width)
            mt = R[tpl_r_min:tpl_r_max,mt_c_min:mt_c_max].astype(np.float32)

            # slide template patch to the target matching area. Caculate and when SSD(r,c) has the minimum assign to D(r,c)
            #_, min_loc = min_ssd(tpl,mt,window_size)
            #D[r, c] = c - (mt_c_min + min_loc + tpl_rows//2)  # disparity

            #_, _, min_loc, _ = cv2.minMaxLoc(ssd(tpl,mt,window_size))
            ##D[r,c] = c - (mt_c_min + min_loc[0] +  tpl_cols//2)  # disparity
            #Faster. Mine is unbearbly slow.
            _,_, min_loc, _ = cv2.minMaxLoc(cv2.matchTemplate(tpl, mt, method = cv2.TM_SQDIFF_NORMED))
            D[r, c] = c - (mt_c_min + min_loc[0] + tpl_cols // 2)
    return D

