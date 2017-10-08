import numpy as np
import cv2
def disparity_ncorr(L, R, size = 5, disparity = 30):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

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
            #D[r,c] = c - (mt_c_min + min_loc[0] +  tpl_cols//2)  # disparity
            #Faster. Mine is unbearbly slow.
            _,_, _, max_loc = cv2.minMaxLoc(cv2.matchTemplate(tpl, mt, method = cv2.TM_CCORR_NORMED))
            D[r, c] = c - (mt_c_min + max_loc[0] + tpl_cols // 2)
    return D