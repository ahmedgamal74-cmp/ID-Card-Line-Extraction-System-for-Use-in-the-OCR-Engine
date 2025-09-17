# python rectify.py 0_IDs_input/  1_IDs_rectified/

import os
import sys
import cv2
import imutils
import time
import numpy as np
from os.path import join as pjoin
from skimage import exposure, img_as_ubyte
from imutils.perspective import four_point_transform
from itertools import combinations

"""
Classical edge detection function
"""
def detect_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    mean_gray = cv2.mean(gray)
    TH_LIGHT = 150      # brightness threshold
    if mean_gray[0] > TH_LIGHT:
        gray = exposure.adjust_gamma(gray, gamma=6)     # darken if the image is very bright
        gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
        gray = img_as_ubyte(gray)

    kernel = np.ones((15, 15), np.uint8)
    # denoise (morph close → median blur → bilateral filter)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)   # closing (dilation then erosion)
    blurred = cv2.medianBlur(closing, 5)    # remove salt and pepper noise
    blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10) # smoothing 

    # edge detection using canny
    edged = cv2.Canny(blurred, 75, 200)

    return edged

"""
Helper to get the intersection between two lines 
"""
def cross_point(line1, line2):
    x = 0
    y = 0
    x1 = line1[0]; y1 = line1[1]; x2 = line1[2]; y2 = line1[3]  # line 1 two end points
    x3 = line2[0]; y3 = line2[1]; x4 = line2[2]; y4 = line2[3]  # line 2 two end points

    # Line 1 slope k1 and bias b1
    if (x2 - x1) == 0:
        k1 = None   # slope = inf => vertical line
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
    # Line 2 slope k2 and bias b2
    if (x4 - x3) == 0:
        k2 = None   # slope = inf
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    # 
    if k1 is None:
        if not k2 is None:
            x = x1  # x is const for vertical lines
            y = k2 * x1 + b2 # the y for that const x
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:  # here: y1 = k1 x1 + b1 = k2 x1 + b2
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0

    return [x, y]

"""
Helper to get the angle between two lines 
"""
def get_angle(start_point, mid_point, end_point):  # __|
    # MA vector length
    ma_x = start_point[0][0] - mid_point[0][0]
    ma_y = start_point[0][1] - mid_point[0][1]
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    # MB vector length
    mb_x = end_point[0][0] - mid_point[0][0]
    mb_y = end_point[0][1] - mid_point[0][1]
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    # AB vector length
    ab_x = start_point[0][0] - end_point[0][0]
    ab_y = start_point[0][1] - end_point[0][1]
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    # AB^2 = MA^2 +  MB^2 − 2 * MA * MB cos(AMB)
    cos_M = (ma_val2 + mb_val2 - ab_val2) / (2 * np.sqrt(ma_val2) * np.sqrt(mb_val2))
    angle = np.arccos(cos_M) / np.pi * 180  # angle in radian then convert to deg
    return angle

"""
Helper to check we got an approx rectangualar ( right angles)
"""
def checked_valid_corners(approx):
    hull = cv2.convexHull(approx)   # get CCW points
    TH_ANGLE = 45   # tolearance to check if a right angle  
    if len(hull) == 4:
        # loop to check angle by angle   __|
        for i in range(4):
            p1 = hull[(i - 1) % 4]
            p2 = hull[i]
            p3 = hull[(i + 1) % 4]
            angel = get_angle(p1, p2, p3)
            if 90 - TH_ANGLE < angel < 90 + TH_ANGLE:   # (45 - 135)
                continue
            else:
                raise Exception("Corner not valid (not right angle)")
    else:
        raise Exception("Got less than 4 corners")
    return True

"""
Four corners detection functions
"""
def get_cnt(edged, img, ratio):
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)     # tiny dilation to fix small unconnected lines
    mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
    mask[10:edged.shape[0] - 10, 10:edged.shape[1] - 10] = 1    # turn off the false edges of the image
    edged = edged * mask

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
    edgelines = np.zeros(edged.shape, np.uint8)
    cNum = 4

    for i in range(min(cNum, len(cnts))):
        TH = 1 / 20.0       # contour perimeter threshold
        if cv2.contourArea(cnts[i]) < TH * img.shape[0] * img.shape[1]:
            # whiten the small contours (not our goal)
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
        else:
            # black our goal contour
            cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1) # big contour is the card
            edgelines = edgelines * edged
            break
        # whiten other areas (not our goal)
        cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

    # Lines detector set of points x,y
    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200) # ρ = xcosθ + ysinθ
    if lines is None or len(lines) < 4:
        raise Exception("Lines not found")

    strong_lines = np.zeros([4, 1, 2])
    n2 = 0
    for n1 in range(0, len(lines)):
        if n2 == 4:     # take four lines only
            break
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]    # take first line as it is, check for the rest
                n2 = n2 + 1
            else:
                # check if line is close to others with tolerance 80 px
                c1 = np.isclose(abs(rho), abs(strong_lines[0:n2, 0, 0]), atol=80)  
                # check if line is close to others in orientation (flipped norm) with tolerance 5 deg 
                c2 = np.isclose(np.pi - theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)   
                c = np.all([c1, c2], axis=0)
                if any(c):
                    continue
                # check others (I need the near line with diff angle and vice versa)
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=40)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4 and theta != 0:        # this is "NOT ANDing" 
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1

    # converting rho and theta to long lines usign ρ = xcosθ + ysinθ   
    lines1 = np.zeros((len(strong_lines), 4), dtype=int)
    for i in range(0, len(strong_lines)):
        rho, theta = strong_lines[i][0][0], strong_lines[i][0][1]
        a = np.cos(theta); b = np.sin(theta)
        x0 = a * rho; y0 = b * rho
        # 1000 to make them very distant
        lines1[i][0] = int(x0 + 1000 * (-b))
        lines1[i][1] = int(y0 + 1000 * (a))
        lines1[i][2] = int(x0 - 1000 * (-b))
        lines1[i][3] = int(y0 - 1000 * (a))

    # get the 4 rgiht intersections, 4 lines -> 6 intersection (2 outside image due to non ideal parallelism)
    approx = np.zeros((len(strong_lines), 1, 2), dtype=int)
    index = 0
    combs = list((combinations(lines1, 2))) # 6 combinations
    for twoLines in combs:
        x1, y1, x2, y2 = twoLines[0]
        x3, y3, x4, y4 = twoLines[1]
        [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4]) # get the intersection between two lines and check inside image
        if 0 < x < img.shape[1] and 0 < y < img.shape[0] and index < 4:
            # cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)
            approx[index] = (int(x), int(y))
            index = index + 1

    # check valid rgiht angle corners
    if checked_valid_corners(approx):
        return approx * ratio

"""
Image post processing functions
"""
def postprocess(img, ratio):
    # crop the dark edging from lightening
    img = img[int(2*ratio+15):img.shape[0] - int(2*ratio),
              int(int(2*ratio)*2):img.shape[1] - int(int(2* ratio)*2), :]
    
    # keep the ratio as the egyptian card ration
    if img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 1000 * 630.84)))
    else:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 630.84 * 1000)))

    # rotate  90deg if the highet > width (the card is 90deg oriented)
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
    return img

"""
Run nference on the input images (whole directory)
"""
def inference_all(input_dir, output_dir):
    count = 0
    image_process_size = 1000
    file_list = os.listdir(input_dir)

    for i in range(0, len(file_list)):
        in_path = os.path.join(input_dir, file_list[i])
        name = os.path.splitext(file_list[i])[0]
        out_path = os.path.join(output_dir, name + ".png")

        image = cv2.imread(in_path)
        img = cv2.resize(image, (image_process_size, int(image_process_size * image.shape[0] / image.shape[1])))    # (W, H) -> (1000, 1000*H/W)) 
        ratio = image.shape[1] / image_process_size     # W_old / 1000

        try:
            edged = detect_edge(img)
            # cv2.imshow('edges', edged)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            corners = get_cnt(edged, img, ratio)

            # get the four coners and the returns the oriented reactanglar card as horizontal one |___|
            result = four_point_transform(image, corners.reshape(4, 2))

            result = postprocess(result, ratio)

            result = cv2.resize(result, (1000, 631), interpolation=cv2.INTER_AREA)      # 1000/631 = 1.584 is the original egyptian id card scale (W/H)
            
            cv2.imwrite(out_path, result)
            print(f"Rectified image {1+count} saved in " + os.path.abspath(out_path))
            count = count + 1

        except Exception as e:
            print(f"Failed, {file_list[i]} can not be rectified, {e}")

    print(f"Done, rectified {count}/{len(file_list)} image")

if __name__ == "__main__":
    f, input_dir, output_dir = sys.argv
    inference_all(input_dir, output_dir)










































































