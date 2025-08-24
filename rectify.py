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
Parameters Settings
"""
debug = False
debug_dir = 'debug/'
PROCESS_SIZE = 1000
MODEL_INPUT_SIZE = 1000  # kept for minimal changes, not used now
name = ''

"""
Func: Edge Detection (Classical only)
"""

def detect_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    mean_gray = cv2.mean(gray)
    TH_LIGHT = 150
    if mean_gray[0] > TH_LIGHT:
        gray = exposure.adjust_gamma(gray, gamma=6)
        gray = exposure.equalize_adapthist(gray, kernel_size=None, clip_limit=0.02)
        gray = img_as_ubyte(gray)

    kernel = np.ones((15, 15), np.uint8)
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.medianBlur(closing, 5)
    blurred = cv2.bilateralFilter(blurred, d=0, sigmaColor=15, sigmaSpace=10)

    edged = cv2.Canny(blurred, 75, 200)

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + "_Cannyedge.png"), edged)

    return edged

"""
Func: Four Corner Detection
"""

def cross_point(line1, line2):
    x = 0
    y = 0
    x1 = line1[0]; y1 = line1[1]; x2 = line1[2]; y2 = line1[3]
    x3 = line2[0]; y3 = line2[1]; x4 = line2[2]; y4 = line2[3]
    if (x2 - x1) == 0:
        k1 = None
    else:
        k1 = (y2 - y1) * 1.0 / (x2 - x1)
        b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k1 is None:
        if not k2 is None:
            x = x1
            y = k2 * x1 + b2
    elif k2 is None:
        x = x3
        y = k1 * x3 + b1
    elif not k2 == k1:
        x = (b2 - b1) * 1.0 / (k1 - k2)
        y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def get_angle(sta_point, mid_point, end_point):
    ma_x = sta_point[0][0] - mid_point[0][0]
    ma_y = sta_point[0][1] - mid_point[0][1]
    mb_x = end_point[0][0] - mid_point[0][0]
    mb_y = end_point[0][1] - mid_point[0][1]
    ab_x = sta_point[0][0] - end_point[0][0]
    ab_y = sta_point[0][1] - end_point[0][1]
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    cos_M = (ma_val2 + mb_val2 - ab_val2) / (2 * np.sqrt(ma_val2) * np.sqrt(mb_val2))
    angleAMB = np.arccos(cos_M) / np.pi * 180
    return angleAMB

def checked_valid_transform(approx):
    hull = cv2.convexHull(approx)
    TH_ANGLE = 45
    if len(hull) == 4:
        for i in range(4):
            p1 = hull[(i - 1) % 4]
            p2 = hull[i]
            p3 = hull[(i + 1) % 4]
            angel = get_angle(p1, p2, p3)
            if 90 - TH_ANGLE < angel < 90 + TH_ANGLE:
                continue
            else:
                if debug:
                    print("Detection Error: The detected corners could not form a valid quadrilateral for transformation.")
                raise Exception("Corner points invalid.")
    else:
        if debug:
            print("Detection Error: Could not find four corners from the detected edge.")
        raise Exception("Corner points less than 4.")
    return True

def get_cnt(edged, img, ratio):
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    mask = np.zeros((edged.shape[0], edged.shape[1]), np.uint8)
    mask[10:edged.shape[0] - 10, 10:edged.shape[1] - 10] = 1
    edged = edged * mask

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2(or_better=True) else cnts[1]
    cnts = sorted(cnts, key=lambda c: cv2.arcLength(c, True), reverse=True)
    edgelines = np.zeros(edged.shape, np.uint8)
    cNum = 4

    for i in range(min(cNum, len(cnts))):
        TH = 1 / 20.0
        if cv2.contourArea(cnts[i]) < TH * img.shape[0] * img.shape[1]:
            cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)
        else:
            cv2.drawContours(edgelines, [cnts[i]], 0, (1, 1, 1), -1)
            edgelines = edgelines * edged
            break
        cv2.drawContours(edgelines, [cnts[i]], 0, (255, 255, 255), -1)

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + "_edgelines.png"), edgelines)

    lines = cv2.HoughLines(edgelines, 1, np.pi / 180, 200)
    if lines is None or len(lines) < 4:
        if debug:
            print("Detection Error: Could not find enough lines (must more than 4) from the detected edge.")
        raise Exception("Lines not found.")

    if debug:
        lines_draw = np.zeros((len(lines), 4), dtype=int)
        img_draw = img.copy()
        for i in range(0, len(lines)):
            rho, theta = lines[i][0][0], lines[i][0][1]
            a = np.cos(theta); b = np.sin(theta)
            x0 = a * rho; y0 = b * rho
            lines_draw[i][0] = int(x0 + 1000 * (-b))
            lines_draw[i][1] = int(y0 + 1000 * (a))
            lines_draw[i][2] = int(x0 - 1000 * (-b))
            lines_draw[i][3] = int(y0 - 1000 * (a))
            cv2.line(img_draw, (lines_draw[i][0], lines_draw[i][1]), (lines_draw[i][2], lines_draw[i][3]), (0, 255, 0), 1)
        cv2.imwrite(pjoin(debug_dir, name + '_hough1.png'), img_draw)

    strong_lines = np.zeros([4, 1, 2])
    n2 = 0
    for n1 in range(0, len(lines)):
        if n2 == 4:
            break
        for rho, theta in lines[n1]:
            if n1 == 0:
                strong_lines[n2] = lines[n1]
                n2 = n2 + 1
            else:
                c1 = np.isclose(abs(rho), abs(strong_lines[0:n2, 0, 0]), atol=80)
                c2 = np.isclose(np.pi - theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                c = np.all([c1, c2], axis=0)
                if any(c):
                    continue
                closeness_rho = np.isclose(rho, strong_lines[0:n2, 0, 0], atol=40)
                closeness_theta = np.isclose(theta, strong_lines[0:n2, 0, 1], atol=np.pi / 36)
                closeness = np.all([closeness_rho, closeness_theta], axis=0)
                if not any(closeness) and n2 < 4 and theta != 0:
                    strong_lines[n2] = lines[n1]
                    n2 = n2 + 1

    lines1 = np.zeros((len(strong_lines), 4), dtype=int)
    for i in range(0, len(strong_lines)):
        rho, theta = strong_lines[i][0][0], strong_lines[i][0][1]
        a = np.cos(theta); b = np.sin(theta)
        x0 = a * rho; y0 = b * rho
        lines1[i][0] = int(x0 + 1000 * (-b))
        lines1[i][1] = int(y0 + 1000 * (a))
        lines1[i][2] = int(x0 - 1000 * (-b))
        lines1[i][3] = int(y0 - 1000 * (a))
        if debug:
            cv2.line(img, (lines1[i][0], lines1[i][1]), (lines1[i][2], lines1[i][3]), (0, 255, 0), 3)

    approx = np.zeros((len(strong_lines), 1, 2), dtype=int)
    index = 0
    combs = list((combinations(lines1, 2)))
    for twoLines in combs:
        x1, y1, x2, y2 = twoLines[0]
        x3, y3, x4, y4 = twoLines[1]
        [x, y] = cross_point([x1, y1, x2, y2], [x3, y3, x4, y4])
        if 0 < x < img.shape[1] and 0 < y < img.shape[0] and index < 4:
            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), 3)
            approx[index] = (int(x), int(y))
            index = index + 1

    if debug:
        cv2.imwrite(pjoin(debug_dir, name + '_hough2.png'), img)

    if checked_valid_transform(approx):
        return approx * ratio

"""
Func: Image Postprocess
"""

def set_corner(img, r):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
    row = img.shape[0]; col = img.shape[1]

    for i in range(0, r):
        for j in range(0, r):
            if (r - i) * (r - i) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(0, r):
        for j in range(col - r, col):
            if (r - i) * (r - i) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(0, r):
            if (r - row + i + 1) * (r - row + i + 1) + (r - j) * (r - j) > r * r:
                alpha_channel[i][j] = 0

    for i in range(row - r, row):
        for j in range(col - r, col):
            if (r - row + i + 1) * (r - row + i + 1) + (r - col + j + 1) * (r - col + j + 1) > r * r:
                alpha_channel[i][j] = 0

    img_bgra = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_bgra

def finetune(img, ratio):
    offset = int(2 * ratio)
    img = img[offset + 15:img.shape[0] - offset,
              int(offset * 2):img.shape[1] - int(offset * 2), :]
    if img.shape[0] < img.shape[1]:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 856 * 540)))
        r = int(img.shape[1] / 856 * 31.8)
    else:
        img = cv2.resize(img, (img.shape[1], int(img.shape[1] / 540 * 856)))
        r = int(img.shape[1] / 540 * 31.8)
    img = set_corner(img, r)
    if img.shape[0] > img.shape[1]:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
    return img

"""
Func: Inference (CNN removed)
"""

def inference_all(input_dir, output_dir):
    count = 0
    image_format = [".jpg", ".jpeg", ".bmp", ".png"]
    file_list = os.listdir(input_dir)
    file_list.sort()
    for i in range(0, len(file_list)):
        in_path = os.path.join(input_dir, file_list[i])

        if os.path.isfile(in_path):
            if os.path.splitext(in_path)[1] not in image_format:
                print("{} is not an acceptable image file, please use .jpg/.jpeg/.bmp/.png as input.".format(file_list[i]))
                continue
        else:
            continue

        global name
        name = os.path.splitext(file_list[i])[0]
        out_path = os.path.join(output_dir, name + ".png")

        try:
            image = cv2.imread(in_path)
            img = cv2.resize(image, (PROCESS_SIZE, int(PROCESS_SIZE * image.shape[0] / image.shape[1])))
            ratio = image.shape[1] / PROCESS_SIZE
        except:
            continue

        try:
            if debug:
                print("Edge Detection: classical Canny method...")
            edged = detect_edge(img)
            corners = get_cnt(edged, img, ratio)
        except Exception as e:
            print(f"Failed. {file_list[i]} could not be rectified :(  ({e})")
            continue

        try:
            result = four_point_transform(image, corners.reshape(4, 2))
            result = finetune(result, ratio)

            # 🔹 Force fixed size 1000x631
            result = cv2.resize(result, (1000, 631), interpolation=cv2.INTER_AREA)

            cv2.imwrite(out_path, result)
            print("Success! Output saved in " + os.path.abspath(out_path))
            count = count + 1
        except Exception as e:
            print(f"Failed. {file_list[i]} could not be rectified :(  ({e})")

    print("Done! {}/{} success.".format(count, len(file_list)))



if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 3:
        print("Parameters Error: invalid input or output.")
        sys.exit(1)

    _, input_, output_ = sys.argv

    inference_all(input_, output_)










































































