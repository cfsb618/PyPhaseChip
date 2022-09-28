"""
PyPhaseChip is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
PyPhaseChip is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with PyPhaseChip. If not, see <http://www.gnu.org/licenses/>.
Copyright (c) the PyPhaseChip Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/cfsb618/PyPhaseChip>
ACKNOWLEDGMENT:
This code wouldn't be possible without the exhaustive help of my friend Dinesh Pinto.
He is not only a great physicist but also a very good coder. Check out his Github:
https://github.com/dineshpinto
"""

import cv2
import numpy as np
import logging
import os

# Start module level logger
os.remove("PyPhaseChip.log")
logging.basicConfig(filename="PyPhaseChip.log", format='%(name)s ::  %(message)s', level=logging.DEBUG) # %(levelname)s ::
logger = logging.getLogger("fun")


# adjust brightness & contrast
def controller(img, brightness=255, contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:
        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified text string in the image.
    # cv2.putText(cal, 'B:{},_____C:{}'.format(brightness,
    #                                         contrast), (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cal


def first_derivative(gray_image):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(gray_image, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray_image, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad


# find the well with the help of hough
def find_circle(img: np.ndarray, diameter: int, n: float, m: float, dp: float):
    # dp = 1.6
    minDist = 450
    param1 = 50
    param2 = 40
    min_r_chamber = int((diameter / 2) * n)
    max_r_chamber = int((diameter / 2) * m)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                               dp=dp, minDist=minDist, param1=param1, param2=param2,
                               minRadius=min_r_chamber, maxRadius=max_r_chamber)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles_for_dict = circles
        # x, y, radius = circles[0][0]
        x = circles_for_dict[0, 0, 0]
        y = circles_for_dict[0, 0, 1]
        r = circles_for_dict[0, 0, 2]
    else:
        x = 0
        y = 0
        r = 0

    return x, y, r


# little algorithm to find the well reliably
def find_circle_algo(img: np.ndarray, well_data: np.ndarray, diameter: int, dp: float, dev: int):
    n = 0.95
    m = 1.05
    a = 0

    # initial detection
    x, y, r = find_circle(img, diameter, n, m, dp)

    # tries to makes shure that a well is detected
    while x == 0 and a < 5:
        n -= 0.08
        m += 0.08
        a += 1
        logger.warning(f"Could not find well! Retry counter: {a}")
        x, y, r = find_circle(img, diameter, n, m, dp)

    well_data[0, 0] = x
    well_data[0, 1] = y
    well_data[0, 2] = r

    # Account for the case where no well can be found
    if a == 5 or well_data[0, 0] == 0:
        well_found = False
    else:
        well_found = True
    logger.debug(f"well found: {well_found}")

    return x, y, r, well_data, well_found


# little algorithm to find the droplet reliably
def find_droplet_algo(img: np.ndarray, droplet_data: np.ndarray, diameter: int, t, dp: float, dev: int):
    n = 0
    m = 1.05 - t * 0.008
    a = 0
    # initial detection
    x, y, r = find_circle(img, diameter, n, m, dp)
    logger.debug(f"droplet data: x: {x}, y: {y}, r: {r}")

    # tries to makes shure that a well is detected
    while x == 0 and a < 5:
        logger.debug("retry finding droplet...")
        m += 0.05
        a += 1
        x, y, r = find_circle(img, diameter, n, m, dp)
        logger.debug(f"results after {a}-retries: {x},{y},{r}")
    droplet_data[0, 0] = x
    droplet_data[0, 1] = y
    droplet_data[0, 2] = r
    droplet_data[0, 3] = r**2 * 3.14

    # Account for the case where no well can be found
    if a == 5 or droplet_data[0, 0] == 0:
        droplet_found = False
    else:
        droplet_found = True
    logger.debug(f"droplet found: {droplet_found}")

    return x, y, r, droplet_data, droplet_found


def find_multiple_droplets(threshold_img, xw, yw, rw):
    # detects if there are multiple droplets in a well
    # also kicks out too small droplets
    n = 0
    f = 0.65
    threshold = 500 #400

    eroded = cv2.dilate(threshold_img.copy(), (20, 20), iterations=4)
    crop = eroded[int(yw - rw * f):int(yw + rw * f), int(xw - rw * f):int(xw + rw * f)]

    for x, y in np.ndindex(crop.shape):
        if crop[x, y] == 0:
            n += 1
    if n > threshold:
        multidroplet = True
        # print("find_multiple_droplets was triggered!")
        logger.warning("find_multiple_droplets was triggered!")
        logger.warning(f"count: n = {n}")
    else:
        multidroplet = False

    return multidroplet


def mask_img_circle(image, x, y, r, t):
    if t <= 10:
        a = t * 0.002
    else:
        a = 0.02

    f = 0.95 - a
    d = 250

    img_c = cv2.circle(image, (x, y), int(r * f + d), color=(0, 0, 0), thickness=d * 2)

    return img_c


def image_manipulation(masked_img, x, y, r):
    img_circle = masked_img
    blur = cv2.blur(img_circle.copy(), (2, 2))
    thresh_adpt = cv2.adaptiveThreshold(blur.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    morph = cv2.morphologyEx(thresh_adpt.copy(), cv2.MORPH_CLOSE, (8, 8), iterations=1)
    # morph = cv2.erode(thresh_adpt, (5, 5), iterations=5)
    img = cv2.dilate(morph.copy(), (2, 2), iterations=1)

    return img


# iterate over img; starts in the center of the droplet, spirales out of it until edge is reached
# result is a square
def __squircle_iterator(cX_droplet, cY_droplet):
    x = cX_droplet
    y = cY_droplet
    r = 1
    i, j = x - 1, y - 1
    while True:
        while i < x + r:
            i += 1
            yield i, j
        while j < y + r:
            j += 1
            yield i, j
        while i > x - r:
            i -= 1
            yield i, j
        while j > y - r:
            j -= 1
            yield i, j
        r += 1
        j -= 1
        yield i, j


# store selected pixels (more importantly their values) in new array
def squircle_iteration(img, x0, y0, radius):
    radius *= 0.99
    z = np.zeros(shape=(len(img[:, 0]), len(img[0, :])))
    for (idx1, idx2) in __squircle_iterator(x0, y0):
        if (idx1 - x0) ** 2 + (idx2 - y0) ** 2 < radius ** 2:
            z[idx2][idx1] = img[idx2][idx1]
        else:
            break
    return z


# LLPS detector
def LLPS_detector(n_black_pxls, percental_threshold, areas, areas_list, t, droplet_arr, mean_list, r_extrapolated):
    mean_abs = n_black_pxls/r_extrapolated # not abs anymore, now norm
    logger.debug(f"mean norm:  {mean_abs}")

    if len(mean_list) > 1:
        avg_mean_all_previous_images = np.mean(mean_list)
    else:
        avg_mean_all_previous_images = mean_abs

    if n_black_pxls != 0:
        # Calculate percental difference between current mean value and average mean of all previous images
        percental_difference = (mean_abs / avg_mean_all_previous_images) * 100 - 100
        logger.debug(f"perc. diff. (normalised):  {percental_difference}")

        # Calculate absolute difference between current mean value and average mean of all previous images
        abs_difference = abs(np.subtract(mean_abs, avg_mean_all_previous_images))
        logger.debug(f"absolute difference: {abs_difference}")
        abs_threshhold = 7  # Make this a value determined in the interface
        # perc_diff_normalised, percental_threshold
        if percental_threshold < percental_difference and abs_difference > abs_threshhold:
            llps_status = True
            # calculate area based on droplet shrinking history and save to array
            # assumptions: droplets shrink linearly (not totally true in practice, but holds locally)
            p = len(areas_list)
            if len(areas_list) > 8:
                q = len(areas_list) - 8
                a = areas_list[q][0]
                x = t - areas_list[q][1]
            else:
                q = 1
                a = areas_list[1][0]
                x = t
            m = (areas_list[p-3][0] - areas_list[q][0]) / (areas_list[p-3][1] - areas_list[q][1])
            # logger.debug(f"areas list:, {areas_list}")
            # logger.debug(f"p:, {p}, q:, {q}, m:, {m}, a:, {a}, x:, {x}")
            # logger.debug(f"values:  {areas_list[p-3][0]}, {areas_list[q][0]}, {areas_list[p-3][1]}, {areas_list[q][1]} ")

            areas[0, 1] = m * x + a

        else:
            llps_status = False
            # save mean to mean list
    else:
        llps_status = False

    mean_list.append(mean_abs)
    areas_list.append((droplet_arr[0, 3], t))
    logger.debug(f"LLPS Detector | mean list: {mean_list}")

    return llps_status, areas, areas_list, mean_list
