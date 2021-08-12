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


# calculate starting concentrations
def starting_concentration(initial_conc_solution1, initial_conc_solution2, initial_ratio):
    starting_conc = np.zeros(shape=(5, 2))
    # [:,0] = solution 1, [:,1] = solution 2

    # conc. in lane 1 (outer left)
    starting_conc[0, 0] = initial_conc_solution1 / initial_ratio * (initial_ratio - 1)
    starting_conc[0, 1] = initial_conc_solution2 / initial_ratio * 1
    # conc. in lane 5 (outer right)
    starting_conc[4, 0] = initial_conc_solution1 / initial_ratio * 1
    starting_conc[4, 1] = initial_conc_solution2 / initial_ratio * (initial_ratio - 1)
    # conc. in lane 3 (middle)
    starting_conc[2, 0] = (starting_conc[0, 0] + starting_conc[4, 0]) / 2
    starting_conc[2, 1] = (starting_conc[0, 1] + starting_conc[4, 1]) / 2
    # conc. in lane 2
    starting_conc[1, 0] = (starting_conc[0, 0] + starting_conc[2, 0]) / 2
    starting_conc[1, 1] = (starting_conc[0, 1] + starting_conc[2, 1]) / 2
    # conc. in lane 4
    starting_conc[3, 0] = (starting_conc[2, 0] + starting_conc[4, 0]) / 2
    starting_conc[3, 1] = (starting_conc[2, 1] + starting_conc[4, 1]) / 2

    return starting_conc


# create mask
def create_mask(img, circles_from_hc_detection):
    # read information from circle detection
    x0 = circles_from_hc_detection[0, 0, 0]
    y0 = circles_from_hc_detection[0, 0, 1]
    radius = circles_from_hc_detection[0, 0, 2]*1.1

    # create 1 elon mask
    elon_mask = np.zeros(shape=img.shape, dtype="uint8")

    # iterate over image, equ. checks if pixel is part of the circle
    for idx1 in range(img.shape[0]):
        for idx2 in range(img.shape[1]):
            if (idx2 - x0) ** 2 + (idx1 - y0) ** 2 < radius ** 2:
                elon_mask[idx1][idx2] = 255

    return elon_mask


# masking image
def mask_image(img, elon_mask):
    img[elon_mask == 0] = 0
    img[elon_mask != 0] = img[elon_mask != 0]

    return img


# Contour detection
def detect_contours(elon_mask, thresh_value):
    thresh_value = thresh_value*0.90
    _, image = cv2.threshold(elon_mask, thresh_value, 255, cv2.THRESH_BINARY_INV)
    #image = cv2.adaptiveThreshold(elon_mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,25,2) #test

    def canny_threshold(img, val, ratio, kernel_size):
        low_threshold = val
        img_blur = cv2.blur(img, (10, 10))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:, :, ].astype(img.dtype))
        return dst

    #image = canny_threshold(image, val=0, ratio=3, kernel_size=10) # kernel 3 to low

    safespot1 = image.copy()
    # Solidify faint droplet edges
    kernel = np.ones((8, 8), np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)
    image = cv2.erode(image, kernel, iterations=3)

    # and make them rly nice & thick
    kernel_nu = np.ones((7, 7), np.uint8)
    image = cv2.dilate(image, kernel_nu, iterations=1)
    safespot2 = image.copy()

    # this is the core: contour detection
    contours_within_elon, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if somehow the image is not recognised as a contour, use cv2.RETR_EXTERNAL

    return contours_within_elon, safespot1, safespot2


# Select second biggest contour in the image, hence, the biggest droplet since image is
# the biggest contour
def select_droplet(contours):
    # create storage
    area = []
    contour_center_points = np.zeros(shape=((len(contours)), 2))

    # Write areas into list; calculate & write droplet center coordinates into array
    for idx, contour in enumerate(contours):
        area.append(cv2.contourArea(contour))

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_center_points[idx, 0] = cX
        contour_center_points[idx, 1] = cY

    # get index of second highest value
    index_2nd_maxvalue = np.argsort(area)[-2]

    # get droplet area
    area_droplet = int(area[index_2nd_maxvalue])

    # get droplet center coordinates
    cX_droplet = int(contour_center_points[index_2nd_maxvalue, 0])
    cY_droplet = int(contour_center_points[index_2nd_maxvalue, 1])

    # get coordinates of droplet edge
    contour_droplet = np.asarray(contours[index_2nd_maxvalue])

    return area_droplet, cX_droplet, cY_droplet, contour_droplet


# calculate minimal distance between droplet center and droplet contour
def minDistance(contour_droplet, cX_droplet, cY_droplet):
    distances = np.zeros(shape=(len(contour_droplet)))
    n = 0
    x_coordinates = contour_droplet[:, 0, 0]
    y_coordinates = contour_droplet[:, 0, 1]
    min_distance = 0

    for X, Y in zip(x_coordinates, y_coordinates):
        # print('x:', X)
        # print('y:', Y)
        distances[n] = ((cX_droplet - X) ** 2 + (cY_droplet - Y) ** 2) ** (1 / 2)
        min_distance = np.min(distances)
        n += 1

    return min_distance


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
    radius = radius*0.5
    z = np.zeros(shape=(len(img[:, 0]), len(img[0, :])))
    for (idx1, idx2) in __squircle_iterator(x0, y0):
        if (idx1 - x0) ** 2 + (idx2 - y0) ** 2 < radius ** 2:
            z[idx2][idx1] = img[idx2][idx1]
        else:
            break
    return z


# LLPS detector
def LLPS_detection(mean_of_current_image, percental_threshold, area_droplet, dict, time_idx, lane_nr, well_nr):
    if len(dict[0][lane_nr][well_nr]['mean list']) > 1:
        avg_mean_all_previous_images = np.mean(dict[0][lane_nr][well_nr]['mean list'])
    else:
        avg_mean_all_previous_images = mean_of_current_image

    # Calculate percental difference between current mean value and average mean of all previous images
    percental_difference = (mean_of_current_image / avg_mean_all_previous_images) * 100 - 100

    if percental_difference > percental_threshold:
        dict[0][lane_nr][well_nr]['LLPS status'] = True
        # save name of image where LLPS was detected
        dict[0][lane_nr][well_nr]['LLPS name'] = dict[time_idx][lane_nr][well_nr]['name']
        # save area to array
        dict[0][lane_nr][well_nr]['areas'][0, 1] = area_droplet
        # save img ID
        dict[0][lane_nr][well_nr]['ID'] = time_idx

    else:
        dict[0][lane_nr][well_nr]['LLPS status'] = False
        # save mean to mean list
        dict[0][lane_nr][well_nr]['mean list'].append(mean_of_current_image)