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


# TEST
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
    #cv2.putText(cal, 'B:{},_____C:{}'.format(brightness,
    #                                         contrast), (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cal

# create mask
def create_mask(img, circles_from_hc_detection):
    # read information from circle detection
    x0 = circles_from_hc_detection[0, 0, 0]
    y0 = circles_from_hc_detection[0, 0, 1]
    radius = circles_from_hc_detection[0, 0, 2]*1.2

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
    img[elon_mask == 0] = 209
    img[elon_mask != 0] = img[elon_mask != 0]

    return img


# Contour detection
def detect_contours(elon_mask, thresh_value):
    thresh_value = thresh_value*1.0
    thresh_value = 200
    #print("thresh:", thresh_value)
    _, image = cv2.threshold(elon_mask, thresh_value, 255, cv2.THRESH_BINARY_INV)
    #image = cv2.adaptiveThreshold(elon_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

    def canny_threshold(img, val, ratio, kernel_size):
        low_threshold = val
        img_blur = cv2.blur(img, (10, 10))
        detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:, :, ].astype(img.dtype))
        return dst

    #image = canny_threshold(image, val=0, ratio=3, kernel_size=10) # kernel 3 to low
    # TODO: remove canny if not needed

    threshed_img = image.copy()

    # Kernelsize for last dilate operation is very important for findContours
    # needs to be bigger than 1, reason currently unknown

    # Solidify faint droplet edges, mop up dirt
    kernel = np.ones((2, 2), np.uint8)
    #for i in range(2):
    #    image = cv2.erode(image, kernel, iterations=1)
    #    image = cv2.dilate(image, kernel, iterations=1)
    mopped_up_img = image.copy()

    # and make them rly nice & thick
    kernel_nu = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, kernel_nu, iterations=1)
    dilated_img = image.copy()

    # this is the core: contour detection
    contours_within_elon, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours_within_elon, threshed_img, mopped_up_img, dilated_img

def detect_contours_cannytest(elon_mask, thresh_value):
    thresh_value = thresh_value*1.0
    thresh_value = 200
    #print("thresh:", thresh_value)
    #_, image = cv2.threshold(elon_mask, thresh_value, 255, cv2.THRESH_BINARY_INV)
    #image = cv2.adaptiveThreshold(elon_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 2)

    def canny_threshold(img, minval, maxval, kernel_size):
        low_threshold = minval
        max_threshold = maxval
        img_blur = cv2.blur(img, (10, 10))
        img_dil = cv2.dilate(img_blur, (4, 4), iterations=1)
        detected_edges = cv2.Canny(img_dil, low_threshold, max_threshold, kernel_size)
        mask = detected_edges != 0
        dst = img * (mask[:, :, ].astype(img.dtype))
        return dst

    image = canny_threshold(elon_mask, minval=3, maxval=8, kernel_size=10) # kernel 3 to low
    # TODO: remove canny if not needed

    threshed_img = image.copy()

    # Kernelsize for last dilate operation is very important for findContours
    # needs to be bigger than 1, reason currently unknown

    # Solidify faint droplet edges, mop up dirt
    kernel = np.ones((2, 2), np.uint8)
    #for i in range(2):
    #    image = cv2.erode(image, kernel, iterations=1)
    #    image = cv2.dilate(image, kernel, iterations=1)
    mopped_up_img = image.copy()

    # and make them rly nice & thick
    kernel_nu = np.ones((4, 4), np.uint8)
    image = cv2.dilate(image, kernel_nu, iterations=1)
    dilated_img = image.copy()

    # this is the core: contour detection
    contours_within_elon, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours_within_elon, threshed_img, mopped_up_img, dilated_img


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


def select_droplet_test(contours, diameter_well):
    # create storage
    area = []
    idx_list = []
    length = len(contours)
    contour_center_points = np.zeros(shape=(length, 2))

    area_well = 3.141*(diameter_well/2)**2
    area_well_min = 3.141*((diameter_well/2)*0.2)**2  # TODO: can be determined in the jupyter script

    # Normalise droplet area to well area & write areas into list
    # calculate & write droplet center coordinates into array
    for idx, contour in enumerate(contours):
        area.append(cv2.contourArea(contour))

        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_center_points[idx, 0] = cX
        contour_center_points[idx, 1] = cY

    # get idx of too small or too big areas
    for idx in range(len(area)):
        if area[idx] < area_well_min:
            idx_list.append(idx)
        elif area[idx] > area_well:
            idx_list.append(idx)

    # delete them
    for i in reversed(idx_list):
        del area[i]
        contour_center_points = np.delete(contour_center_points, i, 0)
        contours = np.delete(contours, i, 0)
    print("area after del", area)

    # get index of biggest area
    if len(area) > 0:
        index_maxvalue = area.index(max(area))

        # get droplet area
        area_droplet = int(area[index_maxvalue])

        # get droplet center coordinates
        cX_droplet = int(contour_center_points[index_maxvalue, 0])
        cY_droplet = int(contour_center_points[index_maxvalue, 1])

        # get contour of droplet edge
        contour_droplet = np.asarray(contours[index_maxvalue])

        droplet_status = True

    else:
        area_droplet = 0
        cX_droplet = 0
        cY_droplet = 0
        contour_droplet = 0
        droplet_status = False

    return area_droplet, cX_droplet, cY_droplet, contour_droplet, droplet_status


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
    radius = radius*0.9
    z = np.zeros(shape=(len(img[:, 0]), len(img[0, :])))
    for (idx1, idx2) in __squircle_iterator(x0, y0):
        if (idx1 - x0) ** 2 + (idx2 - y0) ** 2 < radius ** 2:
            z[idx2][idx1] = img[idx2][idx1]
        else:
            break
    return z


# LLPS detector
def LLPS_detection(mean_of_current_image, percental_threshold, area_droplet, dict, time_idx, lane_nr, well_nr):
    if len(dict[0][lane_nr][well_nr]['mean list']) >= 1:
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


def hough_test(dropletcontours, grayimg):
    min_r_chamber = int((238/2)*0.2)
    max_r_chamber = int((238/2)*0.9)

    img = np.zeros(shape=grayimg.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    drawncontours_img = cv2.drawContours(img.copy(), dropletcontours, -1,
                                         (255, 165, 0), 2)

    kernel_nu = np.ones((3, 3), np.uint8)
    image = cv2.dilate(drawncontours_img, kernel_nu, iterations=1)

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=50, param1=10,
                               minRadius=min_r_chamber, maxRadius=max_r_chamber)

    for i in circles[0, :]:
        center = (i[0], i[1])
        # circle outline
        radius = (i[2])
        circles_img = cv2.circle(grayimg.copy(), center, radius, (0, 0, 0), 1)

    return circles_img

