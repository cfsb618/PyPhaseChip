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


# find the well with the help of hough
def find_well(img: np.ndarray, diameter: int, n: float, m: float):
    dp = 1.1
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
def find_well_algo(img: np.ndarray, saved_points: np.ndarray, diameter: int, dev: int):
    n = 0.95
    m = 1.05
    a = 0

    # initial detection
    x, y, r = find_well(img, diameter, n, m)

    saved_points[0, 0] = x
    saved_points[0, 1] = y
    saved_points[0, 2] = r

    # tries to makes shure that a well is detected
    while saved_points[0, 0] == 0 and a < 5:
        n -= 0.02
        m += 0.05
        a += 1
        print("pos. first time counter", a)
        x, y, r = find_well(img, diameter, n, m)
        saved_points[0, 0] = x
        saved_points[0, 1] = y
        saved_points[0, 2] = r

    # checks if in last image a well was detected
    # if so, checks if the current position of the well deviates too much from the old one
    if saved_points[1, 0] != 0:
        a = 0
        x_dif = abs(100 - x / saved_points[1, 0] * 100)
        y_dif = abs(100 - y / saved_points[1, 1] * 100)
        r_dif = abs(100 - r / saved_points[1, 2] * 100)

        while (x_dif > dev or y_dif > dev or r_dif > dev) and a < 5:
            n -= 0.02
            m += 0.05
            a += 1
            print("- pos. deviation counter", a)
            x, y, r = find_well(img, diameter, n, m)
            x_dif = abs(100 - x / saved_points[1, 0] * 100)
            y_dif = abs(100 - y / saved_points[1, 1] * 100)
            r_dif = abs(100 - r / saved_points[1, 2] * 100)

    # Account for the case where no well can be found
    if a == 5 or saved_points[0, 0] == 0:
        well_found = False
    else:
        well_found = True
    print("- well found:", well_found)

    return x, y, r, saved_points, well_found


# create mask
def create_mask(img, mask, saved_points, x, y, r, dev: int):
    # read information from circle detection
    x0 = x
    y0 = y
    radius = r * 1.05

    # checks if well position in current img differs too much from
    # well position in previous img
    if saved_points[1, 0] != 0:
        x_dif = abs(100 - x / saved_points[1, 0] * 100)
        y_dif = abs(100 - y / saved_points[1, 1] * 100)
        r_dif = abs(100 - r / saved_points[1, 2] * 100)

        if x_dif > dev or y_dif > dev or r_dif > dev:
            elon_mask = np.zeros(shape=img.shape, dtype="uint8")
            # iterate over image, equ. checks if pixel is part of the circle
            for idx1 in range(img.shape[0]):
                for idx2 in range(img.shape[1]):
                    if (idx2 - x0) ** 2 + (idx1 - y0) ** 2 < radius ** 2:
                        elon_mask[idx1][idx2] = 255
        else:
            elon_mask = mask
    else:
        elon_mask = mask
        # iterate over image, equ. checks if pixel is part of the circle
        for idx1 in range(img.shape[0]):
            for idx2 in range(img.shape[1]):
                if (idx2 - x0) ** 2 + (idx1 - y0) ** 2 < radius ** 2:
                    elon_mask[idx1][idx2] = 255

    saved_points[1, 0] = x
    saved_points[1, 1] = y
    saved_points[1, 2] = r

    return elon_mask, saved_points


# masking image
def mask_image(img, elon_mask):
    img[elon_mask == 0] = 255
    img[elon_mask != 0] = img[elon_mask != 0]

    return img


def image_manipulation(masked_img, x, y, r):
    img_circle = cv2.circle(masked_img, (x, y), r, (200, 0, 0), 5)
    blur = cv2.blur(img_circle.copy(), (5, 5))
    thresh_adpt = cv2.adaptiveThreshold(blur.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    morph = cv2.morphologyEx(thresh_adpt.copy(), cv2.MORPH_CLOSE, (8, 8), iterations=1)
    #morph = cv2.erode(thresh_adpt, (5, 5), iterations=5)
    img = cv2.dilate(morph.copy(), (2, 2), iterations=1)

    return img


def calculate_profile_plot_coordinates(x, y, r):
    f = 1.1
    W = int(x - f * r)
    E = int(x + f * r)
    N = int(y - f * r)
    S = int(y + f * r)

    return f, N, E, S, W


def set_profile_plots(img, N, E, S, W, x, y, r):
    horizontal = []
    vertical = []

    horizontal.append(img[int(y - 1.5 * r / 2), W:E])
    horizontal.append(img[int(y - r / 2), W:E])
    horizontal.append(img[int(y - 0.5 * r / 2), W:E])
    horizontal.append(img[y, W:E])
    horizontal.append(img[int(y + 0.5 * r / 2), W:E])
    horizontal.append(img[int(y + r / 2), W:E])
    horizontal.append(img[int(y + 1.5 * r / 2), W:E])

    vertical.append(img[N:S, int(x - 1.5 * r / 2)])
    vertical.append(img[N:S, int(x - r / 2)])
    vertical.append(img[N:S, int(x - 0.5 * r / 2)])
    vertical.append(img[N:S, x])
    vertical.append(img[N:S, int(x + 0.5 * r / 2)])
    vertical.append(img[N:S, int(x + r / 2)])
    vertical.append(img[N:S, int(x + 1.5 * r / 2)])

    length_hor = np.arange(horizontal[0].size)
    length_vert = np.arange(vertical[0].size)
    length_array = len(length_hor)

    return length_hor, length_vert, length_array, horizontal, vertical


def normalise_profile_plot_length(norm_peaks, centerpoints, m):
    delta = np.zeros(7)
    delta_edges = 0
    cubic_norm_peaks = np.zeros(shape=(7, len(norm_peaks[3])))

    # get index of first 0 from the left of middle array
    for idx, val in enumerate(norm_peaks[3]):
        if val == 0:
            outer_edge_well = idx
            break

    for n in (0, 1, 2, 4, 5, 6):
        add = []
        temp = norm_peaks[n].copy()

        # now get first 0 val from other arrays and calculate difference
        for idx, val in enumerate(norm_peaks[n]):
            if val == 0:
                edge_well = idx
                delta_edges = edge_well - outer_edge_well
                break

        delta[n] = delta_edges
        max_len = len(temp) - 1

        # delete values from the right...
        for i in range(int(delta[n])):
            i_minus = int(max_len - i)
            temp = np.delete(temp, i_minus)

        # ...and from the left
        for i in range(int(delta[n])):
            del_idx = int(delta[n] - i)
            temp = np.delete(temp, del_idx)
            for a in range(2):
                add.append(np.max(temp))

        if centerpoints[0, m] != 0:
            mid = int(centerpoints[0, m])
        else:
            mid = int(len(temp) / 2)
        cubic_norm_peaks[n] = np.insert(temp, mid, add)
    cubic_norm_peaks[3] = norm_peaks[3]

    return cubic_norm_peaks


def compute_droplet_from_peaks(x: int, y: int, r: int, f: float, pp_arrays: np.ndarray, centerpoints: np.ndarray,
                               n: int, radius_old: np.ndarray):

    # n can be 0 for horizontal or 1 for vertical
    edges_idx = np.zeros(shape=(2, 2))
    start_x = int(x - r * f)  # x-values where the horizontal lines start
    start_y = int(y - r * f)  # y-values where the horizontal lines start
    mid_rel_pp_plots = np.zeros(shape=(2, 7))
    dia_temp = np.zeros(7)
    delta_midpoints = np.zeros(7)

    # accounts for a moving droplet center
    if centerpoints[0, n] == 0:
        mid = int(len(pp_arrays[3]) / 2)
    else:
        centerpoints[1, n] = centerpoints[0, n]
        mid = int(centerpoints[1, n])
    print("mid_start:", mid)

    # "walk" right/left from center until value is equal 0, save idx, this is our edge
    for j in range(len(pp_arrays)):
        for i in range(len(pp_arrays[j])-mid):
            if pp_arrays[j][mid + i] == 0:
                edges_idx[n, 1] = mid + i
                break

        for i in range(mid):
            if pp_arrays[j][mid - i] == 0:
                edges_idx[n, 0] = mid - i
                break

        mid_rel_pp_plots[n, j] = int((edges_idx[n, 0] + edges_idx[n, 1]) / 2)
        print(n, j, "L:", edges_idx[n, 0], "R:", edges_idx[n, 1], "m:", mid_rel_pp_plots[n, j])
        dia_temp[j] = np.subtract(edges_idx[n, 1], edges_idx[n, 0])

    # filter
    # if radius_temp is bigger than radius from previous droplet: delete it, its wrong
    print("r_old:", radius_old)
    if radius_old[n] != 0:
        for idx in range(len(dia_temp)):
            if dia_temp[idx] > int(radius_old[n] * 2.2):
                dia_temp[idx] = 0
                mid_rel_pp_plots[n, idx] = 0
                delta_midpoints[idx] = 0
                print("r_deleted:", idx, dia_temp[idx])

            if dia_temp[idx] < int(radius_old[n] * 1.7):
                dia_temp[idx] = 0
                mid_rel_pp_plots[n, idx] = 0
                delta_midpoints[idx] = 0
                print("r_deleted:", idx, dia_temp[idx])

    #print("0) result after first filters")
    #print(mid_rel_pp_plots)
    # if value deviates too much from avg, set it to zero
    avg = np.sum(mid_rel_pp_plots[n, :]) / np.count_nonzero(mid_rel_pp_plots[n, :])
    for j in range(7):
        if mid_rel_pp_plots[n, j] != 0:
            delta_midpoints[j] = abs((mid_rel_pp_plots[n, j]/avg) * 100 - 100)

    while np.max(delta_midpoints) > 10:
        print("average in while loop:", avg, " - np.max:", np.max(delta_midpoints))
        for idx, val in enumerate(delta_midpoints):
            if val == np.max(delta_midpoints):
                print("delets idx:" , idx)
                mid_rel_pp_plots[n, idx] = 0
                delta_midpoints[idx] = 0
                dia_temp[idx] = 0
                break

        avg = np.sum(mid_rel_pp_plots[n, :]) / np.count_nonzero(mid_rel_pp_plots[n, :])
        for j in range(7):
            if delta_midpoints[j] != 0:
                delta_midpoints[j] = abs((mid_rel_pp_plots[n, j] / avg) * 100 - 100)

    print("result after both filters")
    print("midpoints",mid_rel_pp_plots[n, :])
    print("dias", dia_temp)


    # calculate centerpoint
    # sometimes, all arrays equal zero due to llps happening bot got not detected (too much dirt)
    # if so, use old values
    if n == 0:
        start_value = start_x
    else:
        start_value = start_y

    if np.any(mid_rel_pp_plots) != 0:
        centerpoint_rel = int(np.sum(mid_rel_pp_plots[n, :])/np.count_nonzero(mid_rel_pp_plots[n, :]))
        print("mid calc: ", centerpoint_rel)

        centerpoint_abs = centerpoint_rel + start_value
        centerpoints[0, n] = centerpoint_rel

        diameter = int(np.sum(dia_temp) / np.count_nonzero(dia_temp))
        droplet_found = True
    else:
        print("I USED THE OLD VALUES!")
        diameter = int(radius_old[n] * 2 * 0.8)
        droplet_found = True
        centerpoint_abs = int(centerpoints[1, n] + start_value)

    return centerpoint_abs, centerpoints, diameter, droplet_found


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
    radius *= 0.99
    z = np.zeros(shape=(len(img[:, 0]), len(img[0, :])))
    for (idx1, idx2) in __squircle_iterator(x0, y0):
        if (idx1 - x0) ** 2 + (idx2 - y0) ** 2 < radius ** 2:
            z[idx2][idx1] = img[idx2][idx1]
        else:
            break
    return z


# LLPS detector
def LLPS_detection(mean_of_current_image, percental_threshold, area_droplet, areas, mean_list):
    if len(mean_list) > 1:
        avg_mean_all_previous_images = np.mean(mean_list)
    else:
        avg_mean_all_previous_images = mean_of_current_image

    #print("current",mean_of_current_image)
    #print("avg",avg_mean_all_previous_images)

    # Calculate percental difference between current mean value and average mean of all previous images
    percental_difference = abs((mean_of_current_image / avg_mean_all_previous_images) * 100 - 100)
    print("perc. diff.: ", percental_difference)

    if percental_difference > percental_threshold:
        llps_status = True
        # save area to array
        areas[0, 1] = area_droplet
        # save last mean
        mean_list.append(mean_of_current_image)
    else:
        llps_status = False
        # save mean to mean list
        mean_list.append(mean_of_current_image)

    return llps_status, areas, mean_list


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

