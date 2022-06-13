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

# Start module level logger
logging.basicConfig(format='%(name)s :: %(levelname)s :: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


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
def find_well_algo(img: np.ndarray, well_data: np.ndarray, diameter: int, dev: int):
    n = 0.95
    m = 1.05
    a = 0

    # initial detection
    x, y, r = find_well(img, diameter, n, m)

    # tries to makes shure that a well is detected
    while x == 0 and a < 5:
        n -= 0.02
        m += 0.05
        a += 1
        logger.warning(f"Could not find well! Retry counter: {a}")
        x, y, r = find_well(img, diameter, n, m)

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


# create mask
def create_mask(img, mask, well_data, x, y, r, dev: int):
    # read information from circle detection
    x0 = x
    y0 = y
    radius = r * 1.05

    # checks if well position in current img differs too much from
    # well position in previous img
    if well_data[1, 0] != 0:
        x_dif = abs(100 - x / well_data[1, 0] * 100)
        y_dif = abs(100 - y / well_data[1, 1] * 100)
        r_dif = abs(100 - r / well_data[1, 2] * 100)

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

    well_data[1, 0] = x
    well_data[1, 1] = y
    well_data[1, 2] = r

    return elon_mask, well_data


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
    # morph = cv2.erode(thresh_adpt, (5, 5), iterations=5)
    img = cv2.dilate(morph.copy(), (2, 2), iterations=1)

    return img


def calculate_profile_plot_coordinates(x, y, r):
    f = 1.1
    W = int(x - f * r)
    E = int(x + f * r)
    N = int(y - f * r)
    S = int(y + f * r)

    return f, N, E, S, W


def set_profile_plots2(img, N, E, S, W, x, y, r, d, j):
    horizontal = []
    vertical = []

    value = coordinates_value_selector(j, r, x, y, 0)
    horizontal.append(img[int(value + d), W:E])

    value = coordinates_value_selector(j, r, x, y, 1)
    vertical.append(img[N:S, int(value + d)])

    return horizontal, vertical


def profile_plot_filter(img, N, E, S, W, x, y, r):
    # sum profile plots
    adjacent = [-1, 0, 1]

    hor_sum = []
    vert_sum = []

    for j in range(7):
        hor_list = []
        vert_list = []
        for d in adjacent:
            horizontal, vertical = set_profile_plots2(img, N, E, S, W, x, y, r, d, j)
            hor_list.append(horizontal)
            vert_list.append(vertical)

        logger.debug(f"horizontal profile plots: {hor_list}")
        for part in zip(*hor_list):
            hor_sum.append(sum(part))

        for part in zip(*vert_list):
            vert_sum.append(sum(part))

    return hor_sum, vert_sum


def compute_droplet_from_peaks(x: int, y: int, r: int, f: float, pp_arrays: np.ndarray, centerpoints_rel: np.ndarray,
                               n: int, radius_old: np.ndarray, radius_droplet_old: int, avg_sum_prev: int):
    # n can be 0 for horizontal or 1 for vertical
    edges_idx = np.zeros(shape=(2, 2))
    mid_rel_pp_plots = np.zeros(shape=(2, 7))
    dia_temp = np.zeros(7)
    droplet_coordinates = np.zeros(shape=(7, 2, 2))  # [line_nr, edge 1/2, x/y values]

    # TODO: adjust for optimizer: this way it doesnt to shit, mid important for edge detection
    # accounts for a moving droplet center
    if centerpoints_rel[0, n] == 0:
        mid = int(len(pp_arrays[3]) / 2)
    else:
        centerpoints_rel[1, n] = centerpoints_rel[0, n]
        mid = int(centerpoints_rel[1, n])
    logger.debug("Compute droplets: used mid:", mid)

    # "walk" right/left from center until value is equal 0, save idx, this is our edge
    for j in range(len(pp_arrays)):
        for i in range(1, len(pp_arrays[j]) - mid):
            if pp_arrays[j][mid + i] == 0:
                edges_idx[n, 1] = mid + i
                break

        for i in range(1, mid):
            if pp_arrays[j][mid - i] == 0:
                edges_idx[n, 0] = mid - i
                break

        droplet_coordinates = compute_coordinates(edges_idx, x, y, r, f, j, n, droplet_coordinates)  # FOR TESTING

        mid_rel_pp_plots[n, j] = int((edges_idx[n, 0] + edges_idx[n, 1]) / 2)
        logger.debug(f"detected edges: h/v:{n}, line: {j}, L: {edges_idx[n, 0]} | R: {edges_idx[n, 1]}, midpoint: {mid_rel_pp_plots[n, j]}")
        dia_temp[j] = np.subtract(edges_idx[n, 1], edges_idx[n, 0])

    # optimiser from dinesh:
    x_d, y_d = filter_coordinates(droplet_coordinates, 238, n, radius_droplet_old)
    x_droplet, y_droplet, r_droplet = optimise_circle(x_d, y_d)
    avg_sum = 0

    if x_droplet != 0:
        droplet_found = True
    else:
        droplet_found = False

    return droplet_found, x_droplet, y_droplet, r_droplet, avg_sum, droplet_coordinates


def optimise_circle(x: list, y: list):  # -> tuple[float, float, float]
    """
    This function is also born out of the head of Dinesh Pinto
    thank you very much
    """

    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv = np.sum(u * v)
    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suuv = np.sum(u ** 2 * v)
    Suvv = np.sum(u * v ** 2)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1 = np.sqrt((x - xc_1) ** 2 + (y - yc_1) ** 2)
    R_1 = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1 - R_1) ** 2)
    return xc_1, yc_1, R_1


def filter_coordinates(droplet_coordinates: np.ndarray, threshold, n, r_prev):  # -> Tuple[np.ndarray, np.ndarray]:
    x_d, y_d = np.hsplit(droplet_coordinates.reshape(14, 2), 2)
    x_d, y_d = x_d.flatten(), y_d.flatten()

    filtered_list_x, filtered_list_y = [], []

    if n == 0:
        m = x_d
    else:
        m = y_d

    midpoints = []

    for idx in range(0, len(m), 2):
        temp = np.abs(m[idx] + m[idx + 1])
        mid_point = temp / 2
        midpoints.append(mid_point)
    mean = sum(midpoints) / len(midpoints)
    mean_midpoints = int(mean)

    perc_delta_midpoints = []
    for j in range(len(midpoints)):
        if midpoints[j] != 0:
            perc_delta_midpoints.append(abs((midpoints[j] / mean_midpoints) * 100 - 100))

    while np.max(perc_delta_midpoints) > 5:
        for idx, val in enumerate(perc_delta_midpoints):
            if val == np.max(perc_delta_midpoints):
                del (midpoints[idx])
                del (perc_delta_midpoints[idx])
                x_d = np.delete(x_d, (idx * 2, idx * 2 + 1))
                y_d = np.delete(y_d, (idx * 2, idx * 2 + 1))
                break

        avg = np.sum(midpoints) / len(midpoints)
        for j in range(len(perc_delta_midpoints)):
            if perc_delta_midpoints[j] != 0:
                perc_delta_midpoints[j] = abs((midpoints[j] / avg) * 100 - 100)

    if r_prev != 0:
        r_prev = r_prev
    else:
        r_prev = threshold / 2

    # update m
    if n == 0:
        m = x_d
    else:
        m = y_d

    for idx in range(0, len(m), 2):
        if (delta := np.abs(m[idx] - m[idx + 1])) <= threshold and np.abs((m[idx] - m[idx + 1]) / 2) < r_prev * 1.1:
            # print(delta)
            filtered_list_x.append(x_d[idx])
            filtered_list_x.append(x_d[idx + 1])
            filtered_list_y.append(y_d[idx])
            filtered_list_y.append(y_d[idx + 1])

    return filtered_list_x, filtered_list_y


def avg_calculate_droplet(xh, xv, yh, yv, rh, rv, r_prev, avg_sumh, avg_sumv):
    # calculates difference from previous position and radius
    # takes value that is closer to previous pos/r

    if rv != 0 and rh != 0:
        if r_prev != 0:
            delta_h = abs(np.subtract(rh, r_prev))
            delta_v = abs(np.subtract(rv, r_prev))
            if delta_h < delta_v:
                r_droplet = rh
            else:
                r_droplet = rv
        else:
            r_droplet = np.min((rh, rv))
        x_droplet = int((xh + xv) / 2)
        y_droplet = int((yh + yv) / 2)
        avg_sum = int((avg_sumh + avg_sumv) / 2)

    elif rv != 0 and rh == 0:
        r_droplet = rv
        x_droplet = xv
        y_droplet = yv
        avg_sum = avg_sumv
    else:
        r_droplet = rh
        x_droplet = xh
        y_droplet = yh
        avg_sum = avg_sumh

    return x_droplet, y_droplet, r_droplet, avg_sum


def compute_coordinates(edges_idx: np.ndarray, x: int, y: int, r: int, f: float, line_number: int, n: int,
                        droplet_coordinates: np.ndarray):
    # compute absolute circle coordinates and store them in list
    # used after each time edges_idexes get detected

    # choose values to compute coordinates, then compute them
    # horizontal:
    if n == 0:
        y_value = coordinates_value_selector(line_number, r, x, y, n)
        start_x = int(x - r * f)  # x-values where the horizontal lines start

        droplet_coordinates[line_number, 0, 0] = edges_idx[n, 0] + start_x
        droplet_coordinates[line_number, 0, 1] = y_value
        droplet_coordinates[line_number, 1, 0] = edges_idx[n, 1] + start_x
        droplet_coordinates[line_number, 1, 1] = y_value
        logger.debug(f"{droplet_coordinates[line_number, 0, 0]}, {droplet_coordinates[line_number, 0, 1]}, // "
                    f"{droplet_coordinates[line_number, 1, 0]}, {droplet_coordinates[line_number, 1, 1]}")

    # vertical:
    else:
        x_value = coordinates_value_selector(line_number, r, x, y, n)
        start_y = int(y - r * f)  # y-values where the vertical lines start

        droplet_coordinates[line_number, 0, 0] = x_value
        droplet_coordinates[line_number, 0, 1] = edges_idx[n, 0] + start_y
        droplet_coordinates[line_number, 1, 0] = x_value
        droplet_coordinates[line_number, 1, 1] = edges_idx[n, 1] + start_y

    return droplet_coordinates


def coordinates_value_selector(profile_plot_number: int, r: int, x: int, y: int, n: int):
    if n == 0:
        m = y
    else:
        m = x

    if profile_plot_number == 0:
        value = m - 1.5 * r / 2
    elif profile_plot_number == 1:
        value = m - 1.0 * r / 2
    elif profile_plot_number == 2:
        value = m - 0.5 * r / 2
    elif profile_plot_number == 3:
        value = m
    elif profile_plot_number == 4:
        value = m + 0.5 * r / 2
    elif profile_plot_number == 5:
        value = m + 1.0 * r / 2
    else:
        value = m + 1.5 * r / 2

    return value


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
def LLPS_detection(mean_of_current_image, percental_threshold, areas, droplet_arr, mean_list):
    mean_abs = mean_of_current_image

    if len(mean_list) > 1:
        avg_mean_all_previous_images = np.mean(mean_list)
    else:
        avg_mean_all_previous_images = mean_abs

    # Calculate percental difference between current mean value and average mean of all previous images
    percental_difference = (mean_abs / avg_mean_all_previous_images) * 100 - 100
    logger.debug(f"perc. diff.:  {percental_difference}")

    # Calculate absolute difference between current mean value and average mean of all previous images
    abs_difference = abs(np.subtract(mean_abs, avg_mean_all_previous_images))
    abs_threshhold = 30

    if percental_difference > percental_threshold and abs_difference > abs_threshhold:
        llps_status = True
        # save area to array
        areas[0, 1] = droplet_arr[0, 3]
        # save last mean
        mean_list.append(mean_abs)
    else:
        llps_status = False
        # save mean to mean list
        mean_list.append(mean_abs)

    return llps_status, areas, mean_list
