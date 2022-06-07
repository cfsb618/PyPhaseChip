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
        print("well detection retry counter", a)
        x, y, r = find_well(img, diameter, n, m)


    # checks if in last image a well was detected
    # if so, checks if the current position of the well deviates too much from the old one
    #if well_data[1, 0] != 0:
    #    a = 0
    #    x_dif = abs(100 - x / well_data[1, 0] * 100)
    #    y_dif = abs(100 - y / well_data[1, 1] * 100)
    #    r_dif = abs(100 - r / well_data[1, 2] * 100)#

    #    while (x_dif > dev or y_dif > dev or r_dif > dev) and a < 5:
    #        n -= 0.02
    #        m += 0.05
     #       a += 1
    #        print("- pos. deviation counter", a)
    #        x, y, r = find_well(img, diameter, n, m)
    #        x_dif = abs(100 - x / well_data[1, 0] * 100)
    #        y_dif = abs(100 - y / well_data[1, 1] * 100)
    #        r_dif = abs(100 - r / well_data[1, 2] * 100)

    well_data[0, 0] = x
    well_data[0, 1] = y
    well_data[0, 2] = r

    # Account for the case where no well can be found
    if a == 5 or well_data[0, 0] == 0:
        well_found = False
    else:
        well_found = True
    print("- well found:", well_found)

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

    liste = [0, 1, 2, 3, 4, 5, 6]
    for j in liste:
        hor_list = []
        vert_list = []
        for d in adjacent:
            horizontal, vertical = set_profile_plots2(img, N, E, S, W, x, y, r, d, j)
            hor_list.append(horizontal)
            vert_list.append(vertical)

        # print("hor", hor_list)
        for part in zip(*hor_list):
            hor_sum.append(sum(part))

        for part in zip(*vert_list):
            vert_sum.append(sum(part))
    #
    # not needed at implementing stage
    length_hor = 0
    length_vert = 0
    length_array = 0
    return hor_sum, vert_sum


def normalise_profile_plot_length(norm_peaks, centerpoints_rel, m):
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

        if centerpoints_rel[0, m] != 0:
            mid = int(centerpoints_rel[0, m])
        else:
            mid = int(len(temp) / 2)
        cubic_norm_peaks[n] = np.insert(temp, mid, add)
    cubic_norm_peaks[3] = norm_peaks[3]

    return cubic_norm_peaks


def compute_droplet_from_peaks(x: int, y: int, r: int, f: float, pp_arrays: np.ndarray, centerpoints_rel: np.ndarray,
                               n: int, radius_old: np.ndarray, radius_droplet_old: int, avg_sum_prev: int):

    # n can be 0 for horizontal or 1 for vertical
    edges_idx = np.zeros(shape=(2, 2))
    start_x = int(x - r * f)  # x-values where the horizontal lines start
    start_y = int(y - r * f)  # y-values where the horizontal lines start
    mid_rel_pp_plots = np.zeros(shape=(2, 7))
    dia_temp = np.zeros(7)
    delta_midpoints = np.zeros(7)
    droplet_coordinates = np.zeros(shape=(7, 2, 2))  # [line_nr, edge 1/2, x/y values]

    # accounts for a moving droplet center
    if centerpoints_rel[0, n] == 0:
        mid = int(len(pp_arrays[3]) / 2)
    else:
        centerpoints_rel[1, n] = centerpoints_rel[0, n]
        mid = int(centerpoints_rel[1, n])
    print("mid_start:", mid)

    # "walk" right/left from center until value is equal 0, save idx, this is our edge
    for j in range(len(pp_arrays)):
        for i in range(1, len(pp_arrays[j])-mid):
            if pp_arrays[j][mid + i] == 0:
                edges_idx[n, 1] = mid + i
                break

        for i in range(1, mid):
            if pp_arrays[j][mid - i] == 0:
                edges_idx[n, 0] = mid - i
                break
        
        droplet_coordinates = compute_coordinates(edges_idx, x, y, r, f, j, n, droplet_coordinates)  # FOR TESTING

        mid_rel_pp_plots[n, j] = int((edges_idx[n, 0] + edges_idx[n, 1]) / 2)
        print(n, j, "L:", edges_idx[n, 0], "R:", edges_idx[n, 1], "m:", mid_rel_pp_plots[n, j])
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
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)

    # calculation of the reduced coordinates
    u = x - x_m
    v = y - y_m

    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u * v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)

    # Solving the linear system
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
    uc, vc = np.linalg.solve(A, B)

    xc_1 = x_m + uc
    yc_1 = y_m + vc

    Ri_1     = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1      = np.mean(Ri_1)
    residu_1 = np.sum((Ri_1-R_1)**2)
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


def calculate_droplet(circle_data: list):
    # calculate droplet values from remaining entries in circle_data
    x_values = []
    y_values = []
    r_values = []
    for j in range(7):
        if circle_data[j] != 0:
            x_values.append(circle_data[j][0])
            y_values.append(circle_data[j][1])
            r_values.append(circle_data[j][2])
    x_droplet = int(np.sum(x_values) / len(x_values))
    y_droplet = int(np.sum(y_values) / len(y_values))
    r_droplet = int(np.sum(r_values) / len(r_values))
    
    return x_droplet, y_droplet, r_droplet


def filter_edges(circle_data: list, radius_old, avg_sum_prev: int):
    # filter edge values derived from profile plots
    # remove all values, that don't correspond to the droplet
    # ToDO: adjust for case where circle_data is 0
    print("HERE IS THE CIRCLE DATA")
    print(circle_data)
    # remove all values where radius is bigger than the previous radius
    if radius_old != 0:
        for j in range(7):
            if circle_data[j][2] > radius_old * 1.1 or circle_data[j][2] < radius_old * 0.7:
                circle_data[j] = 0
    print("AFTER COMPARING TO PREVIOUS RADIUS", "r_prev:", radius_old)
    print(circle_data)
    # remove all values that are bigger or smaller than the average x/y center point
    if np.any(circle_data) != 0:
        avg_sum_centerpoint = calculate_avg_sum_xy(circle_data)
        for j in range(7):
            if circle_data[j] != 0 and avg_sum_prev == 0:
                sum_centerpoint = circle_data[j][0] + circle_data[j][1]
                if avg_sum_centerpoint * 1.2 < sum_centerpoint or avg_sum_centerpoint * 0.8 > sum_centerpoint:
                    print("avg:", avg_sum_centerpoint, "uL:", (avg_sum_centerpoint * 1.1), "lL:", (avg_sum_centerpoint*0.9))
                    circle_data[j] = 0
            if circle_data[j] != 0 and avg_sum_prev != 0:
                sum_centerpoint = circle_data[j][0] + circle_data[j][1]
                if avg_sum_prev * 1.2 < sum_centerpoint or avg_sum_prev * 0.8 > sum_centerpoint:
                    circle_data[j] = 0

        print("AFTER FILTERING FOR CIRCLE CENTERPOINT", "avg_sum_prev:", avg_sum_prev)
    print(circle_data)
    # remove all values that are bigger or smaller than the average radius
    if np.any(circle_data) != 0:
        if radius_old == 0:
            avg_radius = calculate_avg_radius(circle_data)
        else:
            avg_radius = int((calculate_avg_radius(circle_data) + radius_old * 0.95) / 2)
        for j in range(7):
            if np.count_nonzero(circle_data) > 2:  # if only two values, variance can be too high
                if circle_data[j] != 0:
                    if avg_radius * 1.15 < circle_data[j][2] or avg_radius * 0.85 > circle_data[j][2]:
                        circle_data[j] = 0

        print("AFTER FILTERING FOR RADII", "avg_r:", avg_radius)
    print(circle_data)
    # update avg sum
    if np.any(circle_data) != 0:
        avg_sum_centerpoint = calculate_avg_sum_xy(circle_data)
    else:
        avg_sum_centerpoint = 0

    return circle_data, avg_sum_centerpoint


def calculate_avg_radius(circle_data: list):
    radius_list = []

    for j in range(7):
        if circle_data[j] != 0:
            radius_list.append(circle_data[j][2])
    avg_radius = int(np.sum(radius_list) / np.count_nonzero(radius_list))

    return avg_radius


def calculate_avg_sum_xy(circle_data: list):
    sum_xy = []

    for j in range(7):
        if circle_data[j] != 0:
            sum_xy.append(circle_data[j][0] + circle_data[j][1])
    avg_sum_xy = int(np.sum(sum_xy) / np.count_nonzero(sum_xy))

    return avg_sum_xy


def compute_circles(droplet_coordinates: np.ndarray):
    # select droplet coordinates and feed them into find_circles()
    # take two points of the same line, and one from an adjacent one

    # circle_data([x_coord, y_coord, r_value])
    circle_data = []

    for i in range(7):
        if i < 6:
            idx = i+1
        else:
            idx = i-1
        x1 = droplet_coordinates[i, 0, 0]
        y1 = droplet_coordinates[i, 0, 1]
        x2 = droplet_coordinates[i, 1, 0]
        y2 = droplet_coordinates[i, 1, 1]
        x3 = droplet_coordinates[idx, 0, 0]
        y3 = droplet_coordinates[idx, 0, 1]

        x_circle, y_circle, r_circle = find_circle(x1, y1, x2, y2, x3, y3)
        if x_circle != 0:
            circle_data.append([x_circle, y_circle, r_circle])
        else:
            circle_data.append(0)

    return circle_data


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
        print(droplet_coordinates[line_number, 0, 0], droplet_coordinates[line_number, 0, 1], "//", droplet_coordinates[line_number, 1, 0], droplet_coordinates[line_number, 1, 1])

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


def find_circle(x1, y1, x2, y2, x3, y3):
    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    y31 = y3 - y1
    y21 = y2 - y1

    x31 = x3 - x1
    x21 = x2 - x1

    # x1^2 - x3^2
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2
    sy13 = pow(y1, 2) - pow(y3, 2)

    sx21 = pow(x2, 2) - pow(x1, 2)
    sy21 = pow(y2, 2) - pow(y1, 2)

    f = ((sx13 * x12 + sy13 * x12 + sx21 * x13 + sy21 * x13) // (2 * (y31 * x12 - y21 * x13)))

    g = ((sx13 * y12 + sy13 * y12 + sx21 * y13 + sy21 * y13) // (2 * (x31 * y12 - x21 * y13)))

    c = (-pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
    # where centre is (h = -g, k = -f) and
    # radius r as r^2 = h^2 + k^2 - c
    # print("inf", np.isinf(f), np.isinf(g), "nan", np.isnan(f), np.isnan(g))
    if np.isinf(f) == False and np.isinf(g) == False and np.isnan(f) == False and np.isnan(g) == False:
        h = int(-g)
        k = int(-f)
        sqr_of_r = h * h + k * k - c
        # r is the radius
        r = int(round(np.sqrt(sqr_of_r), 5))
    else:
        h = 0
        k = 0
        r = 0

    # print("Centre = (", h, ", ", k, ")")
    # print("Radius = ", r)

    # source: https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/

    return h, k, r


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
def LLPS_detection(mean_of_current_image, percental_threshold, areas, droplet_arr, mean_list):
    mean_abs = mean_of_current_image

    if len(mean_list) > 1:
        avg_mean_all_previous_images = np.mean(mean_list)
    else:
        avg_mean_all_previous_images = mean_abs

    # Calculate percental difference between current mean value and average mean of all previous images
    percental_difference = (mean_abs / avg_mean_all_previous_images) * 100 - 100
    print("perc. diff.: ", percental_difference)

    if percental_difference > percental_threshold:
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

