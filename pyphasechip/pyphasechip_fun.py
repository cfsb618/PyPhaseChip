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
    radius = circles_from_hc_detection[0, 0, 2]

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
def detect_contours(elon_mask):
    _, image = cv2.threshold(elon_mask, 150, 255, cv2.THRESH_BINARY_INV)

    # Solidify faint droplet edges
    kernel = np.ones((8, 8), np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)
    image = cv2.erode(image, kernel, iterations=3)

    # and make them rly nice & thick
    kernel_nu = np.ones((7, 7), np.uint8)
    image = cv2.dilate(image, kernel_nu, iterations=1)

    # this is the core: contour detection
    contours_within_elon, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if somehow the image is not recognised as a contour, use cv2.RETR_EXTERNAL

    return contours_within_elon


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
        dict[0][lane_nr][well_nr]['LLPSname'] = dict[time_idx][lane_nr][well_nr]['name']
        # save area to array
        dict[0][lane_nr][well_nr]['areas'][0, 1] = area_droplet

    else:
        dict[0][lane_nr][well_nr]['LLPS status'] = False
        # save mean to mean list
        dict[0][lane_nr][well_nr]['mean list'].append(mean_of_current_image)
