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

import os
import cv2
import numpy as np
import csv
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from pyphasechip import pyphasechip_fun as fun


def create():
    image_list = []
    image_names = []
    data_well = {}
    well = {}
    concentration = {}
    time_resolution = {}
    return image_list, image_names, data_well, well, concentration, time_resolution


# save images to list
def images_to_list(image_list, image_names, image_folder, extension):
    # TODO: Add returns
    print("Creating and sorting image list")
    time.sleep(0.2)
    file_list = []
    for file in os.listdir(image_folder):
        file_list.append(os.path.join(image_folder, file))
    file_list.sort(key=lambda x: os.path.getmtime(x))

    for image in tqdm(file_list):
        filename, ext = os.path.splitext(image)

        if ext.lower() == extension:
            image_list.append(cv2.imread(os.path.join(image_folder, image)))
            filename = (os.path.basename(filename))
            image_names.append(filename)


# save images to list
def images_to_list_backup(image_list, image_names, image_folder, extension):
    for image in sorted(os.listdir(image_folder)):
        filename, ext = os.path.splitext(image)
        # print(filename)
        if ext.lower() == extension:
            image_list.append(cv2.imread(os.path.join(image_folder, image)))
            image_names.append(filename)


# rewrite them into a dict of dicts of dicts of dicts
def images_to_dict(h, iph, n_concentrations, n_wells, image_list, image_names, bigdict, concentration, well, data_well):
    print("writing images into big dictionary")
    time.sleep(0.5)
    well_nr = 0
    n = 0
    for time_idx in tqdm(range(int(h * iph))):
        for conc_nr in range(n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    raw_image = image_list[n]
                    gray_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                    gray_image = cv2.convertScaleAbs(gray_image, alpha=0.9, beta=50)
                    gray_image = fun.controller(gray_image, brightness=252, contrast=140)
                    data_well['name'] = image_names[n]
                    data_well['raw'] = raw_image
                    data_well['gray'] = gray_image

                    well[well_nr] = data_well.copy()
                    concentration[conc_nr] = well.copy()
                    bigdict[time_idx] = concentration.copy()

                    well_nr += 1
                    n += 1

                    # print('Name:', data_well['name'])

            well_nr = 0
        well_nr = 0


# Detect the wells and the droplets within them
# create mask when necessary
def droplet_detection(diameter, imgage, saved_points, elon_mask, centerpoints, llps_status, r_old):

    print("detect wells & droplets and create masks")
    time.sleep(0.5)

    ## image manipulation

    # find well and prepare for droplet detection
    x, y, r, saved_points, well_found = fun.find_well_algo(imgage, saved_points, diameter, dev=10)

    if well_found is True and llps_status is False:
        mask, saved_points = fun.create_mask(imgage.copy(), elon_mask, saved_points, x, y, r, dev=2)
        masked_img = fun.mask_image(imgage.copy(), mask)

        img = fun.image_manipulation(masked_img, x, y, r)

        ## detect droplet

        f, N, E, S, W = fun.calculate_profile_plot_coordinates(x, y, r)
        l, l_vert, length_arr, horizontal, vertical = fun.set_profile_plots(img, N, E, S, W, x, y, r)

        # creating dictionary
        # (its a dict for historical reasons, could also be an array)
        # droplet_var[keyword][line_number]
        droplet_var_h = {}
        droplet_var_v = {}
        keywords = ['normalised']

        # normalise plots
        for key in keywords:
            droplet_var_h[key] = {}
            droplet_var_v[key] = {}

        norm_pp_len_h = fun.normalise_profile_plot_length(horizontal, centerpoints, 0)
        x_abs, centerpoints, diameter_x, droplet_found_x = fun.compute_droplet_from_peaks(x, y, r, f, norm_pp_len_h, centerpoints, 0, r_old)
        norm_pp_len_v = fun.normalise_profile_plot_length(vertical, centerpoints, 1)
        y_abs, centerpoints, diameter_y, droplet_found_y = fun.compute_droplet_from_peaks(x, y, r, f, norm_pp_len_v, centerpoints, 1, r_old)

        if droplet_found_y is True and droplet_found_x is True:
            if diameter_y < diameter_x:
                radius_est = int((diameter_y / 2) * 0.99)
            else:
                radius_est = int((diameter_x / 2) * 0.99)
            droplet_found = True
        else:
            droplet_found = False
            radius_est = 0
        print("radius est:", radius_est)
        r_old[0] = int(diameter_x/2)
        r_old[1] = int(diameter_y/2)

    else:
        x_abs = 0
        y_abs = 0
        radius_est = 0
        mask = 0
        masked_img = 0
        droplet_found = False
        l = 0
        l_vert = 0
        norm_pp_len_h = 0
        norm_pp_len_v = 0
        img = 0
        f = 0
        N = 0
        W = 0
        E = 0
        S = 0


    print("- droplet found:", droplet_found)

    return x_abs, y_abs, centerpoints, radius_est, saved_points, mask, masked_img, droplet_found, l, l_vert, norm_pp_len_h, norm_pp_len_v, img, f, N, E, S, W, x, y, r, r_old


# Detect LLPS
# loop over ALL the images
def detect_LLPS(percental_threshold, x_abs, y_abs, radius_est, droplet_arr, llps_status, manip_img, t, areas,
                mean_list, droplet_found):
    print("LLPS detection")
    time.sleep(0.5)

    droplet_arr[0, 0] = radius_est
    droplet_arr[0, 1] = x_abs
    droplet_arr[0, 2] = y_abs

    if droplet_found is True and llps_status is False:

        # fallback if LLPS makes good droplet recognition impossible
        if droplet_arr[1, 0] != 0 and abs(droplet_arr[0, 0]/droplet_arr[1, 0] * 100 - 100) > 15:
            radius_est = int(droplet_arr[1, 0] * 0.9)
            x_abs = int(droplet_arr[1, 1])
            y_abs = int(droplet_arr[1, 2])
            print("Fallback:", x_abs, y_abs)
        else:
            droplet_arr[1, 0] = radius_est
            droplet_arr[1, 1] = x_abs
            droplet_arr[1, 2] = y_abs


        # calculate minimal distance from droplet center to edge
        minimal_distance = radius_est * 0.9  # 90% of droplet radius
        area_droplet = 3.14159 * radius_est**2

        # determine threshold value
        thresh_val = 215

        #blurred_cur = cv2.blur(masked_img.copy(), (4, 4))
        #ret, threshed_img_cur = cv2.threshold(blurred_cur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # save pixel values within squircle inside droplet
        #squircled_pixels = fun.squircle_iteration(subtracted_img, int(x_abs), int(y_abs), int(minimal_distance))
        if t > 0:
            squircled_pixels = fun.squircle_iteration(manip_img, x_abs, y_abs, int(minimal_distance))
            # calculate mean of pixel values
            #mean = (np.sum(squircled_pixels) / np.count_nonzero(squircled_pixels))

            # First detetion method
            # counts zeros in squircle, feed "n" into detector
            # set threshold to 40% or so
            d = int(0.61 * radius_est)
            cropped_squircled_pixels = squircled_pixels[y_abs - d:y_abs + d, x_abs - d:x_abs + d]
            n = 0
            for idx, value in enumerate(cropped_squircled_pixels):
                for i, val in enumerate(cropped_squircled_pixels[idx]):
                    if val == 0:
                        n += 1
            # print("count zeros", n)
            # Second method
            # set threshold to 30%
            x, y = squircled_pixels.shape
            count_total = (x * y)
            mean = (np.sum(squircled_pixels)) / count_total
            #print("x:", x_abs, "y: ", y_abs, "r: ", radius_est, "min_dis.: ", minimal_distance, "0_count: ", n)
            # Detector
            llps_status, areas, mean_list = fun.LLPS_detection(n, percental_threshold, area_droplet, areas,
                                                               mean_list)

        else:
            squircled_pixels = 0

        # blurred_img_prev = blurred_cur
    else:
        squircled_pixels = 0

    return llps_status, areas, mean_list, droplet_arr, squircled_pixels


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


def ccrit_calculation(c_start, areas, conc_nr):

    # calculate c_crit
    # c_crit = area_start/area_end*c_start

    llps_conc = np.zeros(shape=(1, 2))

    llps_conc[0, 0] = areas[0, 0]/areas[0, 1] * c_start[conc_nr, 0]
    llps_conc[0, 1] = areas[0, 0]/areas[0, 1] * c_start[conc_nr, 1]

    return llps_conc


def quality_control(bigdict, time_idx, conc_nr, well_nr, name_sol1, name_sol2, unit_sol1, unit_sol2,
                    starting_concentrations, circles):
    # Status
    print("Well found: ", bigdict[0][conc_nr][well_nr]['well status'])
    print("Droplet found: ", bigdict[0][conc_nr][well_nr]['droplet status'])
    print("LLPS found: ", bigdict[0][conc_nr][well_nr]['LLPS status'])
    print("file name: ", bigdict[0][conc_nr][well_nr]['LLPS name'])
    print(" ")
    print('initial concentrations:')
    print(name_sol1, '', name_sol2)
    print(starting_concentrations)

    # concentrations and areas
    if bigdict[0][conc_nr][well_nr]['well status'] is True \
            and bigdict[0][conc_nr][well_nr]['LLPS status'] is True:
        print(' ')
        print('LLPS concentrations:')
        print(f"{name_sol1}", f"- conc.: {round((bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 0]), 3)} {unit_sol1}")
        print(f"{name_sol2}", f"- conc.: {round((bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 1]), 3)} {unit_sol2}")
        print(" ")
        print("Areas:", int(bigdict[0][conc_nr][well_nr]['areas'][0, 0]), " ",
              int(bigdict[0][conc_nr][well_nr]['areas'][0, 1]))

    fig = plt.figure(figsize=(15, 30))
    ax1 = fig.add_subplot(5,2,1)
    ax1.set_title('gray img of requested t')
    ax1.imshow(bigdict[time_idx][conc_nr][well_nr]['gray'].copy())

    if circles is not None:
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = (i[2])
            circles_img = cv2.circle(bigdict[0][conc_nr][well_nr]['gray'].copy(), center, radius, (0, 0, 0), 1)

            ax2 = fig.add_subplot(5, 2, 2)
            ax2.set_title('detected well')
            ax2.imshow(circles_img)

            if bigdict[0][conc_nr][well_nr]['droplet status']:
                ax3 = fig.add_subplot(5, 2, 3)
                for i in bigdict[0][conc_nr][well_nr]['droplet geometry'][0, :]:
                    center = (i[0], i[1])
                    # circle outline
                    radius = (i[2])
                    droplet_img = cv2.circle(bigdict[0][conc_nr][well_nr]['gray'].copy(), center, radius, (0, 0, 0), 1)

                    ax3.set_title('found droplet at t0')
                    ax3.imshow(droplet_img)

            ax4 = fig.add_subplot(5, 2, 4)
            ax4.plot(np.arange(len(bigdict[0][conc_nr][well_nr]['mean list'])),
                     bigdict[0][conc_nr][well_nr]['mean list'], marker='o', label="mean values")
            ax4.set_xlabel('image nr')
            ax4.set_ylabel('avg. mean of droplet')
            ax4.set_ylim([min(bigdict[0][conc_nr][well_nr]['mean list'])-0.5,
                          max(bigdict[0][conc_nr][well_nr]['mean list'])+0.5])
            ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax4.set_xlim([0, len(bigdict[0][conc_nr][well_nr]['mean list'])-1])
        if bigdict[0][conc_nr][well_nr]['droplet status'] is True:

            if time_idx > 0:
                ax8 = fig.add_subplot(5, 2, 8)
                ax8.set_title('subtraction result')
                ax8.imshow(bigdict[time_idx][conc_nr][well_nr]['thresh'], cmap='gray')
                ax9 = fig.add_subplot(5, 2, 9)
                ax9.set_title('squircle result')
                ax9.imshow(bigdict[time_idx][conc_nr][well_nr]['pixel values'])

    print("List of Means")
    print(bigdict[0][conc_nr][well_nr]['mean list'])

    if bigdict[0][conc_nr][well_nr]['well status'] is True \
            and bigdict[0][conc_nr][well_nr]['LLPS status'] is True:

        fig2, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(15, 15))
        # before LLPS detections
        A_10 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 3)][conc_nr][well_nr]['masked image'].copy()
        A_11 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 2)][conc_nr][well_nr]['masked image'].copy()
        A_12 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 1)][conc_nr][well_nr]['masked image'].copy()
        # at LLPS detection
        grimes = bigdict[(bigdict[0][conc_nr][well_nr]['ID'])][conc_nr][well_nr]['gray'].copy()

        ax11.imshow(A_10, cmap='gray')
        ax11.set_title('three t before LLPS detection')
        ax12.imshow(A_11, cmap='gray')
        ax12.set_title('two t before LLPS detection')
        ax13.imshow(A_12, cmap='gray')
        ax13.set_title('one t before LLPS detection')
        ax14.imshow(grimes, cmap='gray')
        ax14.set_title('at LLPS detection')


def save_results_to_csv(bigdict, image_folder, n_concentrations, n_wells, h, iph, name_sol1, name_sol2, unit_sol1, unit_sol2):
    well_nr = 0
    pathtocsv = os.path.join(image_folder, "csv")

    try:
        os.mkdir(pathtocsv)
    except OSError as e:
        print("Attention: Directory where .csv gets saved already exists")

    csvname = f"results_{name_sol1}_{name_sol2}.csv"

    with open(os.path.join(pathtocsv, csvname), 'w', newline='') as csvFile:
        writer = csv.writer(csvFile, delimiter=';', dialect='excel', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["Image names", f"LLPS conc. {name_sol1} [{unit_sol1}]",
                         f"LLPS conc. {name_sol2} [{unit_sol2}]"])
        writer.writerow(" ")

        for conc_nr in range(n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    if bigdict[0][conc_nr][well_nr]['well status'] is True and \
                            bigdict[0][conc_nr][well_nr]['LLPS status'] is True:
                        writer.writerow(
                            [bigdict[0][conc_nr][well_nr]['LLPS name'], bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 0],
                             bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 1]])

                    well_nr += 1
            well_nr = 0

        writer.writerow(" ")
        writer.writerow(" ")
        writer.writerow(["If you use natively a ',' as a decimal separator, you probably need/"
                         "to change it for correct display of numbers"])
        writer.writerow(["In excel you can do this via File -> Options -> Advanced, here you can change separators"])

