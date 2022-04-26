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
def droplet_detection(diameter, imgage, saved_points, elon_mask, centerpoints):

    print("detect wells & droplets and create masks")
    time.sleep(0.5)

    ## image manipulation

    # find well and prepare for droplet detection
    x, y, r, saved_points, well_found = fun.find_well_algo(imgage, saved_points, diameter, dev=10)

    if well_found is True:
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

        #for idx, val in enumerate(horizontal):
        #    droplet_var_h['normalised'][idx] = np.array(val) / np.mean(np.array(val))

        #for idx, val in enumerate(vertical):
        #    droplet_var_v['normalised'][idx] = np.array(val) / np.mean(np.array(val))

        norm_pp_len_h = fun.normalise_profile_plot_length(horizontal)
        x_abs, centerpoints, delta_x, droplet_found_x = fun.compute_droplet_from_peaks(x, y, r, f, norm_pp_len_h, centerpoints, n=0)
        norm_pp_len_v = fun.normalise_profile_plot_length(vertical)
        y_abs, centerpoints, delta_y, droplet_found_y = fun.compute_droplet_from_peaks(x, y, r, f, norm_pp_len_v, centerpoints, n=1)

        if droplet_found_y is True and droplet_found_x is True:
            if delta_y < delta_x:
                radius_est = int((delta_y / 2) * 0.99)
            else:
                radius_est = int((delta_x / 2) * 0.99)
            droplet_found = True
        else:
            droplet_found = False
            radius_est = 0

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

    return x_abs, y_abs, centerpoints, radius_est, saved_points, mask, masked_img, droplet_found, l, l_vert, norm_pp_len_h, norm_pp_len_v, img, f, N, E, S, W, x, y, r, horizontal


# Detect LLPS
# loop over ALL the images
def detect_LLPS(percental_threshold, x_abs, y_abs, radius_est, llps_status, masked_img, t, threshed_img_prev, areas,
                mean_list, droplet_found):
    print("LLPS detection")
    time.sleep(0.5)

    if droplet_found is True and llps_status is False:

        # calculate minimal distance from droplet center to edge
        minimal_distance = radius_est * 0.8  # 80% of droplet radius
        area_droplet = 3.14159 * radius_est**2

        #bigdict[time_idx][conc_nr][well_nr]['minimal distance'] = fun.minDistance(
        #    contour_droplet, cX_droplet, cY_droplet)

        # adjust contrast
        #contrasted_current = cv2.convertScaleAbs(
        #    bigdict[time_idx][conc_nr][well_nr]['masked image'], beta=-40)
        #contrasted_old = cv2.convertScaleAbs(
        #    bigdict[(time_idx - 1)][conc_nr][well_nr]['masked image'], beta=-40)

        # determine threshold value
        thresh_val = 215

        blurred_cur = cv2.blur(masked_img.copy(), (4, 4))
        ret, threshed_img_cur = cv2.threshold(blurred_cur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        if t > 1:

            # subtract current img from old
            subtracted_img = cv2.subtract(threshed_img_cur, threshed_img_prev)

            # calculate pixel values within squircle inside droplet
            # TODO: changed subtracted to thresh, see above
            squircled_pixels = fun.squircle_iteration(subtracted_img, x_abs, y_abs, int(minimal_distance))

            # calculate mean of pixel values
            mean = (np.sum(squircled_pixels) / np.count_nonzero(squircled_pixels))
            print(t, "sum:", sum, "mean:", mean)

            # Detector
            llps_status, areas, mean_list = fun.LLPS_detection(mean, percental_threshold, area_droplet, areas, mean_list)

        threshed_img_prev = threshed_img_cur

    return llps_status, threshed_img_prev, areas, mean_list


def ccrit_calculation(initc_sol1, initc_sol2, init_ratio, h, iph, n_concentrations, n_wells, bigdict):
    well_nr = 0
    # calculate starting concentrations
    starting_concentrations = fun.starting_concentration(initc_sol1, initc_sol2, init_ratio)

    # calculate c_crit
    print("calculation of the critical concentration")
    time.sleep(0.5)
    for time_idx in tqdm(range(int((h * iph) + 1))):
        for conc_nr in range (n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    if bigdict[0][conc_nr][well_nr]['well status'] is True and \
                            bigdict[0][conc_nr][well_nr]['LLPS status'] is True:
                        bigdict[0][conc_nr][well_nr]['LLPS conc'] = np.zeros(shape=(1, 2))
                        # c_crit = area_start/area_end*c_start
                        bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 0] = (
                                bigdict[0][conc_nr][well_nr]['areas'][0, 0] /
                                bigdict[0][conc_nr][well_nr]['areas'][0, 1] * starting_concentrations[
                                    conc_nr, 0])
                        bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 1] = (
                                bigdict[0][conc_nr][well_nr]['areas'][0, 0] /
                                bigdict[0][conc_nr][well_nr]['areas'][0, 1] * starting_concentrations[
                                    conc_nr, 1])

                    well_nr += 1
            well_nr = 0
        well_nr = 0

    return starting_concentrations


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

