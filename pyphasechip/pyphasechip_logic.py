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

from pyphasechip import pyphasechip_fun


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


# Detect the reaction chamber and create a specific mask
# only necessary on the first image
# mask will be applied in detect_LLPS
def chamber_detection_and_mask_creation(n_concentrations, n_wells, bigdict, min_r_chamber, max_r_chamber):
    well_nr = 0
    print("detect chambers and create masks")
    time.sleep(0.5)
    for conc_nr in tqdm(range(n_concentrations)):
        for n_rows_per_conc in range(2):
            for n_wells_per_row in range(n_wells):
                circles = cv2.HoughCircles(bigdict[0][conc_nr][well_nr]['gray'], cv2.HOUGH_GRADIENT, dp=1,
                                           minDist=50, param1=10,
                                           minRadius=min_r_chamber, maxRadius=max_r_chamber)
                # Draw circles
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle outline
                        radius = (i[2]) # subtract as safety
                        image = cv2.circle(bigdict[0][conc_nr][well_nr]['gray'].copy(), center, radius, (0, 0, 0), 1)
                        plt.subplots(figsize=(15, 15))
                        plt.subplot(141)
                        plt.title('circles')
                        plt.imshow(image, cmap='gray')

                if circles is not None:
                    # calculate mean
                    bigdict[0][conc_nr][well_nr]['init_mean'] = np.mean(bigdict[0][conc_nr][well_nr]['gray'])

                    # create mask
                    bigdict[0][conc_nr][well_nr]['elon'] = \
                        pyphasechip_fun.create_mask(bigdict[0][conc_nr][well_nr]['gray'].copy(), circles)

                    # apply mask
                    bigdict[0][conc_nr][well_nr]['masked image'] = pyphasechip_fun.mask_image(
                        bigdict[0][conc_nr][well_nr]['gray'].copy(),
                        bigdict[0][conc_nr][well_nr]['elon'])

                    # detect contours in the image
                    contours, _, _ = pyphasechip_fun.detect_contours(bigdict[0][conc_nr][well_nr]['masked image'],
                                                                     bigdict[0][conc_nr][well_nr]['init_mean'])

                    # select droplet, and get all necessary information from it
                    area_droplet, _, _, _ = pyphasechip_fun.select_droplet(contours)

                    # save area of droplet to array
                    bigdict[0][conc_nr][well_nr]['areas'] = np.zeros(shape=(1, 2))
                    bigdict[0][conc_nr][well_nr]['areas'][0, 0] = area_droplet

                    bigdict[0][conc_nr][well_nr]['mean list'] = []

                    # no LLPS yet (hopefully), save this information for processing later
                    bigdict[0][conc_nr][well_nr]['LLPS status'] = False
                    bigdict[0][conc_nr][well_nr]['well found'] = True

                else:
                    bigdict[0][conc_nr][well_nr]['well found'] = False

                # print('Name: 'f"{dict[0][conc_nr][well_nr]['name']}" ' Area: ', dict[0][conc_nr][well_nr]['areas'])

                well_nr += 1
            #well_nr += 0
        well_nr = 0
    well_nr = 0


# Detect LLPS
# loop over ALL the images
def detect_LLPS(h, iph, n_concentrations, n_wells, bigdict, percental_threshold):
    print("LLPS detection")
    time.sleep(0.5)
    well_nr = 0
    for time_idx in tqdm(range(1, int(h * iph) + 1)):
        for conc_nr in range (n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    if bigdict[0][conc_nr][well_nr]['well found'] is True \
                            and bigdict[0][conc_nr][well_nr]['LLPS status'] is False:

                        # mask image
                        bigdict[time_idx][conc_nr][well_nr]['masked image'] = pyphasechip_fun.mask_image(
                            bigdict[time_idx][conc_nr][well_nr]['gray'].copy(),
                            bigdict[0][conc_nr][well_nr]['elon'])

                        # detect contours in the image
                        bigdict[0][conc_nr][well_nr]['contours'], safespot1, safespot2 = pyphasechip_fun.detect_contours(
                            bigdict[time_idx][conc_nr][well_nr]['masked image'],
                            bigdict[0][conc_nr][well_nr]['init_mean'])
                        bigdict[0][conc_nr][well_nr]['thresh'] = safespot1
                        bigdict[0][conc_nr][well_nr]['dilateanderode'] = safespot2

                        # select droplet, and get all necessary information from it
                        area_droplet, cX_droplet, cY_droplet, contour_droplet = pyphasechip_fun.select_droplet(
                            bigdict[0][conc_nr][well_nr]['contours'])
                        bigdict[0][conc_nr][well_nr]['contours droplet'] = contour_droplet

                        # calculate minimal distance from droplet center to edge
                        bigdict[time_idx][conc_nr][well_nr]['minimal distance'] = pyphasechip_fun.minDistance(
                            contour_droplet,
                            cX_droplet,
                            cY_droplet)

                        if time_idx > 1:
                            # subtract current img from old
                            bigdict[time_idx][conc_nr][well_nr]['subtracted'] = cv2.subtract(
                                bigdict[time_idx][conc_nr][well_nr]['masked image'],
                                bigdict[(time_idx - 1)][conc_nr][well_nr]['masked image'])

                            # calculate pixel values within squircle inside droplet
                            bigdict[time_idx][conc_nr][well_nr]['pixel values'] = pyphasechip_fun.squircle_iteration(
                                bigdict[time_idx][conc_nr][well_nr]['subtracted'], cX_droplet, cY_droplet,
                                int(bigdict[time_idx][conc_nr][well_nr]['minimal distance']))

                            # calculate mean of pixel values
                            mean = (np.sum(bigdict[time_idx][conc_nr][well_nr]['pixel values']) /
                                    np.count_nonzero(bigdict[time_idx][conc_nr][well_nr]['pixel values']))

                            # Detector
                            pyphasechip_fun.LLPS_detection(mean, percental_threshold, area_droplet,
                                                           bigdict, time_idx, conc_nr, well_nr)
                            # dict[0][conc_nr][well_nr]['mean list'],
                            # print('Name: 'f"{dict[time_idx][conc_nr][well_nr]['name']}" ' Area: ', dict[
                            # 0][conc_nr][well_nr]['areas'])

                    well_nr += 1
            well_nr = 0
        well_nr = 0


def ccrit_calculation(initc_sol1, initc_sol2, init_ratio, h, iph, n_concentrations, n_wells, bigdict):
    well_nr = 0
    # calculate starting concentrations
    starting_concentrations = pyphasechip_fun.starting_concentration(initc_sol1, initc_sol2, init_ratio)

    # calculate c_crit
    print("calculation of the critical concentration")
    time.sleep(0.5)
    for time_idx in tqdm(range(int((h * iph) + 1))):
        for conc_nr in range (n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    if bigdict[0][conc_nr][well_nr]['well found'] is True and \
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
                #well_nr += 1
            well_nr = 0
        well_nr = 0

    return starting_concentrations


def quality_control(bigdict, conc_nr, well_nr, name_sol1, name_sol2, unit_sol1, unit_sol2, starting_concentrations):
    print('initial concentrations:')
    print(name_sol1, '', name_sol2)
    print(starting_concentrations)

    if bigdict[0][conc_nr][well_nr]['well found'] is True \
            and bigdict[0][conc_nr][well_nr]['LLPS status'] is True:
        print(' ')
        print('LLPS concentrations:')
        print(f"{name_sol1}", f"- conc. {conc_nr}: {round((bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 0]), 3)} {unit_sol1}")
        print(f"{name_sol2}", f"- conc. {conc_nr}: {round((bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 1]), 3)} {unit_sol2}")

    gray_img = bigdict[0][conc_nr][well_nr]['raw'].copy()

    plt.subplots(figsize=(50, 50))
    plt.subplot(141)
    plt.imshow(gray_img, cmap='binary')
    plt.title('raw image')

    if bigdict[0][conc_nr][well_nr]['well found'] is True:

        musk_img = gray_image = bigdict[0][conc_nr][well_nr]['elon'].copy()
        musked_img = bigdict[0][conc_nr][well_nr]['masked image'].copy()

        plt.subplot(142)
        plt.imshow(musked_img, cmap='gray')
        plt.title('masked image')

        if bigdict[0][conc_nr][well_nr]['LLPS status'] is True:

            # draw elon masked image with contour tattoos
            A_12 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 1)][conc_nr][well_nr]['masked image'].copy()
            contour_img = cv2.drawContours(A_12, bigdict[0][conc_nr][well_nr]['contours droplet'], -1, (255, 165, 0), 5)

            # draw elon masked LLPS image with contour tattoos
            grimes = bigdict[(bigdict[0][conc_nr][well_nr]['ID'])][conc_nr][well_nr]['masked image'].copy()
            contour_img_LLPS = cv2.drawContours(grimes, bigdict[0][conc_nr][well_nr]['contours droplet'], -1,
                                                (255, 165, 0), 5)



            plt.subplot(241)
            plt.imshow(contour_img, cmap='gray')
            plt.title('detected droplet before LLPS occures')
            plt.subplot(242)
            plt.imshow(contour_img_LLPS, cmap='gray')
            plt.title('detected droplet when LLPS occures')

        else:
            plt.subplot(142)
            plt.imshow(bigdict[(len(bigdict) - 2)][conc_nr][well_nr]['gray'], cmap='gray')
            plt.title('third last image')
            print(' ')
            print("No LLPS could be detected in this well :(")
            print(' ')
    else:
        print(' ')
        print("No well could be detected :(")
        print(' ')


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
                    if bigdict[0][conc_nr][well_nr]['well found'] is True and \
                            bigdict[0][conc_nr][well_nr]['LLPS status'] is True:
                        writer.writerow(
                            [bigdict[0][conc_nr][well_nr]['LLPS name'], bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 0],
                             bigdict[0][conc_nr][well_nr]['LLPS conc'][0, 1]])

                    well_nr += 1
                #well_nr += 1
            well_nr = 0

        writer.writerow(" ")
        writer.writerow(" ")
        writer.writerow(["If you use natively a ',' as a decimal separator, you propably need/"
                         "to change it for correct display of numbers"])
        writer.writerow(["In excel you can do this via File -> Options -> Advanced, here you can change separators"])
