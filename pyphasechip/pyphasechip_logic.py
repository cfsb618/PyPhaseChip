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
                    gray_image = pyphasechip_fun.controller(gray_image, brightness=252, contrast=140)
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
    n = 0
    for conc_nr in tqdm(range(n_concentrations)):
        for n_rows_per_conc in range(2):
            for n_wells_per_row in range(n_wells):
                circles = cv2.HoughCircles(bigdict[0][conc_nr][well_nr]['gray'], cv2.HOUGH_GRADIENT, dp=1,
                                           minDist=50, param1=10,
                                           minRadius=min_r_chamber, maxRadius=max_r_chamber)

                n += 1
                # print("run:", n)
                # print("conc", conc_nr, "well", well_nr)

                if circles is None:
                    print("No chamber could be detected, please adjust radius")
                    print("conc", conc_nr, "well", well_nr)
                    bigdict[0][conc_nr][well_nr]['circles'] = False
                    bigdict[0][conc_nr][well_nr]['detected droplet'] = False

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    bigdict[0][conc_nr][well_nr]['circles'] = circles

                    bigdict[0][conc_nr][well_nr]['well found'] = True

                    # calculate mean
                    bigdict[0][conc_nr][well_nr]['init_mean'] = np.mean(bigdict[0][conc_nr][well_nr]['gray'])

                    # create mask
                    bigdict[0][conc_nr][well_nr]['elon'] = \
                        pyphasechip_fun.create_mask(bigdict[0][conc_nr][well_nr]['gray'].copy(), circles)

                    # apply mask
                    bigdict[0][conc_nr][well_nr]['masked image'] = pyphasechip_fun.mask_image(
                        bigdict[0][conc_nr][well_nr]['gray'].copy(),
                        bigdict[0][conc_nr][well_nr]['elon'])

                    ## detect contours in the image
                    #contours, _, _, _ = pyphasechip_fun.detect_contours(bigdict[0][conc_nr][well_nr]['masked image'],
                    #                                                    bigdict[0][conc_nr][well_nr]['init_mean'])
                    #bigdict[0][conc_nr][well_nr]['contours'] = contours

                    ## select droplet, and get all necessary information from it
                    #area_droplet, _, _, contour_droplet, droplet_status = pyphasechip_fun.select_droplet_test(contours, 238)
                    ## TODO: 238 -> jupyter script determinieren lassen
                    #bigdict[0][conc_nr][well_nr]['contours droplet'] = contour_droplet

                    # SWITCHED TO HOUGH FOR TESTING PURPOSES
                    # blur before Mr. Hough is always a good idea
                    temp = cv2.blur(bigdict[0][conc_nr][well_nr]['masked image'].copy(), (2, 2))

                    for i in range(10):
                        a = 0.99
                        minradius = int(bigdict[0][conc_nr][well_nr]['circles'][0, 0, 2] * 0.4)
                        maxradius = int(bigdict[0][conc_nr][well_nr]['circles'][0, 0, 2] * a)

                        detected_droplet = cv2.HoughCircles(temp,
                                                            cv2.HOUGH_GRADIENT, dp=1.1, minDist=150, param1=50,
                                                            param2=40, minRadius=minradius, maxRadius=maxradius)

                        if detected_droplet is not None:
                            detected_droplet = np.uint16(np.around(detected_droplet))
                            droplet_status = True
                            bigdict[0][conc_nr][well_nr]['detected droplet'] = droplet_status
                            bigdict[0][conc_nr][well_nr]['droplet information'] = detected_droplet
                            # print(f"Droplet found in well t{time_idx}_c{conc_nr}_w{well_nr}")
                            break  # end loop when droplet is detected

                        else:
                            a -= 0.01

                        if i == 9:
                            print(f"no droplet could be found in well t{0}_c{conc_nr}_w{well_nr}")
                            # Print error massage if no droplet was found
                            droplet_status = False
                            bigdict[0][conc_nr][well_nr]['detected droplet'] = droplet_status

                    if droplet_status is True:
                        # calculate minimal distance from droplet center to edge
                        bigdict[0][conc_nr][well_nr]['minimal distance'] = \
                            bigdict[0][conc_nr][well_nr]['droplet information'][
                                0, 0, 2] * 0.8  # 80% of droplet radius
                        cX_droplet = bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 0]
                        cY_droplet = bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 1]
                        area_droplet = 3.14159 * bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 2] ** 2
                        radius_droplet = bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 2]
                        radius_well = bigdict[0][conc_nr][well_nr]['circles'][0, 0, 2]

                    # TODO: If hough works, take care of case where first detected droplet is smaller than chamber
                    #  and also set droplet radius here
                    if droplet_status is True:
                        bigdict[0][conc_nr][well_nr]['droplet information'] = circles  # this is set for droplet detection in detect_LLPS
                        bigdict[0][conc_nr][well_nr]['detected droplet'] = True


                    # save area of droplet to array
                    bigdict[0][conc_nr][well_nr]['areas'] = np.zeros(shape=(1, 2))

                    if droplet_status is True and radius_droplet < radius_well:  # TODO: what is this equation?
                        bigdict[0][conc_nr][well_nr]['areas'][0, 0] = area_droplet
                    else:
                        bigdict[0][conc_nr][well_nr]['areas'][0, 0] = (3.141*(238/2)**2)*1.1
                        # TODO: 238 -> jupyter script
                        # if droplet is not found in the first frame, it is assumed that it fills
                        # the whole well and also parts of the channel, hence the 1.1 factor

                    bigdict[0][conc_nr][well_nr]['mean list'] = []

                    # no LLPS yet (hopefully), save this information for processing later
                    bigdict[0][conc_nr][well_nr]['LLPS status'] = False

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
    for time_idx in tqdm(range(3, int(h * iph))):  ##### CHANGED 1 to 3 for HOUGH CIRCLE
        for conc_nr in range(n_concentrations):
            for n_rows_per_conc in range(2):
                for n_wells_per_row in range(n_wells):
                    if bigdict[0][conc_nr][well_nr]['well found'] is True \
                            and bigdict[0][conc_nr][well_nr]['LLPS status'] is False:

                        # mask image
                        bigdict[time_idx][conc_nr][well_nr]['masked image'] = pyphasechip_fun.mask_image(
                            bigdict[time_idx][conc_nr][well_nr]['gray'].copy(),
                            bigdict[0][conc_nr][well_nr]['elon'])

                        # detect contours in the image
                        #bigdict[time_idx][conc_nr][well_nr]['contours'], threshed_img, mopped_up_img, dilated_img = pyphasechip_fun.detect_contours(
                        #    bigdict[time_idx][conc_nr][well_nr]['masked image'],
                        #    bigdict[0][conc_nr][well_nr]['init_mean'])
                        #bigdict[time_idx][conc_nr][well_nr]['thresh'] = threshed_img
                        #bigdict[time_idx][conc_nr][well_nr]['dilateanderode'] = mopped_up_img
                        #bigdict[time_idx][conc_nr][well_nr]['dilate'] = dilated_img
                        ## TODO: when script works, switch time_idx in 'contours' back to 0
                        ## TODO: when script works, switch time_idx in 'thresh' back to 0
                        ## TODO: when script works, switch time_idx in 'dilateanderode' back to 0
                        ## TODO: when script works, switch time_idx in 'dilate' back to 0

                        ## select droplet, and get all necessary information from it
                        #area_droplet, cX_droplet, cY_droplet, contour_droplet, droplet_status = pyphasechip_fun.select_droplet_test(
                        #    bigdict[time_idx][conc_nr][well_nr]['contours'], 238)
                        #bigdict[time_idx][conc_nr][well_nr]['contours droplet'] = contour_droplet
                        #bigdict[0][conc_nr][well_nr]['detected droplet'] = droplet_status
                        ## TODO: when script works, switch time_idx in 'contours droplet' back to 0
                        ## TODO: when script works, switch time_idx in 'contours' back to 0

                        # hough circle test for droplet detection

                        if bigdict[0][conc_nr][well_nr]['detected droplet'] is False:
                            bigdict[0][conc_nr][well_nr]['droplet information'] = \
                                bigdict[0][conc_nr][well_nr]['circles']

                        # blur before Mr. Hough is always a good idea
                        temp = cv2.blur(bigdict[time_idx][conc_nr][well_nr]['masked image'].copy(), (2, 2))

                        for i in range(10):
                            a = 0.97
                            minradius = int(bigdict[0][conc_nr][well_nr]['circles'][0, 0, 2] * 0.4)
                            maxradius = int(bigdict[0][conc_nr][well_nr]['circles'][0, 0, 2] * a)

                            detected_droplet = cv2.HoughCircles(temp,
                                                                cv2.HOUGH_GRADIENT, dp=1.1, minDist=150, param1=50,
                                                                param2=40, minRadius=minradius, maxRadius=maxradius)

                            if time_idx > 3 and bigdict[0][conc_nr][well_nr]['detected droplet'] is True:
                                previous_droplet_radius = bigdict[time_idx-1][conc_nr][well_nr]['droplet information'][0, 0, 2]
                            else:
                                previous_droplet_radius = bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 2]

                            if detected_droplet is not None and previous_droplet_radius >= detected_droplet[0, 0, 2]:
                                detected_droplet = np.uint16(np.around(detected_droplet))
                                droplet_status = True
                                bigdict[0][conc_nr][well_nr]['detected droplet'] = droplet_status
                                bigdict[time_idx][conc_nr][well_nr]['droplet information'] = detected_droplet
                                #print(f"Droplet found in well t{time_idx}_c{conc_nr}_w{well_nr}")
                                break  # end loop when droplet is detected

                            else:
                                a -= 0.01

                            if i == 9:
                                print(f"no droplet could be found in well t{time_idx}_c{conc_nr}_w{well_nr}")
                                # Print error massage if no droplet was found
                                droplet_status = False
                                bigdict[0][conc_nr][well_nr]['detected droplet'] = droplet_status

                        if droplet_status is True:

                            # calculate minimal distance from droplet center to edge
                            bigdict[time_idx][conc_nr][well_nr]['minimal distance'] = \
                                bigdict[time_idx][conc_nr][well_nr]['droplet information'][0, 0, 2]*0.8  # 80% of droplet radius
                            cX_droplet = bigdict[time_idx][conc_nr][well_nr]['droplet information'][0, 0, 0]
                            cY_droplet = bigdict[time_idx][conc_nr][well_nr]['droplet information'][0, 0, 1]
                            area_droplet = 3.14159 * bigdict[0][conc_nr][well_nr]['droplet information'][0, 0, 2]**2
                            #bigdict[time_idx][conc_nr][well_nr]['minimal distance'] = pyphasechip_fun.minDistance(
                            #    contour_droplet,
                            #    cX_droplet,
                            #    cY_droplet)

                            if time_idx > 3:
                                # adjust contrast
                                contrasted_current = cv2.convertScaleAbs(
                                    bigdict[time_idx][conc_nr][well_nr]['masked image'], beta=-40)
                                contrasted_old = cv2.convertScaleAbs(
                                    bigdict[(time_idx - 1)][conc_nr][well_nr]['masked image'], beta=-40)
                                # TODO: remove, this is just a test
                                #houghimg = pyphasechip_fun.hough_test(contour_droplet,
                                #                                      bigdict[time_idx][conc_nr][well_nr]['gray'])
                                #bigdict[time_idx][conc_nr][well_nr]['hough'] = houghimg

                                # subtract current img from old
                                bigdict[time_idx][conc_nr][well_nr]['subtracted'] = cv2.subtract(
                                    contrasted_current, contrasted_old)

                                # calculate pixel values within squircle inside droplet
                                bigdict[time_idx][conc_nr][well_nr]['pixel values'] = pyphasechip_fun.squircle_iteration(
                                    bigdict[time_idx][conc_nr][well_nr]['subtracted'], cX_droplet, cY_droplet,
                                    int(bigdict[time_idx][conc_nr][well_nr]['minimal distance']))

                                # calculate mean of pixel values
                                mean = (np.sum(bigdict[time_idx][conc_nr][well_nr]['pixel values']) /
                                        np.count_nonzero(bigdict[time_idx][conc_nr][well_nr]['pixel values']))

                                # Detector
                                #print(f"{mean}, t{time_idx}_well{conc_nr}_{well_nr}")
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
            well_nr = 0
        well_nr = 0

    return starting_concentrations


def quality_control(bigdict, time_idx, conc_nr, well_nr, name_sol1, name_sol2, unit_sol1, unit_sol2,
                    starting_concentrations, circles):
    # Status
    print("Well found: ", bigdict[0][conc_nr][well_nr]['well found'])
    print("Droplet found: ", bigdict[0][conc_nr][well_nr]['detected droplet'])
    print("LLPS found: ", bigdict[0][conc_nr][well_nr]['LLPS status'])
    print("file name: ", bigdict[0][conc_nr][well_nr]['LLPS name'])
    print(" ")
    print('initial concentrations:')
    print(name_sol1, '', name_sol2)
    print(starting_concentrations)

    # concentrations and areas
    if bigdict[0][conc_nr][well_nr]['well found'] is True \
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
    ax1.set_title('gray')
    ax1.imshow(bigdict[time_idx][conc_nr][well_nr]['gray'].copy())

    if circles is not None:
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle outline
            radius = (i[2])
            circles_img = cv2.circle(bigdict[0][conc_nr][well_nr]['gray'].copy(), center, radius, (0, 0, 0), 1)

            ax2 = fig.add_subplot(5,2,2)
            ax2.set_title('found circles')
            ax2.imshow(circles_img)

    if bigdict[0][conc_nr][well_nr]['well found'] is True:
        ax3 = fig.add_subplot(5,2,3)
        ax3.set_title('threshed')
        ax3.imshow(bigdict[time_idx][conc_nr][well_nr]['thresh'])

        ax4 = fig.add_subplot(5, 2, 4)
        ax4.set_title('after mop-up')
        ax4.imshow(bigdict[time_idx][conc_nr][well_nr]['dilateanderode'])

        ax5 = fig.add_subplot(5, 2, 5)
        ax5.set_title('dilated')
        ax5.imshow(bigdict[time_idx][conc_nr][well_nr]['dilate'])

        allcontours_img = cv2.drawContours(bigdict[time_idx][conc_nr][well_nr]['gray'].copy(),
                                           bigdict[0][conc_nr][well_nr]['contours'], -1, (255, 165, 0), 2)
        ax6 = fig.add_subplot(5, 2, 6)
        ax6.set_title('all detected contours')
        ax6.imshow(allcontours_img)

        if bigdict[0][conc_nr][well_nr]['detected droplet'] is True:

            selectedcontours_img = cv2.drawContours(bigdict[time_idx][conc_nr][well_nr]['gray'].copy(),
                                                    bigdict[time_idx][conc_nr][well_nr]['contours droplet'], -1,
                                                    (255, 165, 0), 2)
            ax7 = fig.add_subplot(5, 2, 7)
            ax7.set_title('selected contours')
            ax7.imshow(selectedcontours_img)

            if time_idx > 0:
                ax8 = fig.add_subplot(5, 2, 8)
                ax8.set_title('subtraction result')
                ax8.imshow(bigdict[time_idx][conc_nr][well_nr]['subtracted'])
                ax9 = fig.add_subplot(5, 2, 9)
                ax9.set_title('squircle result')
                ax9.imshow(bigdict[time_idx][conc_nr][well_nr]['pixel values'])

    print("List of Means")
    print(bigdict[0][conc_nr][well_nr]['mean list'])

    if bigdict[0][conc_nr][well_nr]['well found'] is True \
            and bigdict[0][conc_nr][well_nr]['LLPS status'] is True:

        fig2, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(15, 15))
        # before LLPS detections
        A_10 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 3)][conc_nr][well_nr]['masked image'].copy()
        A_11 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 2)][conc_nr][well_nr]['masked image'].copy()
        A_12 = bigdict[((bigdict[0][conc_nr][well_nr]['ID']) - 1)][conc_nr][well_nr]['masked image'].copy()
        # at LLPS detection
        grimes = bigdict[(bigdict[0][conc_nr][well_nr]['ID'])][conc_nr][well_nr]['masked image'].copy()

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
        writer.writerow(["If you use natively a ',' as a decimal separator, you probably need/"
                         "to change it for correct display of numbers"])
        writer.writerow(["In excel you can do this via File -> Options -> Advanced, here you can change separators"])

