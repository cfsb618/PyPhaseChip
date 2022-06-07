import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from datetime import datetime
import PIL.ExifTags

import dateutil.parser
import os
import re

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, MaxNLocator)

from pyphasechip import pyphasechip_logic as pypc
from typing import Tuple

if __name__ == "__main__":
    # reagent 1 name
    name_sol1 = "BSA"
    # reagent 1 starting contentration
    initial_conc_sol1 = 266
    # unit
    unit_sol1 = "µM"

    # reagent 2 name
    name_sol2 = "PEG"
    # reagent 2 starting concentration
    initial_conc_sol2 = 10
    # unit
    unit_sol2 = "%"

    # mixing ratio of the concentrations 0
    # [1:X]
    initial_ratio = 7

    # Number of different concentrations used
    number_of_diff_conc = 3

    # Number of wells per horizontal(!) line per concentrations
    # full chip equals 10
    n_wells = 1

    # How many hours did the experiment last?
    hours_of_experiment = 17

    # How many pictures where taken per hour per well?
    images_per_hour = 1

    ###### delete hoe and iph, use this instead
    # total amount of images
    total_n_images = 102
    n_timepoints = int(total_n_images / (number_of_diff_conc * n_wells * 2))

    # percental difference of img mean to all previous imgages.
    # value is similar to a sensivity-value
    percental_threshold = 55

    # measured diameter of chamber to calculate radii for circle detection -> min and max [px]
    # around +-30 of the real value, best measured with something like imageJ/Fiji
    diameter = 238
    minRadiusChamber = int((diameter / 2) * 0.7)
    maxRadiusChamber = int((diameter / 2) * 1.3)

    # height of chamber [µm]
    chamber_height = 60

    # folder where images are stored
    # best is to use only paths without any whitespace
    # image_folder = "C:Users/DigitalStation/Documents/Universität Stuttgart/Institut für industrielle Bioverfahrenstechnik/1_Masterthesis/Experiments/20210804 First OFM Exp/Images"
    # image_folder = r"C:\Users\DigitalStation\Documents\Python_Scripts\DropletDetection\TestImages"

    image_folder = r"C:\Users\DigitalStation\Documents\Python_Scripts\DropletDetection\testimages2+"
    # image_folder = r"C:\Users\DigitalStation\Documents\Python_Scripts\DropletDetection\TestImages3"

    # datatype of the images
    extension = ".jpg"

    # TODO: use a .yml file for storing these variables

    # Create all the necessary dicts and lists
    image_list, image_names, data_well, well, concentration, time_resolution = pypc.create()

    # Load images & convert to grayscale
    pypc.images_to_list(image_list, image_names, image_folder, extension)
    pypc.images_to_dict(hours_of_experiment, images_per_hour, number_of_diff_conc, n_wells, image_list, image_names,
                        time_resolution, concentration, well, data_well)
    starting_conc = pypc.starting_concentration(initial_conc_sol1, initial_conc_sol2, initial_ratio)

    well_nr = 0
    for conc_nr in range(number_of_diff_conc):
        for n_rows_per_conc in range(2):
            for n_wells_per_row in range(n_wells):

                # Initialise variables, arrays and lists
                well_data = np.zeros(shape=(2, 3))
                centerpoints = np.zeros(shape=(2, 2))
                elon_mask = np.zeros(shape=time_resolution[0][0][0]['gray'].shape, dtype="uint8")
                threshed_img = np.zeros(shape=time_resolution[0][0][0]['gray'].shape, dtype="uint8")
                areas = np.zeros(shape=(1, 2))
                droplet_arr = np.zeros(shape=(2, 4))
                r_old_hv = np.zeros(2)
                mean_list = []
                llps_status = False
                r_0 = 0
                n_0 = 0
                r_droplet_old = 0
                avg_sum_prev = 0

                for time_idx in range(n_timepoints):  # n_timepoints
                    if llps_status is False:
                        print("---", "C:", conc_nr, "W:", well_nr, " T:", time_idx, "---")

                        image = time_resolution[time_idx][conc_nr][well_nr]['gray'].copy()

                        well_data, elon_mask, masked_img, droplet_found, norm_pp_len_h, norm_pp_len_v, img, f, N, E, S, W, x, y, droplet_arr, radius_old_hv, hor, vert, r_0, avg_sum_prev, droplet_coords = pypc.droplet_detection(
                            diameter, image, well_data, elon_mask, centerpoints, llps_status, droplet_arr, r_0,
                            time_idx, r_old_hv, r_droplet_old, avg_sum_prev)

                        time_resolution[time_idx][conc_nr][well_nr]['img'] = img
                        time_resolution[time_idx][conc_nr][well_nr][
                            'normg_pp_len_h'] = norm_pp_len_h  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr][
                            'normg_pp_len_v'] = norm_pp_len_v  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr]['masked img'] = masked_img
                        time_resolution[time_idx][conc_nr][well_nr]['manipulated img'] = img  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr]['droplet data'] = droplet_arr.copy()
                        time_resolution[time_idx][conc_nr][well_nr]['x'] = x  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr]['y'] = y  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr]['r'] = well_data[0, 2]  # needed for display
                        time_resolution[time_idx][conc_nr][well_nr]['N'] = N
                        time_resolution[time_idx][conc_nr][well_nr]['E'] = E
                        time_resolution[time_idx][conc_nr][well_nr]['S'] = S
                        time_resolution[time_idx][conc_nr][well_nr]['W'] = W

                        time_resolution[time_idx][conc_nr][well_nr]['well data'] = well_data
                        time_resolution[time_idx][conc_nr][well_nr]['mask'] = elon_mask
                        time_resolution[time_idx][conc_nr][well_nr]['hor'] = hor
                        time_resolution[time_idx][conc_nr][well_nr]['vert'] = vert

                        time_resolution[time_idx][conc_nr][well_nr]['coords'] = droplet_coords

                        if time_idx == 0:
                            areas[0, 0] = droplet_arr[0, 3]

                        llps_status, areas, mean_list, droplet_arr, squi, cro_squi, n_0 = pypc.detect_LLPS(
                            percental_threshold, droplet_arr, llps_status, img, time_idx, areas, mean_list,
                            droplet_found, n_0)
                        time_resolution[0][conc_nr][well_nr]['areas'] = areas
                        time_resolution[0][conc_nr][well_nr]['mean list'] = mean_list
                        time_resolution[time_idx][conc_nr][well_nr]['squ'] = squi
                        time_resolution[time_idx][conc_nr][well_nr]['cro squ'] = cro_squi

                        time_resolution[time_idx][conc_nr][well_nr]['droplet array'] = droplet_arr

                        print("LLPS status: ", llps_status)
                        if llps_status is True:
                            # save img time where llps was found
                            time_resolution[0][conc_nr][well_nr]['time idx'] = time_idx
                            # save name of image where LLPS was detected
                            time_resolution[0][conc_nr][well_nr]['LLPS name'] = \
                            time_resolution[time_idx][conc_nr][well_nr]['name']
                            # calculate the critical concentration
                            print("c_crit calculation input: ", starting_conc, "areas:", areas, conc_nr)
                            llps_conc = pypc.ccrit_calculation(starting_conc, areas, conc_nr)
                            time_resolution[0][conc_nr][well_nr]['LLPS conc'] = llps_conc

                well_nr += 1
        well_nr = 0

    # Two possibilities:
    # use script if you did a pipetting series
    starting_concentrations = pypc.starting_concentration(initial_conc_sol1, initial_conc_sol2, initial_ratio)
    # or
    # write list

    # saves the image names where LLPS was detected and the calculated concentrations to a csv file
    # .csv gets safed in the image folder
    # pypc.save_results_to_csv(time_resolution, image_folder, number_of_diff_conc, n_wells, hours_of_experiment,
    #                       images_per_hour, name_sol1, name_sol2, unit_sol1, unit_sol2)