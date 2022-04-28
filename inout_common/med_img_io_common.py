import os
from os.path import getmtime
from os import walk, listdir
from os.path import join

import numpy as np
import SimpleITK as sitk


def get_dicom_files_in_folder(input_folder):
    """Gets the list of all dicom files inside a folder"""
    list_of_files = []
    # Iterate all the files
    for dirName, subdirList, fileList in walk(input_folder):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                list_of_files.append(join(dirName, filename))

    return list_of_files


def write_itk_imgs(output_folder, pretxt, imgs, img_names):
    """
    :param output_folder: str path to save the image
    :param pretxt: str prefix to store for each img name
    :param imgs: itk array of images
    :param img_names: str array of image names
    :return:
    """
    create_folder(output_folder)
    print("\t\t\tSaving images ({})...".format(pretxt))
    for img_idx in range(len(imgs)):
        file_name = F'{pretxt}_{img_names[img_idx]}.nrrd'
        sitk.WriteImage(imgs[img_idx], join(output_folder, file_name))


def get_earliest_mri_from_folder(mri_series_name, input_foler):
    '''
    Reads from a list of mri_series folder names and selects the earliest folder that matches the name
    :param mri_series_name: Names of mri_series
    :param input_foler: Path where the mri_series is stored
    :return:
    '''

    # Verify all the ROIs interested TODO should be replaced to a regex
    mri_series_files = [f for f in listdir(input_foler) if f.lower().find(mri_series_name.lower()) != -1 ]

    if len(mri_series_files) == 0:
        raise Exception("ERROR: didn't found mri_series ", mri_series_name, " inside ", input_foler)

    elif len(mri_series_files) > 0: # In this case we need to check the date and select the earliest
        # This is a horrible patch to fix the reading for BVAL and ADC (forcing it to give priority to the aligend)
        if (mri_series_name.find('ADC') != -1 or mri_series_name.find('BVAL') != -1):
            # Check first for aligned
            for cfile in mri_series_files:
                if (cfile.find(mri_series_name) != -1 and cfile.find('Aligned') != -1):
                    return cfile

        # If is not ADC nor BVAL or we didn't find the 'aligned folder' then get the latest one
        times = []
        if len(mri_series_files) > 1: # Then search for the latest one
            for cfile in mri_series_files:
                dicomFiles =[f for f in listdir(join(input_foler,cfile)) if f.find('.dcm') != -1 ]
                # Here we are using the first mri_series
                ds = pydicom.read_file(join(input_foler,cfile,dicomFiles[0])) # Reads structure
                try:
                    f_date = np.datetime64( ds.AcquisitionDate[0:4]+'-'+ \
                                   ds.AcquisitionDate[4:6]+'-'+ \
                                   ds.AcquisitionDate[6:8]+'T'+ \
                                   ds.AcquisitionTime[0:2]+':'+ \
                                   ds.AcquisitionTime[2:4]+':'+ \
                                   ds.AcquisitionTime[4:6] )
                    times.append(f_date)
                except Exception as e:
                    print("!!!!!!!!!!!!!!! Failed to read file {} Error: {e} !!!!!!!!!!!!!!!!!!!".format(dicomFiles[0], e=e))

            closest_date = max(times)
            idx_file = np.where(closest_date == times)
            final_file = mri_series_files[idx_file[0][0]]
        else:# Only one file in this case
            final_file = mri_series_files[0]

    return final_file


def get_dicom_header(folder):
    '''Gets the dicom header from a folder'''
    list_of_files = listdir(folder)
    RefDs = pydicom.read_file(join(folder,list_of_files[0]))
    return RefDs, len(list_of_files)-2


def select_cases_from_folder(input_folder, folders_to_read='all'):
    if isinstance(folders_to_read, str):
        if folders_to_read == 'all':
            print(F'Reading all folders from {input_folder}')
            folder_names = listdir(input_folder)
    else:
        if not(isinstance(folders_to_read[0], str)): # Verify each element on the list is a string or a number
            folder_names = [join(input_folder, 'Case-{num:04d}'.format(num=f)) for f in folders_to_read]
        else:
            folder_names = folders_to_read
    folder_names.sort()
    return np.array(folder_names)
