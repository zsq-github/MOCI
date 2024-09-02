'''
import os
import shutil

def merge_folders(source_folders, destination_folder):
    for source_folder in source_folders:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                source_file = os.path.join(root, file)
                relative_path = os.path.relpath(source_file, source_folder)
                destination_file = os.path.join(destination_folder, relative_path)
                os.makedirs(os.path.dirname(destination_file), exist_ok=True)
                shutil.move(source_file, destination_file)

source_folders = ['/media/lenovo/12TB1/zsq/test/one', '/media/lenovo/12TB1/zsq/test/two', '/media/lenovo/12TB1/zsq/test/three']
destination_folder = '/media/lenovo/12TB1/zsq/test/merge'

merge_folders(source_folders, destination_folder)
'''

import os
import shutil

def merge_folders(source_folders, target_folder):
    #  Creating a Destination Folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for folder in source_folders:
        #  Check that the source folder exists
        if not os.path.exists(folder):
            print(f"Source Folder '{folder}'  Don't exist")
            continue

        # Get all files in the source folder
        files = os.listdir(folder)

        # Iterate through each file and move to destination folder
        for file in files:
            source_file = os.path.join(folder, file)
            target_file = os.path.join(target_folder, file)
            # Copy files to the destination folder
            shutil.copyfile(source_file, target_file)
            #print(f"copy '{source_file}' to '{target_file}'。")

source_folders = ['/media/lenovo/12TB1/zsq/CT-Scan/images_001', '/media/lenovo/12TB1/zsq/CT-Scan/images_002', '/media/lenovo/12TB1/zsq/CT-Scan/images_003', '/media/lenovo/12TB1/zsq/CT-Scan/images_004', '/media/lenovo/12TB1/zsq/CT-Scan/images_005', '/media/lenovo/12TB1/zsq/CT-Scan/images_006', '/media/lenovo/12TB1/zsq/CT-Scan/images_007', '/media/lenovo/12TB1/zsq/CT-Scan/images_008', '/media/lenovo/12TB1/zsq/CT-Scan/images_009', '/media/lenovo/12TB1/zsq/CT-Scan/images_010', '/media/lenovo/12TB1/zsq/CT-Scan/images_011', '/media/lenovo/12TB1/zsq/CT-Scan/images_012']
target_folder = '//media/lenovo/12TB1/zsq/chest_test/images'

merge_folders(source_folders, target_folder)



