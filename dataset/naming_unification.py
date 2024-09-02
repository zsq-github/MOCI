'''
import os
import shutil

def rename_images(source_dir, prefix="image_"):

    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    for index, filename in enumerate(image_files, start=1):
        new_filename = f"{prefix}{index:04d}.png"

        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(source_dir, new_filename)

        shutil.move(source_file, target_file)

rename_images("/media/lenovo/12TB1/zsq/test/merge/Cardiomegaly")
'''

import os
def rename_images(parent_dir):
    for dir_name in os.listdir(parent_dir):
        dir_path = os.path.join(parent_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue

        image_count = 1
        for image_name in sorted(os.listdir(dir_path)):
            if not image_name.endswith('.jpeg'):
                continue

            image_path = os.path.join(dir_path, image_name)
            new_image_name = f"image_{image_count:04d}.jpeg"
            new_image_path = os.path.join(dir_path, new_image_name)
            os.rename(image_path, new_image_path)
            image_count += 1


#parent_dir = "/media/lenovo/12TB1/zsq/chest_test/merge_test"
parent_dir = "/media/lenovo/12TB1/zsq/CT_scan/data_all"
#parent_dir = "/media/lenovo/12TB1/zsq/chestX_rayTwo/images"
rename_images(parent_dir)
