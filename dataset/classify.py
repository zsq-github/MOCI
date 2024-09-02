import os
import shutil
import pandas as pd

csv_file = '/media/lenovo/12TB1/zsq/chest_test/chest_label.csv'
df = pd.read_csv(csv_file)

source_dir = '/media/lenovo/12TB1/zsq/chest_test/images'
target_dir = '/media/lenovo/12TB1/zsq/chest_test/merge_test'

os.makedirs(target_dir, exist_ok=True)

for index, row in df.iterrows():
    image_name = row['Image Index']
    label = row['Finding Label']
    target_label_dir = os.path.join(target_dir, label)
    os.makedirs(target_label_dir, exist_ok=True)
    source_path = os.path.join(source_dir, image_name)
    target_path = os.path.join(target_label_dir, image_name)
    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        #print(f'Copied {image_name} to {target_label_dir}')
    else:
        print(f'File {image_name} not found in source directory')

print('Done!')
