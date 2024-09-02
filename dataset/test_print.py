import numpy as np
# Loading .npy files
custom_dict = np.load('/media/lenovo/12TB1/zsq/CT_scan/CTScancustom_dict.npy', allow_pickle=True).item()

print("Dictionary content:")
for key, value in custom_dict.items():
    print(f"{key}: {value}")
