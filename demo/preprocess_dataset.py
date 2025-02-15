import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
def process_tls_dataset():
    root_dataset = "/data2/dataset/chenhh_dataset/TLS_dataset"
    raw_dataset_dir = os.path.join(root_dataset, "patches")
    patch_folder = os.path.join(raw_dataset_dir, 'patches1')
    mask_folder = os.path.join(raw_dataset_dir, 'mask1')
    patch_names = os.listdir(patch_folder)
    
    np.random.shuffle(patch_names)
    test_num = int(len(patch_names) * 0.2)
    test_patch_names = patch_names[:test_num]
    train_patch_names = patch_names[test_num:]

    for pahse, cur_patch_names in zip(["test", "train"], [test_patch_names, train_patch_names]):
        image_save_folder = os.path.join(root_dataset, pahse, "images")
        mask_save_folder = os.path.join(root_dataset, pahse, "masks")
        os.makedirs(image_save_folder, exist_ok=True)
        os.makedirs(mask_save_folder, exist_ok=True)
        for patch_name in tqdm(cur_patch_names):
            src_image_fn = os.path.join(patch_folder, patch_name)
            tar_image_fn = os.path.join(image_save_folder, patch_name)
            if os.path.exists(tar_image_fn):
                os.remove(tar_image_fn)
            os.symlink(src_image_fn, tar_image_fn)
            src_mask_fn = os.path.join(mask_folder, patch_name)
            tar_mask_fn = os.path.join(mask_save_folder, patch_name)
            mask_image = cv2.imread(src_mask_fn, cv2.IMREAD_GRAYSCALE)
            mask_image[mask_image > 0] = 1
            cv2.imwrite(tar_mask_fn, mask_image)
            
    
if __name__ == "__main__":
    process_tls_dataset()