import os
from PIL import Image
import numpy as np
from visualize import resize


def convert_to_npz(img_path, npz_dir):
    img = Image.open(img_path)
    img_np = np.array(img)
    img_resized = resize(img_np)
    base_name = os.path.basename(img_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    npz_path = os.path.join(npz_dir, base_name_no_ext + '.npz')
    np.savez_compressed(npz_path, img_x=img_resized)
    return npz_path

if __name__ == '__main__':
    root_dir = './example/'
    save_dir = './experiments_input/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for dir_name in os.listdir(root_dir):
        sub_dir = os.path.join(root_dir, dir_name)
        if os.path.isdir(sub_dir):
            for file_name in os.listdir(sub_dir):
                if file_name.endswith('.JPEG'):
                    img_path = os.path.join(sub_dir, file_name)
                    npz_path = convert_to_npz(img_path, save_dir)
                    print(f"Image saved at: {npz_path}")