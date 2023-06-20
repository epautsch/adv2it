import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def resize(img, new_size=(224, 224)):
    if len(img.shape) == 2:  # Grayscale image, no need for transpose
        img = np.uint8(255 * img)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return np.float32(img / 255.)
    else:  # Color image
        if img.shape[-1] == 3:
            img = img.transpose([2, 0, 1])
        img = np.uint8(255 * img)
        # Transpose the image back to (height, width, channels) for resizing
        img = img.transpose([1, 2, 0])
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        # Transpose the image back to (channels, height, width)
        img = img.transpose([2, 0, 1])
        return np.float32(img / 255.)


def plot(img, heatmap):
    #     m1 = np.uint8(255 * cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR))
    m1 = np.uint8(heatmap)
    m1 = cv2.applyColorMap(m1, cv2.COLORMAP_JET)
    m1 = np.float32(m1 / 255.).transpose([2, 0, 1])[::-1]
    m1 = (img + m1)
    m1 = m1 / m1.max()
    return np.float32(m1).transpose((1, 2, 0))


def visualize(img_path, model_folder=None):
    if not img_path.endswith('.npz'):
        print('Invalid data format for visualization')
        return

    if not os.path.isfile(img_path):
        print(f'Image not found in the directory: {img_path}')
        return

    img = np.load(img_path)

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))
    axarr[0, 0].imshow(img['img_x'].transpose((1, 2, 0)))
    axarr[0, 0].set_title("Benign Image", fontdict=None, loc='center', color="k")
    axarr[0, 1].imshow(plot(img['img_x'], img['ben_cam'].squeeze()))
    axarr[0, 1].set_title("Benign Interpretation", fontdict=None, loc='center', color="k")
    axarr[1, 0].imshow(img['adv_x'].transpose((1, 2, 0)))
    axarr[1, 0].set_title("Adversarial Image", fontdict=None, loc='center', color="k")
    axarr[1, 1].imshow(plot(img['adv_x'], img['adv_cam'].squeeze()))
    axarr[1, 1].set_title("Adversarial Interpretation", fontdict=None, loc='center', color="k")

    name = img_path.split('/')[-1].split('.')[0]

    if model_folder is not None:
        save_name = f'{model_folder}/{name}.png'
        plt.savefig(save_name)
        # print(f'Viz saved at: {save_name}')
        plt.close()
    else:
        plt.savefig(f'visualization/{name}.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Transformers')
    parser.add_argument('--img_path', default='', help='Image path (in npz format)')
    args = parser.parse_args()
    visualize(args.img_path)


if __name__ == '__main__':
    main()