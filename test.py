import argparse
import datetime
import json
import logging
import os
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

import numpy as np
import torch
from timm.models import create_model
from torch.utils.data import DataLoader
from torchvision import transforms, models, utils as vutils
from tqdm import tqdm
from ViT_explanation_generator import LRP

from ViT_LRP import vit_base_patch16_224 as vit_b_LRP
from ViT_LRP import vit_large_patch16_224 as vit_l_LRP
# from ViT_LRP import deit_small_patch16_224 as vit_LRP
from PIL import Image

import vit_models
from attack_warm_start import normalize, local_adv, local_adv_target
from visualize import resize, visualize

from torchvision.datasets import ImageFolder

# torch.cuda.set_device(2)
# torch.cuda.current_device()

targeted_class_dict = {
    24: "Great Grey Owl",
    99: "Goose",
    245: "French Bulldog",
    344: "Hippopotamus",
    471: "Cannon",
    555: "Fire Engine",
    661: "Model T",
    701: "Parachute",
    802: "Snowmobile",
    919: "Street Sign ",
}


def parse_args(src_model=None):
    parser = argparse.ArgumentParser(description='Transformers')
    # parser.add_argument('--img_path', default="", help='Image path (in npz format)')
    parser.add_argument('--src_model', type=str, default='deit_small_patch16_224', help='Source Model Name')
    parser.add_argument('--tar_model', type=str, default='T2t_vit_24', help='Target Model Name')
    parser.add_argument('--src_pretrained', type=str, default=None, help='pretrained path for source model')
    parser.add_argument('--scale_size', type=int, default=256, help='')
    parser.add_argument('--img_size', type=int, default=224, help='')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--eps', type=int, default=8, help='Perturbation Budget')
    parser.add_argument('--iter', type=int, default=10, help='Attack iterations')
    parser.add_argument('--index', type=str, default='last', help='last or all')
    parser.add_argument('--attack_type', type=str, default='fgsm', help='fgsm, mifgsm, dim, pgd')
    parser.add_argument('--tar_ensemble', action="store_true", default=False)
    parser.add_argument('--apply_ti', action="store_true", default=False)
    parser.add_argument('--save_im', action="store_true", default=True)

    args = parser.parse_args(['--src_model', src_model,
                              '--tar_model', 'lrp_b',
                              '--attack_type', 'pgd',
                              '--eps', '8',
                              '--index', 'all',
                              '--batch_size', '1'])

    # return parser.parse_args()
    return args


def get_model(model_name):
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(models.__dict__[name]))
    other_model_names = vars(vit_models)

    # get the source model
    if model_name in model_names:
        model = models.__dict__[model_name](pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'deit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'hierarchical' in model_name or "ensemble" in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'vit' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'T2t' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif 'tnt' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'lrp_b' in model_name:
        model = vit_b_LRP(pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'lrp_l' in model_name:
        model = vit_l_LRP(pretrained=True)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif 'swin' in model_name:
        model = create_model(model_name, pretrained=True)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        raise NotImplementedError(f"Please provide correct model names: {model_names}")

    return model, mean, std


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap  # + np.float32(img)
    cam = cam / np.max(cam)
    return cam


def generate_visualization_(original_image, attribution_generator, device, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).to(device), device,
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).to(device).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def start(args=None, npz_tensor=None, file_paths=None):
    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load source and target models
    src_model, src_mean, src_std = get_model(args.src_model)
    if args.src_pretrained is not None:
        if args.src_pretrained.startswith("https://"):
            src_checkpoint = torch.hub.load_state_dict_from_url(args.src_pretrained, map_location='cpu')
        else:
            src_checkpoint = torch.load(args.src_pretrained, map_location='cpu')
        src_model.load_state_dict(src_checkpoint['model'])
    src_model = src_model.to(device)
    src_model.eval()

    # Target is LRP interpreter
    tar_model, tar_mean, tar_std = get_model(args.tar_model)
    tar_model = tar_model.to(device)
    tar_model.eval()

    attribution_generator = LRP(tar_model)

    # setup eval parameters
    eps = args.eps / 255
    criterion = torch.nn.CrossEntropyLoss()

    clean_acc = 0.0
    adv_acc = 0.0
    fool_rate = 0.0
    # files = [args.img_path]

    with tqdm(enumerate(zip(npz_tensor, file_paths)), total=len(npz_tensor)) as p_bar:
        for j, (img_tensor, f_name) in p_bar:
            img = img_tensor.unsqueeze(0).to(device)
            name = os.path.basename(f_name).split('.')[0]
            ben_cam = generate_visualization_(img.clone().squeeze(0), attribution_generator, device)

            # with torch.no_grad():
            if args.tar_ensemble:
                clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std), get_average=True)
            else:
                clean_out = tar_model(normalize(img.clone(), mean=tar_mean, std=tar_std))
            if isinstance(clean_out, list):
                clean_out = clean_out[-1].detach()
            label = clean_out.argmax(dim=-1)
            clean_acc += torch.sum(clean_out.argmax(dim=-1) == label).item()

            adv, adv_cam = local_adv(device, src_model, attribution_generator, criterion, img, label, eps, ben_cam,
                                     attack_type=args.attack_type, iters=args.iter,
                                     std=src_std, mean=src_mean, index=args.index, apply_ti=args.apply_ti,
                                     src_name=args.src_model)

            if args.tar_ensemble:
                adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std), get_average=True)
            else:
                adv_out = tar_model(normalize(adv.clone(), mean=tar_mean, std=tar_std))
            if isinstance(adv_out, list):
                adv_out = adv_out[-1].detach()
            adv_acc += torch.sum(adv_out.argmax(dim=-1) == label).item()
            fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

            if (adv_out.argmax(dim=-1) != label) and args.save_im:
                save_dobj = {'adv_x': vutils.make_grid(adv, normalize=False, scale_each=True).cpu(),
                             'img_x': vutils.make_grid(img, normalize=False, scale_each=True).cpu(),
                             'adv_cam': adv_cam,
                             'ben_cam': ben_cam,
                             'adv_label': adv_out.detach().cpu().numpy()
                             }
                os.makedirs(f'{args.src_model}/output_success/', exist_ok=True)
                np.savez(f'{args.src_model}/output_success/output_{name}.npz', **save_dobj)

                p_bar.set_postfix({'status': 'Attack completed successfully!',
                                   'success rate': f'{(fool_rate / (j+1)) * 100:.2f}%'})

            else:
                save_dobj = {'adv_x': vutils.make_grid(adv, normalize=False, scale_each=True).cpu(),
                             'img_x': vutils.make_grid(img, normalize=False, scale_each=True).cpu(),
                             'adv_cam': adv_cam,
                             'ben_cam': ben_cam,
                             'adv_label': adv_out.detach().cpu().numpy()
                             }
                os.makedirs(f'{args.src_model}/output_fail/', exist_ok=True)
                np.savez(f'{args.src_model}/output_fail/output_{name}.npz', **save_dobj)

                p_bar.set_postfix({'status': 'Attack failed.',
                                   'success rate': f'{(fool_rate / (j+1)) * 100:.2f}%'})


def convert_to_npz(img_path, npz_dir):
    img = Image.open(img_path)
    img_np = np.array(img)
    img_resized = resize(img_np)
    base_name = os.path.basename(img_path)
    base_name_no_ext = os.path.splitext(base_name)[0]
    npz_path = os.path.join(npz_dir, base_name_no_ext + '.npz')
    np.savez_compressed(npz_path, img_x=img_resized)

    return npz_path


def load_npz_to_tensor(npz_file_path):
    npz_file = np.load(npz_file_path)
    img_array = npz_file['img_x']
    if img_array.ndim == 2:
        img_array = np.repeat(img_array[None, :, :], 3, axis=0)
    return torch.from_numpy(img_array), npz_file_path


def batch_viz(model_dir):
    directories = ['output_fail', 'output_success']

    for dir_name in directories:
        viz_dir = os.path.join(model_dir, f'{dir_name}_viz')
        os.makedirs(viz_dir, exist_ok=True)

    for dir_name in directories:
        npz_files = glob(os.path.join(model_dir, dir_name, '*.npz'))

        for npz_file in tqdm(npz_files, desc=f'Processing {dir_name}'):
            visualize(npz_file, model_folder=os.path.join(model_dir, f'{dir_name}_viz'))


if __name__ == '__main__':
    npz_path = './experiments_input/'
    npz_files = [f for f in os.listdir(npz_path) if f.endswith('.npz')]
    tensors_and_paths = [load_npz_to_tensor(os.path.join(npz_path, f)) for f in npz_files]
    tensors, file_paths = zip(*tensors_and_paths)
    stacked_tensors = torch.stack(tensors)

    # src_model = 'deit_base_patch16_224'
    # src_model = 'deit3_base_patch16_224.fb_in1k'
    # src_model = 'deit3_large_patch16_224.fb_in1k'
    # src_model = 'deit3_huge_patch14_224.fb_in1k'
    # src_model = 'vit_huge_patch14_224.orig_in21k'
    # src_model = 'vit_giant_patch14_clip_224.laion2b'
    # src_model = 'vit_gigantic_patch14_clip_224.laion2b'
    # src_model = 'swinv2_cr_small_ns_224.sw_in1k'
    src_model = 'swinv2_cr_tiny_ns_224.sw_in1k'
    args = parse_args(src_model)
    # start(args, stacked_tensors, file_paths)
    batch_viz(src_model)






