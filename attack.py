import numpy as np
import torch
import torch.nn.functional as F

import cv2

from utils.gaussian_blur import gaussian_blur

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def normalize(t, mean, std):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def input_diversity(img):
    rnd = torch.randint(224, 257, (1,)).item()
    rescaled = F.interpolate(img, (rnd, rnd), mode='nearest')
    h_rem = 256 - rnd
    w_hem = 256 - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_hem + 1, (1,)).item()
    pad_right = w_hem - pad_left
    padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
    padded = F.interpolate(padded, (224, 224), mode='nearest')
    return padded


def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern).cuda()
    return stack_kern, kern_size // 2


def project_noise(x, stack_kern, padding_size):
    # x = tf.pad(x, [[0,0],[kern_size,kern_size],[kern_size,kern_size],[0,0]], "CONSTANT")
    x = F.conv2d(x, stack_kern, padding=(padding_size, padding_size), groups=3)
    return x


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


stack_kern, padding_size = project_kern(3)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap #+ np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization_(original_image, attribution_generator, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).cuda().data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def local_adv(model, att_gen, criterion, img, label, eps, ben_cam, attack_type, iters, mean, std, index, apply_ti=False, amp=10):

    adv = img.detach()

    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    if attack_type in ['fgsm', 'rfgsm']:
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

    if attack_type == 'pifgsm':
        # alpha = step = eps / iterations
        alpha_beta = step * amp
        gamma = alpha_beta
        amplification = 0.0
        # images_min = clip_by_tensor(img - eps, 0.0, 1.0)
        # images_max = clip_by_tensor(img + eps, 0.0, 1.0)

    adv_noise = 0
    c_begin, c_final = 20., 20. * 2
    c_inc = (c_final - c_begin) / iterations
    c_now = 20.
    for j in range(iterations):
        if attack_type == 'dim':
            adv_r = input_diversity(adv)
        else:
            adv_r = adv
        # out_adv = model(normalize(torch.nn.functional.interpolate(adv_r.clone(), (224, 224)), mean=mean, std=std))
        out_adv = model(normalize(adv_r.clone(), mean=mean, std=std))

        adv_cam = generate_visualization_(adv_r.clone().squeeze(0), att_gen)

        c_now += c_inc

        adv_cam_flatten = torch.tensor(adv_cam).view(1, -1)
        adv_cam_flatten = adv_cam_flatten - adv_cam_flatten.min(1, True)[0]
        adv_cam_flatten = adv_cam_flatten / adv_cam_flatten.max(1, True)[0]

        ben_cam_flatten = torch.tensor(ben_cam).view(1, -1)
        ben_cam_flatten = ben_cam_flatten - ben_cam_flatten.min(1, True)[0]
        ben_cam_flatten = ben_cam_flatten / ben_cam_flatten.max(1, True)[0]

        diff = adv_cam_flatten - ben_cam_flatten
        loss_cam = (diff * diff).mean(1)

        loss = 0
        if isinstance(out_adv, list) and index == 'all':
            loss = 0
            for idx in range(len(out_adv)):
                loss += criterion(out_adv[idx], label)
        elif isinstance(out_adv, list) and index == 'last':
            loss = criterion(out_adv[-1], label)
        else:
            loss = criterion(out_adv, label)

        loss += c_now*loss_cam.item()

        loss.backward()
        if apply_ti:
            adv.grad = gaussian_blur(adv.grad, kernel_size=(15, 15), sigma=(3, 3))

        if attack_type == 'mifgsm' or attack_type == 'dim':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        # Optimization step
        if attack_type == 'pifgsm':
            amplification += alpha_beta * adv_noise.sign()
            cut_noise = torch.clamp(abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
            projection = gamma * torch.sign(project_noise(cut_noise, stack_kern, padding_size))
            amplification += projection

            adv.data = adv.data + alpha_beta * adv_noise.sign() + projection
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
            adv.data.clamp_(0.0, 1.0)

        else:
            adv.data = adv.data + step * adv_noise.sign()

            # Projection
            if attack_type == 'pgd':
                adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
                adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)

            adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach(), adv_cam


def local_adv_target(model, criterion, img, target, eps, attack_type, iters, mean, std, index):
    adv = img.detach()

    if attack_type == 'rfgsm':
        alpha = 2 / 255
        adv = adv + alpha * torch.randn(img.shape).cuda().detach().sign()
        eps = eps - alpha

    adv.requires_grad = True

    if attack_type in ['fgsm', 'rfgsm']:
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

    adv_noise = 0
    for j in range(iterations):

        if attack_type == 'dim':
            adv_r = input_diversity(adv)
        else:
            adv_r = adv
        out_adv = model(normalize(adv_r.clone(), mean=mean, std=std))

        loss = 0
        if isinstance(out_adv, list) and index == 'all':
            loss = 0
            for idx in range(len(out_adv)):
                loss += criterion(out_adv[idx], target)
        elif isinstance(out_adv, list) and index == 'last':
            loss = criterion(out_adv[-1], target)
        else:
            loss = criterion(out_adv, target)

        loss.backward()

        if attack_type == 'mifgsm' or attack_type == 'dim':
            adv.grad = adv.grad / torch.mean(torch.abs(adv.grad), dim=(1, 2, 3), keepdim=True)
            adv_noise = adv_noise + adv.grad
        else:
            adv_noise = adv.grad

        # Optimization step
        adv.data = adv.data - step * adv_noise.sign()

        # Projection
        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()
