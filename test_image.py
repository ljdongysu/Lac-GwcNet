import argparse
import torch
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import tqdm, trange
from tqdm.contrib import tzip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time

from dataloader import KITTIloader as kt
from networks.stackhourglass import PSMNet
import cv2
from file import Walk, MkdirSimple

DATA_TYPE = ['kitti', 'indemind', 'depth', 'i18R']


def GetArgs():
    parser = argparse.ArgumentParser(description='LaC')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--gpu_id', type=str, default='2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_path', type=str, default='/media/data/dataset/KITTI/data_scene_flow/training/')
    parser.add_argument('--load_path', type=str, default='state_dicts/kitti2015.pth')
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--lsp_width', type=int, default=3)
    parser.add_argument('--lsp_height', type=int, default=3)
    parser.add_argument('--lsp_dilation', type=list, default=[1, 2, 4, 8])
    parser.add_argument('--lsp_mode', type=str, default='separate')
    parser.add_argument('--lsp_channel', type=int, default=4)
    parser.add_argument('--no_udc', action='store_true', default=False)
    parser.add_argument('--refine', type=str, default='csr')
    parser.add_argument('--output', type=str)
    parser.add_argument('--bf', type=float, default=14.2)

    args = parser.parse_args()

    return args


def GetImages(path, flag='kitti'):
    if os.path.isfile(path):
        # Only testing on a single image
        paths = [path]
        root_len = len(os.path.dirname(paths).rstrip('/'))
    elif os.path.isdir(path):
        # Searching folder for images
        paths = Walk(path, ['jpg', 'png', 'jpeg'])
        root_len = len(path.rstrip('/'))
    else:
        raise Exception("Can not find path: {}".format(path))

    left_files, right_files = [], []
    if 'kitti' == flag:
        left_files = [f for f in paths if 'image_02' in f]
        right_files = [f.replace('/image_02/', '/image_03/') for f in left_files]
    elif 'indemind' == flag:
        left_files = [f for f in paths if 'cam0' in f]
        right_files = [f.replace('/cam0/', '/cam1/') for f in left_files]
    elif 'depth' == flag:
        left_files = [f for f in paths if 'left' in f]
        right_files = [f.replace('left', 'right') for f in left_files]
    elif 'i18R' == flag:
        left_files = [f for f in paths if '.L' in f]
        right_files = [f.replace('L/', 'R/').replace('L.', 'R.') for f in left_files]
    else:
        raise Exception("Do not support mode: {}".format(flag))

    return left_files, right_files, root_len


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)

    return depth_img_rgb.astype(np.uint8)

def WriteDepth(depth, limg, path, name, bf):
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    output_gray = os.path.join(path, "gray", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_gray)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)

    predict_np = depth.squeeze().cpu().numpy()

    disp = depth

    predict_np = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np, cv2.COLORMAP_HOT)
    limg_cv = cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img = bf / predict_np * 100  # to cm
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)

def readImg(left_file, right_file):
    # limg = Image.open(left_file).convert('RGB')
    # rimg = Image.open(right_file).convert('RGB')

    left_img = cv2.imread(left_file)
    right_img = cv2.imread(right_file)

    in_h, in_w = left_img.shape[:2]

    if in_h % 8 != 0:
        pad_h = in_h % 8
        left_img = np.pad(left_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')

    if in_w % 8 != 0:
        pad_w = in_w % 8
        left_img = np.pad(left_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')

    in_h, in_w = left_img.shape[:2]

    if in_h % 8 != 0:
        pad_h = in_h % 8
        left_img = np.pad(left_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((pad_h // 2, pad_h // 2), (0, 0), (0, 0)), mode='reflect')

    if in_w % 8 != 0:
        pad_w = in_w % 8
        left_img = np.pad(left_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')
        right_img = np.pad(right_img, ((0, 0), (pad_w // 2, pad_w // 2), (0, 0)), mode='reflect')

    in_h, in_w = left_img.shape[:2]
    # Resize image in case the GPU memory overflows
    eval_h, eval_w = (in_h, in_w)
    assert eval_h % 8 == 0, "input height should be divisible by 8"
    assert eval_w % 8 == 0, "input width should be divisible by 8"

    limg = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
    rimg = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

    limg = cv2.cvtColor(limg, cv2.COLOR_BGRA2RGB)
    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGRA2RGB)


    return limg, rimg

def main():
    args = GetArgs()

    output_directory = args.output

    if not args.no_cuda:
        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    left_files, right_files, root_len = [], [], []
    for k in DATA_TYPE:
        left_files, right_files, root_len = GetImages(args.data_path, k)

        if len(left_files) != 0:
            break

    affinity_settings = {}
    affinity_settings['win_w'] = args.lsp_width
    affinity_settings['win_h'] = args.lsp_width
    affinity_settings['dilation'] = args.lsp_dilation
    udc = not args.no_udc

    model = PSMNet(maxdisp=args.max_disp, struct_fea_c=args.lsp_channel, fuse_mode=args.lsp_mode,
                   affinity_settings=affinity_settings, udc=udc, refine=args.refine)
    model = nn.DataParallel(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if use_cuda:
        model.cuda()
    model.eval()

    ckpt = torch.load(args.load_path)
    model.load_state_dict(ckpt)

    mae = 0
    op = 0
    for left_image_file, right_image_file in tzip(left_files, right_files):
        if not os.path.exists(left_image_file) or not os.path.exists(right_image_file):
            continue

        output_name = left_image_file[root_len + 1:]
        limg, rimg = readImg(left_image_file, right_image_file)

        limg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
        rimg_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
        limg_tensor = limg_tensor.unsqueeze(0).cuda()
        rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            start = time()
            pred_disp = model(limg_tensor, rimg_tensor)
            # print("use: ", (time() - start))

        predict_np = pred_disp.squeeze().cpu().numpy()

        WriteDepth(pred_disp, limg, args.output, output_name, args.bf)


if __name__ == '__main__':
    main()
