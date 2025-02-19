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
        right_files = [f.replace('/left/', '/right/') for f in left_files]
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

def  WriteDepth(depth, limg,  path, name, bf):
    name = os.path.splitext(name)[0] + ".png"
    output_concat_color =  os.path.join(path, "concat_color", name)
    output_concat_gray =  os.path.join(path, "concat_gray", name)
    output_gray =  os.path.join(path, "gray", name)
    output_color =  os.path.join(path, "color", name)
    output_concat_depth =  os.path.join(path, "concat_depth", name)
    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_gray)
    MkdirSimple(output_color)

    # predict_np = depth.squeeze().cpu().numpy()
    predict_np = np.squeeze(np.array(depth))

    disp = depth

    predict_np = predict_np.astype(np.uint8)
    color_img = cv2.applyColorMap(predict_np, cv2.COLORMAP_HOT)
    limg_cv = cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img = bf / predict_np * 100 # to cm
    depth_img_rgb = GetDepthImg(depth_img)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)
    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_concat_depth, concat_img_depth)

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

    #ONNX
    # print(model.)
    # print(model.module)
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_input = (torch.randn(1, 3, 400, 640, device=device), torch.randn(1, 3, 400, 640, device=device))
    # print(dummy_input.is_cuda)
    # dummy_input_right =
    input_names = ['L','R']
    # output_names = ['cls_logits', 'bbox_preds', 'anchors']
    output_names = ['output']
    module_list=model.module
    torch.onnx.export(
        model.module,
        dummy_input,
        "kitti-opset11.onnx",
        verbose=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names)

if __name__ == '__main__':
    main()