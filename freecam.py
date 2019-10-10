import argparse
import io
import os
from typing import Callable, List, Tuple

import PIL.Image
import PIL.ImageDraw2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.patches import Patch

from model import produce_model

model = None
metadata = None


def setup_model(initial_weight: str) -> None:
    global model
    if model is None:
        model = produce_model()
        model.load_state_dict(torch.load(initial_weight))
        model.eval()
        model = model.features
    return


def classify_image(img: PIL.Image.Image,
                   topk: int = 1,
                   feature_threshold: float = 0.33) -> Tuple[List[Tuple[int, float]], List[np.ndarray], np.ndarray]:
    global model
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x: torch.Tensor = transform_tensor(img)
    x = x.unsqueeze(0)
    with torch.no_grad():
        features = model(x)[0]
        output = F.adaptive_avg_pool2d(features, (1, 1)).view(-1)
        act = F.softmax(output, dim=0)
        _, pred_list = output.topk(topk)
        softmax_map, object_map = F.softmax(features, dim=0).max(dim=0)

    # prepare image classification results and respective activation map
    pred_list = list(pred_list.numpy())
    r = []
    activation_maps = []
    for pred in pred_list:
        r.append((pred, act[pred].item()))
        activation_maps.append(features[pred].numpy())

    # prepare object detection
    threshold_map = softmax_map < feature_threshold
    object_map[threshold_map] = -1
    object_map = object_map.numpy()
    return r, activation_maps, object_map


def idx_to_label(idx: int) -> str:
    global metadata
    if metadata is None:
        metadata = {}
        classes = torch.load('data/classes.bin')
        meta = torch.load('data/meta.bin')[0]
        metadata['classes'] = classes
        metadata['labels'] = {}
        for cls in classes:
            metadata['labels'][cls] = meta[cls][0]
    return metadata['labels'][metadata['classes'][idx]]


def wnid_to_idx(wnid: str) -> int:
    global metadata
    return metadata[1].index(wnid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Class Activation Map for images')
    parser.add_argument('-i', '--input', type=str, metavar='INPUT_FILE', required=True,
                        help="one input image for producing CAM")
    parser.add_argument('-o', '--output', type=str, metavar='OUTPUT_DIR', required=True,
                        help="output image directory")
    parser.add_argument('-w', '--initial-weight', type=str, metavar='INITIAL_CKPT', required=True,
                        help="path to initial weights")
    parser.add_argument('-k', '--topk', type=int, metavar='TOP_K', default=1,
                        help="produce TOP_K number of heat-maps")
    parser.add_argument('-t', '--threshold', type=float, default=0.33, metavar='DET_THRESHOLD',
                        help="threshold for object detection")
    parser.add_argument('--font', type=str, default=os.path.expanduser("~/Library/Fonts/OpenSans-Bold.ttf"),
                        help="font for rendering text")
    parser.add_argument('--activation', action='store_true',
                        help="produce activation map")
    parser.add_argument('--detection', action='store_true',
                        help="produce object detection map")

    args = parser.parse_args()

    setup_model(args.initial_weight)

    transform_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ])

    # initialize data
    image = PIL.Image.open(args.input)
    image: PIL.Image.Image = transform_image(image)

    pred_list, activation_maps, object_map = classify_image(image, topk=args.topk, feature_threshold=args.threshold)

    output_name = os.path.basename(args.output)
    os.makedirs(args.output, exist_ok=True)

    image.save(os.path.join(args.output, 'orig.png'))

    if args.activation:
        colormap: Callable = plt.cm.gnuplot2
        # unitize using same min/max
        activation_maps = np.asarray(activation_maps)
        activation_maps = (activation_maps - activation_maps.min()) / activation_maps.max()
        for pred, activation_map in zip(pred_list, activation_maps):
            pred_idx, confidence = pred
            # produce activation map
            activation_map = colormap(activation_map)
            activation_map[:, :, 3] = 0.75  # alpha channel
            activation_map *= 255
            activation_map = np.uint8(activation_map.clip(0, 255))
            activation_map_image = PIL.Image.fromarray(activation_map).resize(image.size, resample=PIL.Image.BILINEAR)
            # obtain the image with activation map overlay
            final_cam_image = PIL.Image.alpha_composite(image.convert('RGBA'), activation_map_image)
            # produce text label
            label = idx_to_label(pred_idx)
            draw = PIL.ImageDraw2.Draw(final_cam_image)
            color = 'white'
            font = PIL.ImageDraw2.Font(color, args.font, size=14)
            draw.text((5, 5), f"{label} {confidence:.4f}", font)

            final_cam_image.save(os.path.join(args.output, f'cam_{pred_idx}.png'))

    if args.detection:
        color_list = np.asarray(plt.cm.Set3.colors)
        color_list = np.concatenate((color_list, np.ones((12, 1))), axis=1)
        # set alpha channel
        color_list[:, 3] = 0.75
        size = object_map.shape
        size = (*size, 4)
        rgb_overlay = np.zeros(size)
        # render detection map
        labels = []
        for color_idx, cat_idx in enumerate(np.unique(object_map)):
            if cat_idx == -1:
                continue
            # will simply crash when more object types are present
            rgb_overlay[object_map == cat_idx] = color_list[color_idx]
            labels.append((color_list[color_idx], idx_to_label(cat_idx)))
        # produce image
        # TODO: use matplotlib to generate images with legend/labels
        rgb_overlay = np.uint8(np.clip(rgb_overlay * 255, 0, 255))
        rgb_overlay_img = PIL.Image.fromarray(rgb_overlay)
        rgb_overlay_img = rgb_overlay_img.resize(image.size, resample=PIL.Image.BILINEAR)

        final_detection_map = PIL.Image.alpha_composite(image.convert('RGBA'), rgb_overlay_img)
        final_detection_map.save(os.path.join(args.output, f"det_thr_{args.threshold:.2f}.png"))
        # eh? https://stackoverflow.com/questions/4534480
        patches = [
            Patch(color=color, label=label) for (color, label) in labels
        ]
        fig = plt.figure(figsize=(4, 3), dpi=600)
        fig.legend(patches, [label for (_, label) in labels], loc='center', frameon=False)
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight')
        legend = PIL.Image.open(buf)
        legend.save(os.path.join(args.output, f"det_legend_thr_{args.threshold:.2f}.png"))
