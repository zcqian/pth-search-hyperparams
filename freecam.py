import argparse

import PIL.Image
import torch
import torchvision.transforms as transforms
from torchvision.models import MobileNetV2

from model import produce_model
from matplotlib import cm
import numpy as np

from typing import Callable
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Class Activation Map for images')
    parser.add_argument('-i', '--input', type=str, metavar='INPUT_FILE', required=True,
                        help="one input image for producing CAM")
    parser.add_argument('-o', '--output', type=str, metavar='OUTPUT_DIR', required=True,
                        help="output image directory")
    parser.add_argument('-w', '--initial-weight', type=str, metavar='INITIAL_CKPT', required=True,
                        help="path to initial weights")

    args = parser.parse_args()

    model: MobileNetV2 = produce_model()
    model.load_state_dict(torch.load(args.initial_weight))

    model = model.eval()

    # In my model(s) the feature extractor outputs CAMs directly without any extra steps
    feature_extractor = model.features

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # initialize data
    image = PIL.Image.open(args.input)
    tensor: torch.Tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax()
        heatmap = feature_extractor(tensor)[0]
        heatmap: np.ndarray = heatmap[pred].numpy()

    heatmap = (heatmap - heatmap.min()) / heatmap.max()
    colormap: Callable = cm.hot
    heatmap = np.uint8((colormap(heatmap) * 255).clip(0, 255))

    heatmap_img = PIL.Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize(image.size, resample=PIL.Image.LINEAR)
    r, g, b, _ = heatmap_img.split()
    a = PIL.Image.new('L', heatmap_img.size, 127)
    heatmap_img = PIL.Image.merge('RGBA', (r, g, b, a))

    final_image = PIL.Image.alpha_composite(image.convert('RGBA'), heatmap_img)

    filename = os.path.basename(args.input)
    filename = filename.split('.')[0]
    filename = f"{filename}_pred{pred.item()}.png"
    output_path = os.path.join(args.output, filename)
    final_image.save(output_path)
