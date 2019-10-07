import os

import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageNet


def imagenet_1k():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    dataset_dir = os.path.expanduser('~/Datasets/imagenet')
    dataset_train = ImageNet(
        dataset_dir, split='train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_val = ImageNet(
        dataset_dir, split='val',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset_train, dataset_val


def imagenet_subset(num_categories: int, *, random: bool):
    if random:
        raise NotImplementedError
    ds_trn, ds_val = imagenet_1k()
    accept_categories = list(range(num_categories))
    subset_idx = [i for i in range(len(ds_trn)) if ds_trn.targets[i] in accept_categories]
    ds_trn = Subset(ds_trn, subset_idx)
    subset_idx = [i for i in range(len(ds_val)) if ds_val.targets[i] in accept_categories]
    ds_val = Subset(ds_val, subset_idx)
    return ds_trn, ds_val


def imagenet_100():
    return imagenet_subset(100, random=False)


def imagenet_10():
    return imagenet_subset(10, random=False)
