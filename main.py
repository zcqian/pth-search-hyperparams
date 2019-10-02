#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training script to look for hyper-parameters

This script will train models under given settings, given a pre-trained model as starting point.
It is intended to find whatever hyper-parameters that can squeeze some more accuracy from the current
trained model.

I am skeptical if I will find anything, and if this is a good idea, but why not?

"""

import argparse
import shutil
from ast import literal_eval

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import imagenet_1k
from helpers import set_all_rng_seed
from model import produce_model  # just returns a model which I cannot disclose
from train import process

if __name__ == '__main__':
    # I only use the correct ones so this is safe
    # which means I will not use "optimizer" or "lr_scheduler" or anything dumb
    # besides those, this dict will contain all optimizer constructors with their lower case names
    optimizer_dict = {k.lower(): optim.__dict__[k] for k in optim.__dict__ if not k[:2] == '__'}

    parser = argparse.ArgumentParser(description='Script to test settings')
    parser.add_argument('-b', '--batch-size', type=int, metavar='BATCH_SIZE',
                        help="size of each mini-batch")
    parser.add_argument('-o', '--optimizer', type=str, metavar='OPTIMIZER',
                        help="optimizer to use")
    parser.add_argument('--optimizer-options', type=str, metavar='OPTIMIZER_OPTS',
                        help="options for optimizer, in dict literal")
    parser.add_argument('-e', '--epochs', default=10, type=int, metavar='EPOCHS',
                        help='number of total epochs to run')
    parser.add_argument('--initial-weight', type=str, metavar='INITIAL_CKPT',
                        help="path to initial weights")

    # parse command line arguments
    args = parser.parse_args()
    batch_size = args.batch_size
    # the optimizer options are passed in dict literals via command line
    # like {'lr': 0.1, 'momentum': 0.9}
    # so that I can automate testing multiple of these combinations
    optimizer_options = literal_eval(args.optimizer_options)
    optimizer_name = args.optimizer.lower()
    epochs = args.epochs

    # setup the model and move to multiple GPUs
    model: nn.Module = produce_model()
    model.load_state_dict(torch.load(args.initial_weight))
    model = torch.nn.DataParallel(model)
    model.cuda()
    # optimizer and criterion
    optimizer = optimizer_dict[optimizer_name](model.parameters(), **optimizer_options)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # initialize data
    ds_trn, ds_val = imagenet_1k()
    dl_trn = DataLoader(ds_trn, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size, num_workers=4, pin_memory=True)

    # release unused objects
    del ds_trn, ds_val
    del parser, args, optimizer_name, optimizer_options, optimizer_dict

    # obtain best accuracy from checkpoint
    _, best_acc1, _ = process(dl_trn, model, criterion, None, mode='eval')
    print(f" *** Starting Acc@1 {best_acc1:.4f}")

    set_all_rng_seed(2019)

    for epoch in range(epochs):
        l, t1, t5 = process(dl_trn, model, criterion, optimizer, mode='train')
        print(f" * Train Loss {l:.4f} Acc@1 {t1:.4f} Acc@5 {t5:.4f}")
        l, t1, t5 = process(dl_trn, model, criterion, None, mode='eval')
        print(f" ** Test Loss {l:.4f} Acc@1 {t1:.4f} Acc@5 {t5:.4f}")
        is_best = t1 > best_acc1
        best_acc1 = max(t1, best_acc1)
        checkpoint_filename = f'checkpoint-epoch-{epoch}.pt'
        torch.save(model.state_dict(), checkpoint_filename)
        if is_best:
            shutil.copyfile(checkpoint_filename, 'model_best.pt')

    print(f" *** Best Acc@1 {best_acc1:.4f}")
