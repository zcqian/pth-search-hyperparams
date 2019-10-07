from typing import Callable, Union

from tqdm import tqdm
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from helpers import Meter
from third_party import accuracy


def process(data_loader: DataLoader,
            model: nn.Module,
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer: Union[Optimizer, None],
            mode: str,
            progress: bool,
            device: torch.device):
    """Process one epoch of data given train/eval mode

    Arguments:
        data_loader:
        model:
        criterion:
        optimizer: all these are pretty much self explanatory
        mode: 'train' or 'eval'
        progress: whether to show progress bar
        device: device to move data to

    Returns:
        Tuple of three floating number, mean {loss, top-1 accuracy, top-5 accuracy} for this epoch
    """

    losses = Meter()
    top1, top5 = Meter(), Meter()

    if mode == 'train':
        model.train()
        context = torch.enable_grad
        if optimizer is None:
            raise RuntimeError("Invoked training without optimizer")
    elif mode == 'eval':
        model.eval()
        context = torch.no_grad
    else:
        raise RuntimeError(f"Invoked with invalid mode: {mode}")

    iterator = tqdm(data_loader, leave=False, disable=not progress, )

    with context():
        for data, target in iterator:
            # move to GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            # compute output
            output = model(data)
            loss = criterion(output, target)
            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if mode == 'train':
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                pass
            # update trackers
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            iterator.set_postfix_str(f"A@1 {acc1[0]:6.3f} ({top1.mean:6.3f}) L {loss.item():6.4f} ({losses.mean:6.4f})")
    return losses.mean, top1.mean, top5.mean


