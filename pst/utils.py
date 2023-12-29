import math
from typing import Optional

import torch
from torch import Tensor
from torch_scatter import scatter


def to_dense_batch(x, ptr, return_mask=False, fill_value=0.0):
    batch_size = len(ptr) - 1
    num_nodes = torch.diff(ptr)
    max_num_nodes = num_nodes.max().item()
    all_num_nodes = ptr[-1].item()

    cls_tokens = False
    x_size = len(x[0]) if isinstance(x, (list, tuple)) else len(x)
    if x_size > all_num_nodes:
        cls_tokens = True
        max_num_nodes += 1

    device = x[0].device if isinstance(x, (list, tuple)) else x.device

    batch = torch.arange(batch_size, dtype=torch.long, device=device)
    batch = batch.repeat_interleave(num_nodes)
    idx = torch.arange(all_num_nodes, dtype=torch.long, device=device)
    idx = (idx - ptr[batch]) + batch * max_num_nodes

    size = [batch_size * max_num_nodes]
    if isinstance(x, (list, tuple)):
        new_x = []
        for j in range(len(x)):
            xj = x[j].new_full(size + list(x[j].size())[1:], fill_value)
            xj[idx] = x[j][:all_num_nodes]
            xj = xj.view([batch_size, max_num_nodes] + list(x[j].size())[1:])
            if cls_tokens:
                xj[:, -1, ...] = x[j][all_num_nodes:]
            new_x.append(xj)
    else:
        new_x = x.new_full(size + list(x.size())[1:], fill_value)
        new_x[idx] = x[:all_num_nodes]
        new_x = new_x.view([batch_size, max_num_nodes] + list(x.size())[1:])
        if cls_tokens:
            new_x[:, -1, ...] = x[all_num_nodes:]

    if return_mask:
        mask = torch.ones(batch_size * max_num_nodes, dtype=torch.bool, device=device)
        mask[idx] = 0
        mask = mask.view(batch_size, max_num_nodes)
        if cls_tokens:
            mask[:, -1] = 0
        return new_x, mask
    return new_x


def unpad_dense_batch(x, mask, ptr):
    bsz, n, d = x.shape
    max_num_nodes = torch.diff(ptr).max().item()
    cls_tokens = False
    if n > max_num_nodes:
        cls_tokens = True
    if cls_tokens:
        new_x = x[:, :-1][~mask[:, :-1]]
        new_x = torch.cat((new_x, x[:, -1]))
    else:
        new_x = x[~mask]
    return new_x


def to_nested_tensor(
    x: Tensor,
    batch: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Copied from PyG implementation"""
    if ptr is not None:
        offsets = ptr[1:] - ptr[:-1]
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    elif batch is not None:
        offsets = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        sizes: List[int] = offsets.tolist()
        xs = list(torch.split(x, sizes, dim=0))
    else:
        xs = [x]

    # This currently copies the data, although `x` is already contiguous.
    # Sadly, there does not exist any (public) API to preven this :(
    return torch.nested.as_nested_tensor(xs)


def from_nested_tensor(
    x: Tensor,
    return_batch: bool = False,
):
    """Copied from PyG implementation"""
    if not x.is_nested:
        raise ValueError("Input tensor in 'from_nested_tensor' is not nested")

    sizes = x._nested_tensor_size()

    for dim, (a, b) in enumerate(zip(sizes[0, 1:], sizes.t()[1:])):
        if not torch.equal(a.expand_as(b), b):
            raise ValueError(
                f"Not all nested tensors have the same size "
                f"in dimension {dim + 1} "
                f"(expected size {a.item()} for all tensors)"
            )

    out = x.contiguous().values()
    out = out.view(-1, *sizes[0, 1:].tolist())

    if not return_batch:
        return out

    batch = torch.arange(x.size(0), device=x.device)
    batch = batch.repeat_interleave(sizes[:, 0].to(batch.device))

    return out, batch


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_inverse_sqrt_schedule_with_warmup(optimizer, warmup_epochs, max_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return max(1e-06, epoch / max(1, warmup_epochs))
        return math.sqrt(warmup_epochs / epoch)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
