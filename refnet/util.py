import torch
import numpy.random as random
import itertools
import importlib

from tqdm import tqdm
from inspect import isfunction


def exists(x):
    return x is not None

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def zero_drop(x, p):
    return append_dims(torch.bernoulli((1 - p) * torch.ones(x.shape[0], device=x.device, dtype=x.dtype)), x.ndim)


def expand_to_batch_size(x, bs):
    x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
    return x


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def fitting_weights(model, sd):
    n_params = len([name for name, _ in
                    itertools.chain(model.named_parameters(),
                                    model.named_buffers())])
    for name, param in tqdm(
            itertools.chain(model.named_parameters(),
                            model.named_buffers()),
            desc="Fitting old weights to new weights",
            total=n_params
    ):
        if not name in sd:
            continue
        old_shape = sd[name].shape
        new_shape = param.shape
        assert len(old_shape) == len(new_shape)
        if len(new_shape) > 2:
            # we only modify first two axes
            assert new_shape[2:] == old_shape[2:]
        # assumes first axis corresponds to output dim
        if not new_shape == old_shape:
            new_param = param.clone()
            old_param = sd[name]
            if len(new_shape) == 1:
                for i in range(new_param.shape[0]):
                    new_param[i] = old_param[i % old_shape[0]]
            elif len(new_shape) >= 2:
                for i in range(new_param.shape[0]):
                    for j in range(new_param.shape[1]):
                        new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                n_used_old = torch.ones(old_shape[1])
                for j in range(new_param.shape[1]):
                    n_used_old[j % old_shape[1]] += 1
                n_used_new = torch.zeros(new_shape[1])
                for j in range(new_param.shape[1]):
                    n_used_new[j] = n_used_old[j % old_shape[1]]

                n_used_new = n_used_new[None, :]
                while len(n_used_new.shape) < len(new_shape):
                    n_used_new = n_used_new.unsqueeze(-1)
                new_param /= n_used_new

            sd[name] = new_param
    return sd

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params

def random_mask(img, grid_num, grid_size=128, image_size=512, mask_value=1):
    max_grid_num = image_size // grid_size
    grids = random.randint(max_grid_num, size=(grid_num,2))
    for grid in grids:
        r, c = grid
        img[:, :, r*grid_size: (r+1)*grid_size, c*grid_size: (c+1)*grid_size] = mask_value
    return img