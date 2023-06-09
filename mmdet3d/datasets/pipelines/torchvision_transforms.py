"""This is adapted from https://github.com/open-mmlab/OpenSelfSup/blob/1db69ece
bbc129e8fa90cdcea6f2082f0a4e3d17/openselfsup/datasets/pipelines/transforms.py.

However, their pipeline directly passes images into transforms, while for us,
we need to pass in data_dicts. So, have to generate a new class that overwrites
the __call__ method. idk if this is hacky or elegant tbh
reference: https://www.geeksforgeeks.org/create-classes-dynamically-in-python/

Further, add a PIL to numpy converter
"""
import inspect
import numpy as np
import PIL
import torch
from mmcv.utils import build_from_cfg
from PIL import ImageFilter
from torchvision import transforms as _transforms

from mmdet.datasets import PIPELINES

# # Don't use geometric transforms here, since the parameters are not stored.
# # Just photometric.
# def class_wrapper(c):

#     def call_wrapper(self, results):
#         for key in results.get('img_fields', ['img']):
#             results[key] = super(self.__class__, self).__call__(results[key])
#         return results

#     return type('TV' + c.__name__, (c, ), {'__call__': call_wrapper})


# # # register all existing transforms in torchvision
# # # This list is a *little* longer because openselfsup does not use mmdet augs
# # _EXCLUDED_TRANSFORMS = ['GaussianBlur', 'AutoAugment', 'AutoAugmentPolicy',
# #                         'Compose', 'Enum', 'InterpolationMode', 'Normalize',
# #                         'Pad', 'RandomCrop', 'Resize', 'ToTensor']

# # List of transforms to NOT include from torchvision.
# # Ones included will have prefix "TV"
# _EXCLUDED_TRANSFORMS = [
#     'ToTensor', 'AutoAugmentPolicy', 'Enum', 'InterpolationMode'
# ]
# for m in inspect.getmembers(_transforms, inspect.isclass):
#     if m[0] not in _EXCLUDED_TRANSFORMS:
#         PIPELINES.register_module(class_wrapper(m[1]))


@PIPELINES.register_module
class TVToPILImage(_transforms.ToPILImage):
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = super().__call__(results[key])
        return results


@PIPELINES.register_module
class TVColorJitter(_transforms.ColorJitter):
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = super().__call__(results[key])
        return results


@PIPELINES.register_module
class TVRandomGrayscale(_transforms.RandomGrayscale):
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = super().__call__(results[key])
        return results


@PIPELINES.register_module
class TVRandomErasing(_transforms.RandomErasing):
    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = super().__call__(results[key])
        return results


@PIPELINES.register_module
class ToNumpy(object):

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            if isinstance(results[key], torch.Tensor):
                raise Exception('Not Yet implemented')
            elif isinstance(results[key], PIL.Image.Image):
                results[key] = np.asarray(results[key])
            elif isinstance(results[key], np.ndarray):
                pass
            else:
                raise Exception('{} type not supported'.format(
                    results[key].__class__))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class TVToTensor(object):
    """ToTensor is already defined in mmdet, want original functionality of
    ToTensor of torvision.

    TV stands for torchvision.
    """

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = _transforms.ToTensor()(results[key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class RandomAppliedTrans(object):
    """Randomly applied transformations.

    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, results):
        results = self.trans(results)
        # because internal transformations still expect dicts

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR
    https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
            results[key] = results[key].filter(
                ImageFilter.GaussianBlur(radius=sigma))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str