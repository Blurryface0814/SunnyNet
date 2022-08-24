import random

import torch


class RosCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, distance, reflectivity):
        for t in self.transforms:
            distance, reflectivity = t(distance, reflectivity)
        return distance, reflectivity


class RosToTensor(object):

    def __call__(self, distance, reflectivity):
        distance = distance.unsqueeze(0)
        reflectivity = reflectivity.unsqueeze(0)

        return distance, reflectivity


class RosNormalize(object):
    """Normalize a tensor with mean and standard deviation.

    Args:
        mean (sequence): Sequence of means [distance_mean, reflectivity_mean].
        std (sequence): Sequence of standard deviations [distance_std, reflectivity_std].
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, distance, reflectivity):
        distance = self._normalize(distance, self.mean[0], self.std[0])
        reflectivity = self._normalize(reflectivity, self.mean[1], self.std[1])

        return distance, reflectivity

    @staticmethod
    def _normalize(inp, mean, std):
        mean = torch.tensor(mean, dtype=inp.dtype, device=inp.device)
        std = torch.tensor(std, dtype=inp.dtype, device=inp.device)
        return (inp - mean) / std


class RosRandomHorizontalFlip(object):
    """Horizontally flip the given tensors randomly with a given probability.

    Args:
        p (float): probability of the tensors being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, distance, reflectivity):
        if random.random() < self.p:
            distance = distance.flip(1)
            reflectivity = reflectivity.flip(1)

        return distance, reflectivity
