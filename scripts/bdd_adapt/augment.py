"""ALDI strong augmentation (photometric only, so boxes stay valid), random erasing and MIC.

All randomness comes from an explicit CPU generator, so a sample's augmentation is a pure
function of (run seed, stream, sample index) and resume is exact. Images are CxHxW in [0, 1].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# (probability, area scale range, aspect ratio range): ALDI's three RandomErase layers
ERASING = ((0.7, (0.05, 0.2), (0.3, 3.3)), (0.5, (0.02, 0.2), (0.1, 6.0)), (0.3, (0.02, 0.2), (0.05, 8.0)))
GRAY = torch.tensor([0.299, 0.587, 0.114])


def generator_for(*key) -> torch.Generator:
    """Deterministic CPU generator from a key; independent of global RNG state."""
    text = "/".join(str(part) for part in key)
    seed = 0
    for character in text.encode("utf-8"):
        seed = (seed * 1099511628211 + character) % (2 ** 63)
    return torch.Generator().manual_seed(seed)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def _apply(generator: torch.Generator, probability: float) -> bool:
    return float(torch.rand((), generator=generator)) < probability


def grayscale(image: torch.Tensor) -> torch.Tensor:
    return (image * GRAY.to(image).view(3, 1, 1)).sum(0, keepdim=True)


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    kernel = torch.exp(-offsets ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    batch = F.pad(image.unsqueeze(0), (radius, radius, radius, radius), mode="reflect")
    batch = F.conv2d(batch, kernel.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
    batch = F.conv2d(batch, kernel.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
    return batch.squeeze(0)


def photometric(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Contrast, brightness, saturation (p 0.8), grayscale (p 0.2), Gaussian blur (p 0.5)."""
    if _apply(generator, 0.8):
        factors = [_uniform(generator, 0.6, 1.4) for _ in range(3)]
        image = (image.mean() + factors[0] * (image - image.mean())).clamp(0, 1)
        image = (factors[1] * image).clamp(0, 1)
        image = (grayscale(image) + factors[2] * (image - grayscale(image))).clamp(0, 1)
    if _apply(generator, 0.2):
        image = grayscale(image).expand(3, -1, -1).contiguous()
    if _apply(generator, 0.5):
        image = gaussian_blur(image, _uniform(generator, 0.1, 2.0)).clamp(0, 1)
    return image


def random_erasing(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    image = image.clone()
    channels, height, width = image.shape
    for probability, (low, high), (ratio_low, ratio_high) in ERASING:
        if not _apply(generator, probability):
            continue
        for _ in range(100):
            area = _uniform(generator, low, high) * height * width
            ratio = _uniform(generator, ratio_low, ratio_high)
            h, w = round(math.sqrt(area * ratio)), round(math.sqrt(area / ratio))
            if 1 < w < width and 1 < h < height:
                top = int(torch.randint(0, height - h, (), generator=generator))
                left = int(torch.randint(0, width - w, (), generator=generator))
                image[:, top:top + h, left:left + w] = torch.rand((channels, h, w), generator=generator).to(image)
                break
    return image


def mic_mask(image: torch.Tensor, generator: torch.Generator, ratio: float = 0.5, block: int = 32) -> torch.Tensor:
    """Masked image consistency: zero a random `ratio` of block x block patches."""
    _, height, width = image.shape
    keep = (torch.rand((round(height / block), round(width / block)), generator=generator) > ratio).float()
    keep = F.interpolate(keep[None, None], size=(height, width), mode="nearest")[0].to(image)
    return image * keep


def strong(image: torch.Tensor, generator: torch.Generator, *, erase: bool, mic: bool) -> torch.Tensor:
    image = photometric(image, generator)
    if erase:
        image = random_erasing(image, generator)
    if mic:
        image = mic_mask(image, generator)
    return image
