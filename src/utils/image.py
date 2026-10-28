import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def extract_patches(
    image: torch.Tensor, patch_size: int, num_patches: int
) -> torch.Tensor:
    """
    Извлекает случайные патчи из изображения.

    :param image: тензор [B, C, H, W]
    :param patch_size: размер квадрата патча
    :param num_patches: количество патчей

    :return: тензор [B*num_patches, C, patch_size, patch_size]
    """
    B, C, H, W = image.shape

    if patch_size > H or patch_size > W:
        raise ValueError(
            f"patch_size={patch_size} больше размера изображения ({H}x{W})"
        )

    patches = []
    for _ in range(num_patches):
        i = torch.randint(0, H - patch_size, (1,)).item()
        j = torch.randint(0, W - patch_size, (1,)).item()
        patch = image[:, :, i : i + patch_size, j : j + patch_size]
        patches.append(patch)

    return torch.cat(patches, dim=0)


def visual_image(image_sample: torch.Tensor) -> torch.Tensor:
    """
    Визуализирует сгенерированное изображение.

    :param image_sample: тензор (изображение сгенерированное генератором)

    :return: сетка тензоров (изображений)
    """
    img = (image_sample.clamp(-1, 1) + 1) / 2.0
    grid = make_grid(img, nrow=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
    plt.axis("off")
    plt.show()

    return grid
