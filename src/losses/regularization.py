import numpy as np
import torch
import lpips

from src.utils.image import extract_patches
from src.models.clip_utils import preprocessing_image
from src.losses.directional import cosine_dist


def l2_reg(generator, generator_frozen, device) -> torch.Tensor:
    """
    Вычисляет l2-регуляризацию, которая не дает сильно отклониться
    параметрам обучаемого генератора от параметров замороженного генератора

    :param generator: обучаемый генератор StyleGAN2
    :param generator_frozen: замороженный генератор StyleGAN2
    :param device: устройство для вычислений (CPU или CUDA)

    :return: l2-регуляризация по всем размороженным параметрам (scalar)
    """
    reg_loss = 0.0
    reg_count = 0
    for p_frozen, p in zip(
        generator_frozen.synthesis.parameters(), generator.synthesis.parameters()
    ):
        # Вычисление L2‑расстояния между параметром обучаемого и замороженного генератора
        if p.requires_grad:
            reg_loss += (p - p_frozen.detach()).pow(2).sum()
            # Учет количества элементов в текущем параметре (для последующей нормализации)
            reg_count += p.numel()
    # Нормализация по числу параметров
    if reg_count > 0:
        reg_loss = reg_loss / (reg_count + 1e-8)
    else:
        reg_loss = torch.tensor(0.0, device=device)

    return reg_loss


def patch_reg(
    models: list,
    image: torch.Tensor,
    image_frozen: torch.Tensor,
    device,
    patch_size: int = 64,
    num_patches: int = 8,
) -> torch.Tensor:
    """
    Извлекает патчи из изиображения и вычисляет патч-лосс.
    Удаляет отработанные torch.Tensor и очищает память.

    :param models: список моделей, препроцессоров и весов CLIP
    :param image: изображение созданное обучаемым генератором (B, 3, H, W])
    :param image_frozen: изображение созданное замороженным генератором ([B, 3, H, W])
    :param device: устройство для вычислений (CPU или CUDA)
    :param patch_size: размер квадрата патча
    :param num_patches: количество патчей

    :return: усредненный косинусный патч-лосс (scalar)
    """
    # Извлечение патчей
    patches = extract_patches(
        image=image, patch_size=patch_size, num_patches=num_patches
    )
    patches_frozen = extract_patches(
        image=image_frozen, patch_size=patch_size, num_patches=num_patches
    )

    # Преобразование патчей в эмбеддинг
    emb_patches = preprocessing_image(models=models, image=patches, device=device)
    emb_patches_frozen = preprocessing_image(
        models=models, image=patches_frozen, device=device
    )

    # Вычисление косинусного расстояния между эмбеддингами
    patch_loss = cosine_dist(emb_1=emb_patches, emb_2=emb_patches_frozen)

    # Удаление отработанных torch.Tensor
    del patches, patches_frozen, emb_patches, emb_patches_frozen

    return patch_loss


def lpips_fn(net: str, device):
    """
    Загружает нейросетевую метрику perceptual loss (LPIPS).

    :param net: CNN-модель, на признаках которой считается LPIPS
    :param device: устройство для вычислений (CPU или CUDA)

    :return: нейросетевая метрика perceptual loss (LPIPS)
    """
    return lpips.LPIPS(net=net).to(device)


def lpips_reg(
    lpips_metric, image: torch.Tensor, image_frozen: torch.Tensor
) -> torch.Tensor:
    """
    Вычисляет perceptual loss (LPIPS).

    :param lpips_metric: нейросетевая метрика perceptual loss (LPIPS)
    :param image: изображение созданное обучаемым генератором
                  ([B, 3, H, W], нормализованное в [-1,1])
    :param image_frozen: изображение созданное замороженным генератором
                         (([B, 3, H, W], нормализованное в [-1,1]))

    :return: усредненный perceptual loss (scalar)
    """
    # Изображения с осями [B, 3, H, W], нормализованные в [-1, 1]
    lpips_loss = lpips_metric(image, image_frozen)
    return lpips_loss.mean()


def diversity_fn(
    generator, latent_div: torch.Tensor, lpips_metric, c: torch.Tensor = None
) -> float:
    """
    Вычисляет diversity (разнообразие) сгенерированных изображений

    :param generator: обучаемый генератор StyleGAN2
    :param latent_div: батч латентов для diversity ([N, Z])
    :param lpips_metric: нейросетевая метрика perceptual loss (LPIPS)
    :param c: условие для генератора

    :return: среднее значение LPIPS по всем парам изображений
    """
    # Генерация изображений
    generator.eval()
    with torch.no_grad():
        image_div = generator(latent_div, c, truncation_psi=0.7, noise_mode="const")

    diversity_lst = []
    for i in range(latent_div.shape[0]):
        for j in range(i + 1, latent_div.shape[0]):
            lpips = lpips_metric(
                image_div[i].unsqueeze(0), image_div[j].unsqueeze(0)
            ).item()
            diversity_lst.append(lpips)

    generator.train()

    # Удаление отработанных torch.Tensor
    del image_div

    return float(np.mean(diversity_lst))
