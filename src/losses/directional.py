import torch
import torch.nn.functional as F

from src.utils.text import make_prompt_list
from src.models.clip_utils import preprocessing_text, preprocessing_image


def cosine_dist(emb_1: torch.Tensor, emb_2: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет глобальный CLIP-loss (cosine distance).

    :param emb_1: вектор изображения в пространстве CLIP ([B, D])
    :param emb_2: вектор текста в пространстве CLIP ([B, D])

    :return: среднее косинусное расстояние по батчу (scalar)
    """
    global_loss = 1 - F.cosine_similarity(emb_1, emb_2, dim=-1)
    return global_loss.mean()


def delta_text(
    models: list,
    text_target: str,
    text_source: str,
    templates: list[str],
    device,
    flag_prompt: bool = True,
) -> torch.Tensor:
    """
    Вычисляет вектор текстового направления
    между таргетом и текущим стилем для множества промптов.

    :param models: список моделей, препроцессоров и весов CLIP
    :param text_target: текст целевого стиля
    :param text_source: текст исходного стиля
    :param templates: список шаблонов промптов
    :param device: устройство для вычислений (CPU или CUDA)
    :param flag_prompt: флаг использования 1 (False) или нескольких (True) промптов

    :return: нормализованный вектор текстового направления ([1, D])
    """
    # Генерация списка промптов
    prompts_src = make_prompt_list(
        word=text_source, templates=templates, flag_prompt=flag_prompt
    )
    prompts_tgt = make_prompt_list(
        word=text_target, templates=templates, flag_prompt=flag_prompt
    )

    # Получение эмбеддингов в пространстве CLIP для текстовых промптов
    txt_src = preprocessing_text(models=models, prompts=prompts_src, device=device)
    txt_tgt = preprocessing_text(models=models, prompts=prompts_tgt, device=device)

    # Вычисление текстового направления и нормализация
    delta = txt_tgt - txt_src
    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)

    return delta


def delta_image(
    models: list, image: torch.Tensor, image_frozen: torch.Tensor, device
) -> torch.Tensor:
    """
    Вычисляет вектор визуального направления
    между изображением до и после fine-tuning.

    :param models: список моделей, препроцессоров и весов CLIP
    :param image: изображение созданное обучаемым генератором
    :param image_frozen: изображение созданное замороженным генератором
    :param device: устройство для вычислений (CPU или CUDA)

    :return: нормализованный вектор визуального направления ([B, D])
    """
    # Получение эмбеддингов в пространстве CLIP для изображений
    img = preprocessing_image(models=models, image=image, device=device)
    img_frz = preprocessing_image(models=models, image=image_frozen, device=device)

    # Вычисление визуального направления и нормализация
    delta = img - img_frz
    delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-8)

    return delta


def clip_loss(delta_image: torch.Tensor, delta_text: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет направленный CLIP-loss.

    :param delta_image: вектор визуального направления ([B, D])
    :param delta_text: вектор текстового направления ([1, D] или [B, D])

    :return: усредненный направленный CLIP-loss (scalar)
    """
    direction_clip_loss = 1 - F.cosine_similarity(delta_image, delta_text, dim=-1)
    return direction_clip_loss.mean()
