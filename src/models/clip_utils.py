import clip
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms


def load_clip_models(
    model_names: list[str], device, weights: list[float], flag_clip: bool = True
) -> list:
    """
    Загружает модели и препроцессоры CLIP.

    :param model_names: список имен моделей CLIP
    :param device: устройство для вычислений (CPU или CUDA)
    :param weights: вклад модели CLIP в итоговый эмбеддинг
    :param flag_clip: флаг использования 1 (False) или 2 (True) моделей CLIP

    :return: список кортежей (модель, препроцессор, вес)
    """
    models = []

    if flag_clip:
        for name, w in zip(model_names, weights):
            model, preprocess = clip.load(name, device=device)
            for p in model.parameters():
                p.requires_grad = False
            models.append((model, preprocess, w))
    else:
        name, w = model_names[0], weights[0]
        model, preprocess = clip.load(name, device=device)
        for p in model.parameters():
            p.requires_grad = False
        models.append((model, preprocess, w))

    return models


def preprocessing_text(models: list, prompts: list[str], device) -> torch.Tensor:
    """
    Принимает список текстовых промптов,
    возвращает один усреднённый и нормализованный эмбеддинг.

    :param models: список моделей, препроцессоров и весов CLIP
    :param prompts: список текстовых промптов
    :param device: устройство для вычислений (CPU или CUDA)

    :return: эмбеддинг текста в пространстве CLIP, отсоединенный от графа вычислений
             ([1, D] где D — размерность эмбеддинга CLIP)
    """
    total_embedding = 0.0
    weights_sum = sum(model[2] for model in models)

    tokens = clip.tokenize(prompts).to(device)

    for model in models:
        model[0].eval()
        text_features = model[0].encode_text(tokens)
        text_features = text_features / (
            text_features.norm(dim=-1, keepdim=True) + 1e-8
        )
        # Усреднение всех промптов
        text_mean = text_features.mean(dim=0, keepdim=True)
        # Нормализация
        text_mean = text_mean / (text_mean.norm(dim=-1, keepdim=True) + 1e-8)
        # Эмбеддинг с учетом веcа модели
        total_embedding += text_mean * (model[2] / weights_sum)

    # Нормализация
    total_embedding = total_embedding / (
        total_embedding.norm(dim=-1, keepdim=True) + 1e-8
    )

    return total_embedding.detach()


def preprocessing_image(models: list, image, device) -> torch.Tensor:
    """
    Если image - torch.Tensor, то преобразует его в PIL.
    Если image — PIL.Image, используется стандартный CLIP-препроцессор.
    Возвращает нормализованный эмбеддинг изображения.

    :param models: список моделей, препроцессоров и весов CLIP
    :param image: входное изображение (torch.Tensor [B, 3, H, W] или PIL.Image)
    :param device: устройство для вычислений (CPU или CUDA)

    :return: эмбеддинг изображения в пространстве CLIP ([B, D])
    """
    total_embedding = 0.0
    weights_sum = sum(model[2] for model in models)

    if isinstance(image, torch.Tensor):
        # Для torch.Tensor из StyleGAN2 с диапазоном [-1,1]
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: ((x + 1) / 2).clamp(0, 1)),
                transforms.Lambda(
                    lambda x: F.interpolate(
                        x, size=(224, 224), mode="bicubic", align_corners=False
                    )
                ),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        image_preproc = transform(image.to(device))

        for model in models:
            model[0].eval()
            image_features = model[0].encode_image(image_preproc)
            # Нормализация
            image_features = image_features / (
                image_features.norm(dim=-1, keepdim=True) + 1e-8
            )
            # Эмбеддинг с учетом веcа модели
            total_embedding += image_features * (model[2] / weights_sum)

    else:
        # Для PIL-изображений
        for model in models:
            model[0].eval()
            image_preproc = model[1](image).unsqueeze(0).to(device)
            image_features = model[0].encode_image(image_preproc)

            # Нормализация
            image_features = image_features / (
                image_features.norm(dim=-1, keepdim=True) + 1e-8
            )
            # Эмбеддинг с учетом веcа модели
            total_embedding += image_features * (model[2] / weights_sum)

    # Нормализация
    total_embedding = total_embedding / (
        total_embedding.norm(dim=-1, keepdim=True) + 1e-8
    )

    return total_embedding
