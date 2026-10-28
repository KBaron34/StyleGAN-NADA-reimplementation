import torch

from src.models.clip_utils import preprocessing_image
from src.losses.directional import cosine_dist, delta_image, clip_loss
from src.losses.regularization import l2_reg, patch_reg, lpips_reg
from src.training.layers import get_conv_layer
from src.analysis.plots import compute_grad_norms


def optimize_w_plus(
    generator,
    models: list,
    device,
    batch_size_w: int,
    lr_w: float,
    k: int,
    max_norm: float,
    text_w_plus_features: torch.Tensor,
    c: torch.Tensor = None,
    flag_gradient_clipping: bool = True,
) -> torch.Tensor:
    """
    Оптимизирует вектор w_plus_target, ограничивает норму градиента,
    удаляет отработанные torch.Tensor, очищает память
    и возвращает torch.Tensor топ-k индексов для дальнейшего обучения generator.

    :param generator: обучаемый генератор StyleGAN2
    :param models: список моделей, препроцессоров и весов CLIP
    :param device: устройство для вычислений (CPU или CUDA)
    :param batch_size_w: размерность батча для оптимизации векторов из пространства W+
    :param lr_w: скорость обучения для оптимизатора векторов из пространства W+
    :param k: количество слоев которые обучаются
    :param max_norm: максимальная норма градиентов
    :param text_w_plus_features: эмбеддинг текста в пространстве CLIP ([1, D])
    :param c: условие для генератора
    :param flag_gradient_clipping: флаг использования ограничения нормы градиента

    :return: torch.Tensor топ-k индексов слоев для обучения
    """
    # Создание латента для генерации
    latent_plus = torch.randn(batch_size_w, generator.z_dim, device=device)

    # Генерация векторов пространства W+ ([batch_size, num_layers, dim])
    w_plus_source = generator.mapping(latent_plus, c)
    w_plus_target = generator.mapping(latent_plus, c)
    w_plus_target = w_plus_target.detach().requires_grad_(True)

    optimizer_w_plus = torch.optim.AdamW([w_plus_target], lr_w)

    optimizer_w_plus.zero_grad()

    # Генерация изображения и получение из него эмбеддинга
    image_w_plus = generator.synthesis(w_plus_target, noise_mode="const")

    image_w_plus_features = preprocessing_image(
        models=models, image=image_w_plus, device=device
    )

    loss_w_plus = cosine_dist(emb_1=image_w_plus_features, emb_2=text_w_plus_features)

    loss_w_plus.backward()

    # Ограничение нормы градиентов
    if flag_gradient_clipping:
        torch.nn.utils.clip_grad_norm_([w_plus_target], max_norm)

    optimizer_w_plus.step()

    # Вычисление различий между оптимизированными и исходными векторами из W+
    delta_w_plus = w_plus_target.detach() - w_plus_source.detach()
    # Вычисление L2-нормы по последнему измерению, усреднённой по батчу
    delta_norm = delta_w_plus.mean(dim=0).norm(p=2, dim=-1)

    # Создание torch.Tensor топ-k индексов
    topk_indices = torch.topk(delta_norm, k).indices

    # Удаление отработанных torch.Tensor
    del latent_plus, w_plus_source, w_plus_target, image_w_plus
    del image_w_plus_features, loss_w_plus, delta_w_plus, delta_norm

    return topk_indices


def optimize_generator(
    optimizer,
    generator,
    generator_frozen,
    models: list,
    device,
    lst_block: list[str],
    train_params: list,
    emb_delta_text: torch.Tensor,
    lpips_metric,
    lambda_patch: float,
    lambda_l2: float,
    lambda_lpips: float,
    max_norm: float,
    patch_size: int,
    num_patches: int,
    batch_size: int,
    c: torch.Tensor = None,
    flag_reg_l2: bool = True,
    flag_reg_patch: bool = True,
    flag_reg_lpips: bool = True,
    flag_adaptive_layers: bool = True,
    flag_gradient_clipping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Обновляет обучаемые параметры оптимизатора.
    Создает латент для обучения и генерирует из него изображения двумя генераторами.
    Вычисляет вектор визуального направления между изображением до и после fine-tuning.
    Вычисляет патч-лосс, l2-регуляризационный лосс, направленный CLIP-лоос и
    финальный лосс. Делает обратный проход, ограничивает норму градиента,
    делает шаг оптимизации, удаляет отработанные torch.Tensor и очищает память.

    :param optimizer: оптимизатор
    :param generator: обучаемый генератор StyleGAN2
    :param generator_frozen: замороженный генератор StyleGAN2
    :param models: список моделей, препроцессоров и весов CLIP
    :param device: устройство для вычислений (CPU или CUDA)
    :param lst_block: список блоков generator.synthesis (из generator_block)
    :param train_params: список размороженных параметров
    :param emb_delta_text: вектор текстового направления ([1, D])
    :param lpips_metric: нейросетевая метрика perceptual loss (LPIPS)
    :param lambda_patch: коэффициент патч-регуляризации
    :param lambda_l2: коэффициент l2-регуляризации
    :param lambda_lpips: коэффициент LPIPS-регуляризации
    :param max_norm: максимальная норма градиентов
    :param patch_size: размер квадрата патча
    :param num_patches: количество патчей
    :param batch_size: размерность батча для обучения
    :param c: условие для генератора
    :param flag_reg_l2: флаг использования l2-регуляризации
    :param flag_reg_patch: флаг использования patch-лосса
    :param flag_reg_lpips: флаг использования lpips-лосса
    :param flag_adaptive_layers: флаг использования адаптивного выбора обучаемых слоев
    :param flag_gradient_clipping: флаг использования ограничения нормы градиента

    :return: loss (scalar), directional_loss (scalar), lpips_loss (scalar),
             lst_norms — нормы градиентов для каждого блока
    """
    if flag_adaptive_layers:
        # Обновление обучаемых параметров для оптимизатора
        optimizer.param_groups[0]["params"] = train_params

    optimizer.zero_grad()

    # Создание латента для обучения
    latent = torch.randn(batch_size, generator.z_dim, device=device)

    # Генерация изображения замороженным генератором
    with torch.no_grad():
        image_frozen = generator_frozen(
            latent, c, truncation_psi=1.0, noise_mode="const"
        )

    # Генерация изображения обучаемым генератором
    image = generator(latent, c, truncation_psi=1.0, noise_mode="const")

    # Вычисление вектора визуального направления между изображением до и после fine-tuning
    emb_delta_image = delta_image(
        models=models, image=image, image_frozen=image_frozen, device=device
    )

    # Вычисление l2-регуляризационного лосса
    if flag_reg_l2:
        reg_loss = l2_reg(
            generator=generator, generator_frozen=generator_frozen, device=device
        )
    else:
        reg_loss = torch.tensor(0.0, device=device)

    # Вычисление патч-лосса
    if flag_reg_patch:
        patch_loss = patch_reg(
            models=models,
            image=image,
            image_frozen=image_frozen,
            device=device,
            patch_size=patch_size,
            num_patches=num_patches,
        )
    else:
        patch_loss = torch.tensor(0.0, device=device)

    # Вычисление perceptual loss (LPIPS)
    lpips_loss = lpips_reg(
        lpips_metric=lpips_metric, image=image, image_frozen=image_frozen
    )

    # Вычисление направленного CLIP-лосса
    directional_loss = clip_loss(delta_image=emb_delta_image, delta_text=emb_delta_text)

    # Вычисление финального лосса
    if flag_reg_lpips:
        loss = (
            directional_loss
            + lambda_l2 * reg_loss
            + lambda_patch * patch_loss
            + lambda_lpips * lpips_loss
        )
    else:
        loss = directional_loss + lambda_l2 * reg_loss + lambda_patch * patch_loss

    loss.backward()

    # Вычисление L2-нормы градиентов для каждого блока генератора
    lst_norms = compute_grad_norms(generator, lst_block)

    # Ограничение нормы градиентов
    if flag_gradient_clipping:
        torch.nn.utils.clip_grad_norm_(train_params, max_norm)

    optimizer.step()

    # Удаление отработанных torch.Tensor
    del latent, image_frozen, image, emb_delta_image, reg_loss, patch_loss

    return loss, directional_loss, lpips_loss, lst_norms
