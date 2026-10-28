import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from configs.config import TrainConfig
from src.utils.text import make_prompt_list
from src.utils.image import visual_image
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.templates import TEMPLATES
from src.models.stylegan import load_stylegan2_generator, generator_block
from src.models.clip_utils import load_clip_models, preprocessing_text
from src.losses.directional import delta_text
from src.losses.regularization import lpips_fn, diversity_fn
from src.training.layers import unfreeze_layers
from src.training.optimizer import optimize_w_plus, optimize_generator
from src.analysis.metrics import log_metrics

# Глобальный кэш CLIP-моделей (загружаются один раз)
CLIP_MODELS_LOADED = False
GLOBAL_CLIP_MODELS = None


import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from configs.config import TrainConfig
from src.utils.text import make_prompt_list
from src.utils.image import visual_image
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.templates import TEMPLATES
from src.models.stylegan import load_stylegan2_generator, generator_block
from src.models.clip_utils import load_clip_models, preprocessing_text
from src.losses.directional import delta_text
from src.losses.regularization import lpips_fn, diversity_fn
from src.training.layers import unfreeze_layers
from src.training.optimizer import optimize_w_plus, optimize_generator
from src.analysis.metrics import log_metrics

# Глобальный кэш CLIP-моделей (загружаются один раз)
CLIP_MODELS_LOADED = False
GLOBAL_CLIP_MODELS = None


def train_generator(
    cfg: TrainConfig, device, c: torch.Tensor = None
) -> tuple[pd.DataFrame, pd.DataFrame, list[list]]:
    """
    Обучает генератор StyleGAN2 для изменения изображений в направлении текста
    с использованием CLIP-косинусного расстояния в качестве функции потерь.
    Каждые 50 эпох сохраняет сгенерированные изображения.
    В конце визуализирует график лосса.

    :param cfg: конфигурация обучения (TrainConfig)
    :param device: устройство для вычислений (CPU или CUDA)
    :param c: условие для генератора
    
    :return: метрики по эпохам, усредненные по эпохам нормы градиентов для каждого блока,
             нормы градиентов для каждого блока
    """
    global CLIP_MODELS_LOADED, GLOBAL_CLIP_MODELS

    # Создание директорий для сохранения результатов
    os.makedirs(cfg.save_dir, exist_ok=True)

    images_dir = os.path.join(cfg.save_dir, "images")
    ckpt_dir = os.path.join(cfg.save_dir, "checkpoints")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Функционал для логирования метрик текущего эксперимента
    loss_lst = []
    grad_norms_lst = []

    metrics_dict = {
        "epochs": [],
        "loss": [],
        "directional_clip_loss": [],
        "lpips_loss": [],
        "diversity": [],
    }

    torch.cuda.empty_cache()

    # Фиксация генератора случайных чисел
    torch.manual_seed(cfg.RAND)
    np.random.seed(cfg.RAND)

    # Инициализация моделей и препроцессоров CLIP
    if not CLIP_MODELS_LOADED:
        print("Загрузка моделей и препроцессоров CLIP...")
        models = load_clip_models(
            model_names=cfg.model_names,
            device=device,
            weights=cfg.weights,
            flag_clip=cfg.flag_clip,
        )
        CLIP_MODELS_LOADED = True
        GLOBAL_CLIP_MODELS = models
    else:
        print("Используем ранее загруженные модели CLIP.")
        models = GLOBAL_CLIP_MODELS

    # Инициализация генераторов
    print("Загрузка генераторов StyleGAN2...")
    generator, generator_frozen = load_stylegan2_generator(device=device)

    # Создание латента для генерации визуализации
    latent_sample = torch.randn(cfg.sample_size, generator.z_dim, device=device)

    # Создание латента для вычисления diversity
    latent_div = torch.randn(cfg.div_size, generator.z_dim, device=device)

    # Создание промптов
    prompts_tgt = make_prompt_list(
        word=cfg.text_target, templates=TEMPLATES, flag_prompt=cfg.flag_prompt
    )

    # Вычисление текстовых фичей (они неизменны)
    text_w_plus_features = preprocessing_text(
        models=models, prompts=prompts_tgt, device=device
    )

    emb_delta_text = delta_text(
        models=models,
        text_target=cfg.text_target,
        text_source=cfg.text_source,
        templates=TEMPLATES,
        device=device,
    )

    # Создание списка блоков generator.synthesis
    lst_block = generator_block(generator=generator)

    if cfg.flag_adaptive_layers:
        optimizer = torch.optim.AdamW(generator.synthesis.parameters(), cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)
    else:
        train_params = unfreeze_layers(
            generator=generator,
            topk_indices=None,
            lst_block=lst_block,
            flag_adaptive_layers=cfg.flag_adaptive_layers,
        )
        optimizer = torch.optim.AdamW(train_params, cfg.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    # Инициализация нейросетевой метрики perceptual loss (LPIPS)
    lpips_metric = lpips_fn(net=cfg.net, device=device)

    # Загрузка checkpoint
    start_epoch = 0
    if cfg.load_dir is not None:
        start_epoch = load_checkpoint(
            generator=generator,
            device=device,
            load_dir=cfg.load_dir,
        )

    epoch = start_epoch
    for epoch in tqdm(range(start_epoch, cfg.epochs)):

        generator.train()

        if cfg.flag_adaptive_layers:
            # Оптимизация векторов из пространства W+ и вычисление топ-k индексов
            topk_indices = optimize_w_plus(
                generator=generator,
                models=models,
                device=device,
                batch_size_w=cfg.batch_size_w,
                lr_w=cfg.lr_w,
                k=cfg.k,
                max_norm=cfg.max_norm,
                text_w_plus_features=text_w_plus_features,
                c=c,
                flag_gradient_clipping=cfg.flag_gradient_clipping,
            )

            # Заморозка всех слоев генератора кроме топ-k conv слоев
            train_params = unfreeze_layers(
                generator=generator,
                topk_indices=topk_indices,
                lst_block=lst_block,
                flag_adaptive_layers=cfg.flag_adaptive_layers,
            )

        # Оптимизация размороженных параметров генератора
        loss, directional_loss, lpips_loss, lst_norms = optimize_generator(
            optimizer=optimizer,
            generator=generator,
            generator_frozen=generator_frozen,
            models=models,
            device=device,
            lst_block=lst_block,
            train_params=train_params,
            emb_delta_text=emb_delta_text,
            lpips_metric=lpips_metric,
            lambda_patch=cfg.lambda_patch,
            lambda_l2=cfg.lambda_l2,
            lambda_lpips=cfg.lambda_lpips,
            max_norm=cfg.max_norm,
            patch_size=cfg.patch_size,
            num_patches=cfg.num_patches,
            batch_size=cfg.batch_size,
            c=c,
            flag_reg_l2=cfg.flag_reg_l2,
            flag_reg_patch=cfg.flag_reg_patch,
            flag_reg_lpips=cfg.flag_reg_lpips,
            flag_adaptive_layers=cfg.flag_adaptive_layers,
            flag_gradient_clipping=cfg.flag_gradient_clipping,
        )

        loss_lst.append(loss.item())
        grad_norms_lst.append(lst_norms)

        scheduler.step()

        print(f"Эпоха {epoch + 1}: loss={loss.item():.4f}")

        # Генерация, сохранение и визуализация изображения, логирование метрик
        if (epoch + 1) % 50 == 0:

            # Сохранение checkpoint
            save_checkpoint(
                generator=generator,
                epoch=epoch,
                path=f"{ckpt_dir}/checkpoint_{epoch + 1}.pt",
            )

            generator.eval()
            with torch.no_grad():
                image_sample = generator(
                    latent_sample, c, truncation_psi=0.7, noise_mode="const"
                )

            # Визуализация и сохранение изображения
            grid = visual_image(image_sample=image_sample)
            save_image(grid, f"{images_dir}/epoch_{epoch + 1}.jpg")

            diversity = diversity_fn(
                generator=generator,
                latent_div=latent_div,
                lpips_metric=lpips_metric,
                c=c,
            )

            # Логирование метрик
            log_metrics(
                metrics_dict,
                epoch=epoch + 1,
                loss=loss.item(),
                directional_clip_loss=directional_loss.item(),
                lpips_loss=lpips_loss.item(),
                diversity=diversity,
            )

            # Удаление отработанных torch.Tensor
            del image_sample, grid, loss, directional_loss, lpips_loss, diversity

            torch.cuda.empty_cache()

    # Усреднение нормы градиентов по всем эпохам
    mean_norms = np.mean(grad_norms_lst, axis=0)

    # Сохранение нормы градиентов текущего эксперимента в pd.DataFrame
    df_grads = pd.DataFrame(
        {
            "block": lst_block,
            "grad_norm_mean": mean_norms,
            "experiment_name": cfg.experiment_name,
        }
    )

    # Сохранение метрик текущего эксперимента в pd.DataFrame
    df_metrics = pd.DataFrame(metrics_dict)
    df_metrics["experiment_name"] = cfg.experiment_name
    df_metrics["seed"] = cfg.RAND
    print(df_metrics.to_string(index=False))
    df_metrics.to_csv(f"{cfg.save_dir}/metrics.csv", index=False)

    # Визуализация поведения лосса и сохранение графика
    plt.figure(figsize=(8, 8))
    plt.plot(loss_lst)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss ({cfg.experiment_name})")
    plt.grid(True)
    plt.savefig(f"{cfg.save_dir}/loss.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("Сохранение финальной модели...")

    save_checkpoint(
        generator=generator, epoch=epoch, path=f"{ckpt_dir}/checkpoint_final.pt"
    )
    torch.save(generator.state_dict(), f"{cfg.save_dir}/generator_final.pt")

    return df_metrics, df_grads, grad_norms_lst
