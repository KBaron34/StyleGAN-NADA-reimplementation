import os
import sys
from pathlib import Path

import pandas as pd
import torch

# StyleGAN2-ADA подключён как git submodule в корне проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLEGAN_PATH = PROJECT_ROOT / "stylegan2-ada-pytorch"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(STYLEGAN_PATH))

os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"

from configs.config import TrainConfig
from configs.experiments import get_experiments
from src.training.trainer import train_generator
from src.analysis.plots import full_analysis


def run_experiments(save_dir: str, device) -> pd.DataFrame:
    """
    Запускает серию экспериментов с генератором StyleGAN2 и CLIP,
    используя различные конфигурации флагов, и сохраняет метрики и визуализации.

    :param save_dir: директория для сохранения результатов
    :param device: устройство для вычислений (CPU или CUDA)

    :return: pd.DataFrame с метриками всех экспериментов
    """
    all_metrics = []
    all_grads = []

    os.makedirs(save_dir, exist_ok=True)

    experiments = get_experiments()

    for exp in experiments:
        print(f"Запуск эксперимента: {exp.name}.")

        # Путь сохранения результатов текущего эксперимента
        exp_dir = os.path.join(save_dir, exp.name)
        os.makedirs(exp_dir, exist_ok=True)

        cfg = TrainConfig(
            experiment_name=exp.name,
            save_dir=exp_dir,
            epochs=301,
            flag_clip=exp.flag_clip,
            flag_prompt=exp.flag_prompt,
            flag_reg_l2=exp.flag_reg_l2,
            flag_reg_patch=exp.flag_reg_patch,
            flag_reg_lpips=exp.flag_reg_lpips,
            flag_adaptive_layers=exp.flag_adaptive_layers,
            flag_gradient_clipping=exp.flag_gradient_clipping,
        )

        df_metrics, df_grads, grad_norms_lst = train_generator(cfg=cfg, device=device)

        # Сохранение метрик
        df_metrics.to_csv(os.path.join(exp_dir, "df_metrics.csv"), index=False)
        df_grads.to_csv(os.path.join(exp_dir, "df_grads.csv"), index=False)

        # Построение графиков
        full_analysis(
            df_metrics=df_metrics,
            df_grads=df_grads,
            grad_norms_lst=grad_norms_lst,
            save_dir=exp_dir,
            flag_heatmap=True,
        )

        all_metrics.append(df_metrics)
        all_grads.append(df_grads)

    # Формирование и сохранение общего pd.DataFrame с метриками всех экспериментов
    df_all_metrics = pd.concat(all_metrics, ignore_index=True)
    df_all_grads = pd.concat(all_grads, ignore_index=True)

    # Создание директории для сохранения общих результатов
    full_exp_dir = os.path.join(save_dir, "full_exp")
    os.makedirs(full_exp_dir, exist_ok=True)

    # Сохранение общих pd.DataFrame
    df_all_metrics.to_csv(os.path.join(full_exp_dir, "df_all_metrics.csv"), index=False)
    df_all_grads.to_csv(os.path.join(full_exp_dir, "df_all_grads.csv"), index=False)

    # Построение графиков
    full_analysis(
        df_metrics=df_all_metrics,
        df_grads=df_all_grads,
        grad_norms_lst=None,
        save_dir=full_exp_dir,
        flag_heatmap=False,
    )

    print("Все эксперименты завершены!")
    print(f"Результаты отдельных экспериментов: {save_dir}/<exp_name>.")
    print(f"Общие результаты: {full_exp_dir}.")

    return df_all_metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_experiments(save_dir="results", device=device)


if __name__ == "__main__":
    main()
