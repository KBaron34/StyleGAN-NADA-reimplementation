import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_grad_norms(generator, lst_block: list[str]) -> list:
    """
    Вычисляет L2-норму градиентов для каждого блока генератора.

    :param generator: обучаемый генератор StyleGAN2
    :param lst_block: список блоков generator.synthesis (из generator_block)

    :return: список L2 норм градиентов для каждого блока на текущей эпохе
    """
    lst_norms = []

    for block_name in lst_block:
        block = getattr(generator.synthesis, block_name)
        total_norm = 0.0

        for param in block.parameters():
            if param.requires_grad and param.grad is not None:
                param_norm = param.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm**0.5

        lst_norms.append(total_norm)

    return lst_norms


def plot_metric(
    df_metrics: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    save_dir: str,
    marker: str = "o",
    figsize: tuple = (8, 6),
) -> None:
    """
    Универсальная функция для построения и сохранения графиков метрик по экспериментам.

    :param df_metrics: pd.DataFrame с колонками ['experiment_name', 'epoch', metric]
    :param metric: название колонки с метрикой
    :param ylabel: подпись оси Y
    :param title: название графика
    :param save_dir: основная директория для сохранения
    :param marker: маркер для точек на графике
    :param figsize: размеры графика

    :return: None
    """
    if df_metrics.empty:
        print("pd.DataFrame пуст — нечего рисовать")
    else:
        sns.lineplot(
            data=df_metrics,
            x="epochs",
            y=metric,
            hue="experiment_name",
            marker=marker,
            markersize=6,
            linewidth=2,
            err_style=None,
        )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel if ylabel else metric)
        plt.title(title if title else metric)
        plt.legend(title="Experiment")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()

    # Сохранение
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"{title.replace(' ', '_')}.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_all_metrics(df_metrics: pd.DataFrame, save_dir: str) -> None:
    """
    Строит и сохраняет графики метрик по экспериментам.

    :param df_metrics: pd.DataFrame с колонками ['experiment_name', 'epoch', metric]
    :param save_dir: директория для сохранения графика

    :return: None
    """
    lst_metrics = [
        ("loss", "Loss", "Loss over Epochs"),
        (
            "directional_clip_loss",
            "Directional CLIP Loss",
            "Directional CLIP Loss over Epochs",
        ),
        ("lpips_loss", "LPIPS", "LPIPS Stability"),
        ("diversity", "Diversity (LPIPS)", "Generator Diversity"),
    ]

    for metric, ylabel, title in lst_metrics:
        plot_metric(
            df_metrics=df_metrics,
            metric=metric,
            ylabel=ylabel,
            title=title,
            save_dir=save_dir,
            marker="o",
            figsize=(8, 6),
        )


def barplot_mean_grad_norms(
    df_grads: pd.DataFrame,
    save_dir: str,
    title: str = "Mean grad norms per block",
    figsize: tuple = (12, 5),
) -> None:
    """
    Строит barplot норм градиентов.

    :param df_grads: pd.DataFrame с колонками ['block', 'grad_norm_mean', 'experiment_name']
    :param save_dir: директория для сохранения графика
    :param title: заголовок графика
    :param figsize: размеры графика

    :return: None
    """
    if df_grads.empty:
        print("Данных нет")
    else:
        plt.figure(figsize=figsize)

        sns.barplot(
            data=df_grads,
            x="block",
            y="grad_norm_mean",
            hue="experiment_name",
            palette="viridis",
            edgecolor="black",
            linewidth=0.5,
        )

        # Настройка внешнего вида
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("L2 grad norm (mean)")
        plt.xlabel("Block")
        plt.title(title)
        plt.legend(title="Experiment")
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        # Сохранение
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = "grad_norms_mean.png"
            file_path = os.path.join(save_dir, file_name)
            plt.savefig(file_path, dpi=300, bbox_inches="tight")

        plt.show()


def heatmap_block_training(
    df_grads: pd.DataFrame,
    grad_norms_lst: list[list],
    save_dir: str,
    title: str = "Block Training Heatmap (Gradient Norms)",
    figsize: tuple = (16, 8),
) -> None:
    """
    Визуализирует активность обучения блоков генератора по эпохам.
    Heatmap показывает L2 норму градиентов (в логарифмической шкале):
    - Яркие участки - блок активно обучается;
    - Темные участки - блок не обучается.

    :param df_grads: pd.DataFrame с колонками ['block', 'grad_norm_mean', 'experiment_name']
    :param grad_norms_lst: список списков норм градиентов для каждого блока каждой эпохи
    :param save_dir: директория для сохранения графика
    :param title: заголовок графика
    :param figsize: размеры графика

    :return: None
    """
    grad_norms = np.array(grad_norms_lst)
    n_epochs = grad_norms.shape[0]

    plt.figure(figsize=figsize)

    # Формирование heatmap в логарифмической шкале
    im = plt.imshow(
        np.log10(grad_norms.T + 1e-10),
        aspect="auto",
        cmap="viridis",
    )

    # Настройка цветовой шкалы справа от heatmap
    plt.colorbar(im, label="log₁₀(Grad Norm) | Dark: inactive, Light: active")

    blocks = df_grads["block"].tolist()
    plt.yticks(range(len(blocks)), blocks)

    # Настройка подписей оси X (Epoch)
    step = max(1, n_epochs // 10)
    plt.xticks(
        range(0, n_epochs, step), range(1, n_epochs + 1, step), rotation=45, ha="right"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Block")
    plt.title(title)
    plt.tight_layout()

    # Сохранение
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_name = "block_training_heatmap.png"
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches="tight")

    plt.show()


def full_analysis(
    df_metrics: pd.DataFrame,
    df_grads: pd.DataFrame,
    grad_norms_lst: list[list],
    save_dir: str,
    flag_heatmap: bool = True,
) -> None:
    """
    Строит все графики эксперимента

    :param df_metrics: pd.DataFrame с колонками ['experiment_name', 'epoch', metric]
    :param df_grads: pd.DataFrame с колонками ['block', 'grad_norm_mean', 'experiment_name']
    :param grad_norms_lst: список списков норм градиентов для каждого блока каждой эпохи
    :param save_dir: директория для сохранения графика
    :param flag_heatmap: флаг построения heatmap

    :return: None
    """
    # Построение и сохранение графиков метрик по экспериментам
    plot_all_metrics(
        df_metrics=df_metrics,
        save_dir=save_dir,
    )

    # Построение barplot норм градиентов
    barplot_mean_grad_norms(
        df_grads=df_grads,
        save_dir=save_dir,
        title="Mean grad norms per block",
        figsize=(12, 5),
    )

    # Построение Heatmap визуализирующую обучение блоков генератора по эпохам
    if flag_heatmap:
        heatmap_block_training(
            df_grads=df_grads,
            grad_norms_lst=grad_norms_lst,
            save_dir=save_dir,
            title="Block Training Heatmap (Gradient Norms)",
            figsize=(12, 5),
        )
