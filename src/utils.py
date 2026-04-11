import os
import sys
from pathlib import Path

import torch
import numpy as np

# Кореневой каталог проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Путь к StyleGAN2
STYLEGAN2_PATH = PROJECT_ROOT / "stylegan2-ada-pytorch"

# Добавляем StyleGAN2 в sys.path
if STYLEGAN2_PATH.exists():
    sys.path.append(str(STYLEGAN2_PATH))

# Настройка CUDA
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"


def set_seed(rand: int = 42) -> None:
    """
    Фиксирует генератор случайных чисел для воспроизводимости результатов.

    :param rand: начальное значение для генератора случайных чисел

    :return: None
    """
    torch.manual_seed(rand)  # фиксирует случайность в PyTorch
    np.random.seed(rand)  # фиксирует случайность в NumPy


def get_device():
    """
    Возвращает 'cuda' если есть GPU, иначе 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
