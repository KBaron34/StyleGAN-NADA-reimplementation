import copy
import pickle

from pathlib import Path

from src.utils.freeze import freeze_module, unfreeze_module


def load_stylegan2_generator(device, ckpt_path: str = "/content/StyleGAN2/ffhq.pkl"):
    """
    Инициализирует обучаемый генератор StyleGAN2 с замороженным mapping
    и создает его полностью замороженную копию.

    :param device: устройство для вычислений (CPU или CUDA)
    :param ckpt_path: путь к файлу предобученной модели (.pkl)

    :return: обучаемый и замороженный генераторы StyleGAN2
    """
    if ckpt_path is None:
        project_root = Path(__file__).resolve().parent.parent
        ckpt_path = project_root / "StyleGAN2" / "ffhq.pkl"

    with open(ckpt_path, "rb") as f:
        generator = pickle.load(f)["G_ema"].to(device)

    # Создание копии генератора
    generator_frozen = copy.deepcopy(generator)

    generator.train()
    generator_frozen.eval()

    # Заморозка generator_frozen
    freeze_module(generator_frozen)

    # Заморозка mapping generator
    freeze_module(generator.mapping)

    # Разморозка synthesis generator
    unfreeze_module(generator.synthesis)

    return generator, generator_frozen


def generator_block(generator) -> list[str]:
    """
    Создает список блоков generator.synthesis.

    :param generator: обучаемый генератор StyleGAN2

    :return: список имен блоков generator.synthesis
    """
    lst_block = []

    for name, module in generator.synthesis.named_children():
        if name.startswith("b"):
            lst_block.append(name)

    return lst_block
