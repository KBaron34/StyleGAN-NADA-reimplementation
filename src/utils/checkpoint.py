import os
import torch


def save_checkpoint(generator, epoch: int, path: str) -> None:
    """
    Сохраняет состояние обучения.

    :param generator: обучаемый генератор StyleGAN2
    :param epoch: текущая эпоха
    :param path: путь для сохранения

    :return: None
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "generator": generator.state_dict(),
    }

    torch.save(checkpoint, path)
    print(f"Checkpoint сохранен: {path}")


def load_checkpoint(
    generator,
    device,
    load_dir: str,
) -> int:
    """
    Загружает состояние обучения.

    :param generator: обучаемый генератор StyleGAN2
    :param device: устройство для вычислений (CPU или CUDA)
    :param load_dir: путь к файлу checkpoint

    :return: текущая эпоха
    """
    checkpoint = torch.load(load_dir, map_location=device)

    generator.load_state_dict(checkpoint["generator"])

    epoch = checkpoint.get("epoch", 0)

    print(f"Checkpoint загружен: {load_dir} (epoch {epoch})")

    return epoch
