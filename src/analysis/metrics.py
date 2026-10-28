def log_metrics(
    metrics_dict: dict,
    epoch: int,
    loss: float,
    directional_clip_loss: float,
    lpips_loss: float,
    diversity: float,
) -> None:
    """
    Логирует метрики текущей эпохи.

    :param metrics_dict: словарь с метриками, хранящий все эпохи
    :param epoch: номер эпохи
    :param loss: значение основной функции потерь
    :param directional_clip_loss: directional CLIP loss
    :param lpips_loss: perceptual loss (LPIPS)
    :param diversity: diversity (разнообразие) генератора

    :return: None
    """
    # Добавление значений в словарь
    metrics_dict["epochs"].append(epoch)
    metrics_dict["loss"].append(loss)
    metrics_dict["directional_clip_loss"].append(directional_clip_loss)
    metrics_dict["lpips_loss"].append(lpips_loss)
    metrics_dict["diversity"].append(diversity)
