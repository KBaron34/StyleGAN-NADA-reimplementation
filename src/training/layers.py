import torch

from src.utils.freeze import freeze_module


def get_conv_layer(generator, lst_block: list[str], ind: int) -> torch.nn.Module:
    """
    Возвращает сверточный слой генератора StyleGAN2 индексу.

    :param generator: обучаемый генератор StyleGAN2
    :param lst_block: список блоков generator.synthesis
    :param ind: индекс сверточного слоя

    :return: torch.nn.Module соответствующего conv слоя
    """
    block_idx = ind // 2
    conv_num = ind % 2

    block_name = lst_block[block_idx]
    block = getattr(generator.synthesis, block_name)

    if block_name == "b4":
        return block.conv1

    return block.conv0 if conv_num == 0 else block.conv1


def unfreeze_conv(conv) -> None:
    """
    Размораживает параметры (weight и bias (если есть)) сверточного слоя.

    :param conv: сверточный слой

    :return: None
    """
    conv.weight.requires_grad = True
    if getattr(conv, "bias", None) is not None:
        conv.bias.requires_grad = True


def unfreeze_layers(
    generator,
    topk_indices: torch.Tensor,
    lst_block: list[str],
    flag_adaptive_layers: bool = True,
) -> list:
    """
    Замораживает synthesis генератора и размораживает сверточные слои для обучения.

    :param generator: обучаемый генератор StyleGAN2
    :param topk_indices: torch.Tensor топ-k индексов
    :param lst_block: список блоков generator.synthesis
    :param flag_adaptive_layers: флаг использования адаптивного выбора обучаемых слоев

    :return: список размороженных параметров генератора
    """
    # Заморозка synthesis generator
    freeze_module(generator.synthesis)

    if flag_adaptive_layers:
        for ind in topk_indices:
            conv = get_conv_layer(generator, lst_block, ind)
            unfreeze_conv(conv)
    else:
        for block_name in lst_block:
            block = getattr(generator.synthesis, block_name)
            if hasattr(block, "conv0"):
                unfreeze_conv(block.conv0)
            unfreeze_conv(block.conv1)

    train_params = [param for param in generator.parameters() if param.requires_grad]

    return train_params
